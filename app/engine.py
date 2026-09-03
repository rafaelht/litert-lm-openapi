from __future__ import annotations

import asyncio
import ctypes
import gc
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from litert_lm import Backend, Engine

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_engine_lock = asyncio.Lock()
_engine_just_reloaded: bool = False

# Variables para control de TTL
_last_active_time: float = 0.0
_cleanup_task: Optional[asyncio.Task] = None

_libc = None
if sys.platform == "linux":
    try:
        _libc = ctypes.CDLL("libc.so.6")
    except Exception:
        _libc = None


def force_garbage_collection() -> None:
    """Fuerza al recolector de basura de Python y a glibc (en Linux) a devolver memoria al OS."""
    gc.collect()
    if _libc is not None:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass


def update_engine_activity() -> None:
    """Actualiza el timestamp de última actividad."""
    global _last_active_time
    _last_active_time = time.time()


async def _monitor_inactivity() -> None:
    """Loop en segundo plano que descarga el modelo si expira el TTL."""
    global _engine
    settings = get_settings()
    ttl = max(60, settings.session_timeout)

    while _engine is not None:
        await asyncio.sleep(15)  # Verificación periódica para precisión
        
        async with _engine_lock:
            if _engine is None:
                break
            
            elapsed = time.time() - _last_active_time
            if elapsed >= ttl:
                logger.info("TTL de inactividad alcanzado (%ds). Descargando LiteRT de la RAM...", ttl)
                
                engine = _engine
                _engine = None
                await asyncio.to_thread(engine.close)
                logger.info("LiteRT engine liberado automáticamente por inactividad.")
                
                force_garbage_collection()
                break


async def init_engine() -> Engine:
    """Inicializa de manera segura el motor garantizando concurrencia idempotente."""
    global _engine, _cleanup_task, _engine_just_reloaded

    if _engine is not None:
        update_engine_activity()
        return _engine

    async with _engine_lock:
        if _engine is not None:
            update_engine_activity()
            return _engine

        settings = get_settings()
        logger.info("Initializing LiteRT engine with model at %s", settings.model_path)
        engine_kwargs: dict[str, object] = {}
        engine_signature = inspect.signature(Engine)

        # 1. Configurar Backend CPU calibrado con los núcleos del hardware
        if "backend" in engine_signature.parameters:
            engine_kwargs["backend"] = Backend.CPU(thread_count=settings.cpu_threads)

        # 2. Límite del KV Cache en C++
        if "max_num_tokens" in engine_signature.parameters and settings.max_num_tokens > 0:
            engine_kwargs["max_num_tokens"] = settings.max_num_tokens

        # 3. Soporte multimodal controlado
        if "max_num_images" in engine_signature.parameters:
            engine_kwargs["max_num_images"] = settings.max_num_images
        if settings.max_num_images > 0 and "vision_backend" in engine_signature.parameters:
            engine_kwargs["vision_backend"] = Backend.CPU(thread_count=settings.cpu_threads)

        # 4. Ringbuffers para atención local (específico para backend GPU)
        is_gpu = isinstance(engine_kwargs.get("backend"), Backend.GPU)
        if is_gpu and "use_ringbuffers_local_attention" in engine_signature.parameters:
            engine_kwargs["use_ringbuffers_local_attention"] = settings.use_ringbuffers_local_attention

        # 5. Caché en disco de XNNPACK (desactivada por defecto para evitar inflar la RAM)
        if settings.enable_xnnpack_cache:
            cache_dir = os.getenv("CACHE_DIR")
            if not cache_dir:
                model_path_obj = Path(settings.model_path)
                model_dir = (
                    model_path_obj.parent
                    if model_path_obj.is_file() or not model_path_obj.is_dir()
                    else model_path_obj
                )
                if os.access(model_dir, os.W_OK):
                    cache_dir = str(model_dir)
                else:
                    cache_dir = "/tmp/litert_cache"
                    os.makedirs(cache_dir, exist_ok=True)

            if "cache_dir" in engine_signature.parameters and cache_dir:
                engine_kwargs["cache_dir"] = cache_dir

        # 6. Benchmark de LiteRT
        if "enable_benchmark" in engine_signature.parameters:
            engine_kwargs["enable_benchmark"] = settings.enable_benchmark

        _engine = await asyncio.to_thread(Engine, settings.model_path, **engine_kwargs)
        logger.info("LiteRT engine initialized with kwargs: %s", sorted(engine_kwargs.keys()))
        
        _engine_just_reloaded = True
        update_engine_activity()
        
        if _cleanup_task is None or _cleanup_task.done():
            _cleanup_task = asyncio.create_task(_monitor_inactivity())
            
        return _engine


def get_engine() -> Engine:
    """Retorna la instancia actual si existe. Puede retornar None si fue descargado."""
    global _engine
    if _engine is not None:
        update_engine_activity()
    return _engine


def check_and_consume_reload_flag() -> bool:
    """Retorna si el motor se recargó y consume el estado (atómico)."""
    global _engine_just_reloaded
    if _engine_just_reloaded:
        _engine_just_reloaded = False
        return True
    return False


async def close_engine() -> None:
    """Función requerida por app/main.py para liberar recursos al apagar el contenedor."""
    global _engine

    async with _engine_lock:
        if _engine is None:
            return

        engine = _engine
        _engine = None
        logger.info("Closing LiteRT engine")
        await asyncio.to_thread(engine.close)
        logger.info("LiteRT engine closed")
        
        force_garbage_collection()