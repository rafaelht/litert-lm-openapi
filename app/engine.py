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
_libjemalloc = None
if sys.platform == "linux":
    try:
        _libc = ctypes.CDLL("libc.so.6")
    except Exception:
        _libc = None

    for libname in [
        "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
        "libjemalloc.so.2",
        "libjemalloc.so",
    ]:
        try:
            _libjemalloc = ctypes.CDLL(libname)
            break
        except Exception:
            pass


def force_garbage_collection() -> None:
    """Fuerza al recolector de basura de Python, jemalloc y glibc a devolver memoria física (RSS) al kernel."""
    gc.collect()
    if _libjemalloc is not None:
        try:
            _libjemalloc.mallctl(b"arenas.purge", None, None, None, 0)
        except Exception:
            pass
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
    """Loop en segundo plano que descarga el modelo si expira el TTL (solo si ENGINE_TTL > 0)."""
    global _engine
    settings = get_settings()
    ttl = settings.engine_ttl
    if ttl <= 0:
        # Por defecto, el motor permanece cargado en RAM permanentemente para máxima velocidad
        # y para evitar fugas de memoria por recargas cíclicas de C++.
        return

    while _engine is not None:
        await asyncio.sleep(15)

        async with _engine_lock:
            if _engine is None:
                break

            elapsed = time.time() - _last_active_time
            if elapsed >= ttl:
                logger.info("TTL de inactividad alcanzado (%ds). Descargando LiteRT de la RAM...", ttl)

                # CRÍTICO: Cerrar todas las sesiones de conversación en C++ antes de destruir el engine.
                # De lo contrario, C++ lanza 'EngineAdvancedImpl destructed with living sessions!'
                # y retiene la memoria antigua en RAM provocando que la RAM suba a 2.6GB.
                try:
                    from app.conversation_manager import get_conversation_manager
                    manager = get_conversation_manager()
                    await manager.close_all()
                except Exception:
                    logger.exception("Error cerrando conversaciones antes de liberar engine")

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

        # 2. Asignación del KV Cache en C++ (Dinámica mmap vs Estática)
        # NOTA DE RENDIMIENTO Y MEMORIA:
        # Si se pasa max_num_tokens a Engine.__init__, LiteRT-LM reescribe y reasigna
        # 1.424 tensores en C++ (magic_number_utils), rompiendo el mmap de solo lectura
        # del archivo y precargando 1.9GB de RAM.
        # Por defecto (force_static_max_tokens=False), LiteRT usa su gestión dinámica mmap
        # manteniendo la memoria física en solo ~400MB-500MB. El rollover a 4096 tokens
        # se gestiona de forma continua y limpia en Python sin inflar la RAM.
        if (
            settings.force_static_max_tokens
            and "max_num_tokens" in engine_signature.parameters
            and settings.max_num_tokens > 0
        ):
            engine_kwargs["max_num_tokens"] = settings.max_num_tokens

        # 3. Soporte multimodal estrictamente condicional:
        # NUNCA pasar max_num_images ni vision_backend si max_num_images <= 0,
        # para que LiteRT NO compile ni instancie las 3 resoluciones de vision
        # (vision_140, vision_280, vision_70), los adaptadores y audio, lo cual ahorra > 1GB de RAM.
        if settings.max_num_images > 0:
            if "max_num_images" in engine_signature.parameters:
                engine_kwargs["max_num_images"] = settings.max_num_images
            if "vision_backend" in engine_signature.parameters:
                engine_kwargs["vision_backend"] = Backend.CPU(thread_count=settings.cpu_threads)

        # 4. YNNPACK (aceleración CPU nativa de LiteRT)
        if "enable_ynnpack" in engine_signature.parameters:
            engine_kwargs["enable_ynnpack"] = True

        # 5. Ringbuffers para atención local (específico para backend GPU)
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