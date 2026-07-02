from __future__ import annotations

import asyncio
import ctypes
import inspect
import logging
import time
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
TTL_SECONDS: int = 3600  # 1 hora en reposo antes de descargar


def force_garbage_collection() -> None:
    """Fuerza a la biblioteca C (glibc) a liberar y devolver las arenas
    de memoria física (RSS) no utilizadas de vuelta al kernel.
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        result = libc.malloc_trim(0)
        if result == 1:
            logger.info("Memoria física (RSS) devuelta al sistema operativo exitosamente.")
    except Exception as e:
        logger.warning("No se pudo ejecutar malloc_trim de manera nativa: %s", str(e))


def update_engine_activity() -> None:
    """Actualiza el timestamp de última actividad."""
    global _last_active_time
    _last_active_time = time.time()


async def _monitor_inactivity() -> None:
    """Loop en segundo plano que descarga el modelo si expira el TTL."""
    global _engine
    while _engine is not None:
        await asyncio.sleep(15)  # Verificación más frecuente para precisión
        
        async with _engine_lock:
            if _engine is None:
                break
            
            elapsed = time.time() - _last_active_time
            if elapsed >= TTL_SECONDS:
                logger.info("TTL de inactividad alcanzado (%ds). Descargando LiteRT de la RAM...", TTL_SECONDS)
                
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
        if "max_num_images" in engine_signature.parameters:
            engine_kwargs["max_num_images"] = settings.max_num_images
        if settings.max_num_images > 0 and "vision_backend" in engine_signature.parameters:
            engine_kwargs["vision_backend"] = Backend.CPU()

        _engine = await asyncio.to_thread(Engine, settings.model_path, **engine_kwargs)
        logger.info("LiteRT engine initialized")
        
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