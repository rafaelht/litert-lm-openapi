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

_current_model_id: Optional[str] = None
_current_model_path: Optional[str] = None
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
    """Fuerza al recolector de basura de Python y a glibc (en Linux) a devolver memoria física al OS."""
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


def get_current_model_id() -> Optional[str]:
    """Retorna el identificador del modelo actualmente cargado en RAM."""
    global _current_model_id
    return _current_model_id


def _build_engine_kwargs(model_path: Path) -> dict[str, object]:
    """Construye los parámetros optimizados para instanciar el Engine LiteRT."""
    settings = get_settings()
    engine_kwargs: dict[str, object] = {}
    engine_signature = inspect.signature(Engine)

    # 1. Configurar Backend CPU calibrado con los núcleos del hardware
    if "backend" in engine_signature.parameters:
        engine_kwargs["backend"] = Backend.CPU(thread_count=settings.cpu_threads)

    # 2. Asignación del KV Cache en C++ (Dinámica mmap vs Estática)
    if (
        settings.force_static_max_tokens
        and "max_num_tokens" in engine_signature.parameters
        and settings.max_num_tokens > 0
    ):
        engine_kwargs["max_num_tokens"] = settings.max_num_tokens

    # 3. Soporte multimodal estrictamente condicional
    if settings.max_num_images > 0:
        if "max_num_images" in engine_signature.parameters:
            engine_kwargs["max_num_images"] = settings.max_num_images
        if "vision_backend" in engine_signature.parameters:
            engine_kwargs["vision_backend"] = Backend.CPU(thread_count=settings.cpu_threads)

    # 4. Ringbuffers para atención local (específico para backend GPU)
    is_gpu = isinstance(engine_kwargs.get("backend"), Backend.GPU)
    if is_gpu and "use_ringbuffers_local_attention" in engine_signature.parameters:
        engine_kwargs["use_ringbuffers_local_attention"] = settings.use_ringbuffers_local_attention

    # 5. Caché en disco de XNNPACK (Solo si está explícitamente habilitado en config)
    if settings.enable_xnnpack_cache:
        cache_dir = os.getenv("CACHE_DIR")
        if not cache_dir:
            model_dir = (
                model_path.parent
                if model_path.is_file() or not model_path.is_dir()
                else model_path
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

    return engine_kwargs


async def _monitor_inactivity() -> None:
    """Loop en segundo plano que descarga el modelo si expira el TTL (solo si ENGINE_TTL > 0)."""
    global _engine, _current_model_id, _current_model_path
    settings = get_settings()
    ttl = settings.engine_ttl
    if ttl <= 0:
        return

    while _engine is not None:
        await asyncio.sleep(15)

        async with _engine_lock:
            if _engine is None:
                break

            elapsed = time.time() - _last_active_time
            if elapsed >= ttl:
                logger.info("TTL de inactividad alcanzado (%ds). Descargando LiteRT de la RAM...", ttl)

                try:
                    from app.conversation_manager import get_conversation_manager
                    manager = get_conversation_manager()
                    await manager.close_all()
                except Exception:
                    logger.exception("Error cerrando conversaciones antes de liberar engine por TTL")

                engine = _engine
                _engine = None
                _current_model_id = None
                _current_model_path = None
                await asyncio.to_thread(engine.close)
                logger.info("LiteRT engine liberado automáticamente por inactividad.")

                force_garbage_collection()
                break


async def get_or_load_engine(model_id: str | None = None) -> tuple[Engine, str]:
    """
    Retorna el motor LiteRT garantizando exclusión mutua. Si el modelo solicitado difiere
    del cargado actualmente o aún no hay motor en memoria:
    1. Cierra todas las conversaciones y sesiones en C++ activas.
    2. Cierra explícitamente el Engine anterior y destruye referencias.
    3. Ejecuta recolección forzada de basura y trim de memoria.
    4. Carga el perfil YAML del nuevo modelo.
    5. Instancia el nuevo Engine apuntando al archivo .litertlm.
    6. Asigna el nuevo motor al ConversationManager.
    """
    global _engine, _current_model_id, _current_model_path, _engine_just_reloaded, _cleanup_task
    from fastapi import HTTPException
    from app.model_manager import discover_models, resolve_model

    # 1. Determinar el modelo destino
    if model_id:
        resolved = resolve_model(model_id)
        if resolved is None:
            available = list(discover_models().keys())
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found. Available models: {available}",
            )
        target_id = resolved.id
        target_path = resolved.path
    else:
        # Si no se especificó y ya hay uno cargado, reutilizarlo
        if _engine is not None and _current_model_id is not None:
            update_engine_activity()
            return _engine, _current_model_id

        # Si no hay cargado, buscar el primero disponible
        discovered = discover_models()
        if not discovered:
            raise HTTPException(
                status_code=404,
                detail="No LiteRT models found in configured models directory.",
            )
        first_id = next(iter(discovered.keys()))
        target_id = discovered[first_id].id
        target_path = discovered[first_id].path

    # 2. Fast-path sin bloqueo si el modelo solicitado ya está en memoria
    if _engine is not None and _current_model_id == target_id:
        update_engine_activity()
        return _engine, _current_model_id

    # 3. Bloqueo de exclusión mutua para Hot-Swapping / Lazy-Loading seguro
    async with _engine_lock:
        # Doble verificación dentro del lock
        if _engine is not None and _current_model_id == target_id:
            update_engine_activity()
            return _engine, _current_model_id

        # Si hay un modelo diferente cargado, ejecutar limpieza estricta de memoria
        if _engine is not None:
            logger.info(
                "[HOT-SWAP] Descargando modelo actual '%s' para cargar '%s'...",
                _current_model_id,
                target_id,
            )
            try:
                from app.conversation_manager import get_conversation_manager
                manager = get_conversation_manager()
                await manager.close_all()
                logger.info("[HOT-SWAP] Sesiones previas de conversación cerradas en C++.")
            except Exception:
                logger.exception("Error cerrando conversaciones previas en hot-swap")

            old_engine = _engine
            _engine = None
            _current_model_id = None
            _current_model_path = None
            try:
                await asyncio.to_thread(old_engine.close)
                logger.info("[HOT-SWAP] Motor C++ anterior cerrado correctamente.")
            except Exception:
                logger.exception("Error al cerrar motor LiteRT anterior")

            force_garbage_collection()
            logger.info("[HOT-SWAP] Memoria física liberada al sistema operativo.")

        # 4. Cargar perfil específico para el nuevo modelo
        try:
            from app.profile_store import load_profile_for_model
            load_profile_for_model(target_id)
            logger.info("[HOT-SWAP] Perfil YAML cargado para modelo '%s'", target_id)
        except Exception:
            logger.exception("Error cargando perfil para modelo '%s'", target_id)

        # 5. Instanciar nuevo Engine
        logger.info("[HOT-SWAP] Inicializando LiteRT Engine con modelo '%s' en %s", target_id, target_path)
        engine_kwargs = _build_engine_kwargs(target_path)
        _engine = await asyncio.to_thread(Engine, str(target_path), **engine_kwargs)
        _current_model_id = target_id
        _current_model_path = str(target_path)
        _engine_just_reloaded = True
        update_engine_activity()

        # 6. Reenlazar ConversationManager con el nuevo motor
        try:
            from app.conversation_manager import get_conversation_manager
            manager = get_conversation_manager()
            manager.set_engine(_engine)
        except Exception:
            logger.exception("Error enlazando nuevo motor al ConversationManager")

        logger.info(
            "[HOT-SWAP] Modelo '%s' cargado exitosamente en RAM (kwargs: %s)",
            target_id,
            sorted(engine_kwargs.keys()),
        )

        if _cleanup_task is None or _cleanup_task.done():
            _cleanup_task = asyncio.create_task(_monitor_inactivity())

        return _engine, _current_model_id


async def init_engine(model_id: str | None = None) -> Engine:
    """Inicializa de manera segura el motor garantizando concurrencia idempotente."""
    engine, _ = await get_or_load_engine(model_id)
    return engine


def get_engine() -> Engine | None:
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
    """Función para liberar recursos al apagar el contenedor o forzar descarga."""
    global _engine, _current_model_id, _current_model_path

    async with _engine_lock:
        if _engine is None:
            return

        engine = _engine
        _engine = None
        _current_model_id = None
        _current_model_path = None
        logger.info("Closing LiteRT engine")
        try:
            await asyncio.to_thread(engine.close)
        except Exception:
            logger.exception("Error closing LiteRT engine")
        logger.info("LiteRT engine closed and memory freed")

        force_garbage_collection()