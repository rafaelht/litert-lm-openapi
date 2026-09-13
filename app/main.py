from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.config import get_settings
from app.conversation_manager import (
    close_conversation_manager,
    get_conversation_manager,
    init_conversation_manager,
)
from app.engine import close_engine, get_current_model_id, init_engine
from app.model_manager import discover_models
from app.openai_routes import router as openai_router
from app.profile_store import get_profile_store, init_profile_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_cleanup_task: asyncio.Task[None] | None = None


async def _cleanup_loop() -> None:
    settings = get_settings()
    manager = get_conversation_manager()
    sleep_for = max(15, min(60, settings.session_timeout // 2 or 15))

    while True:
        try:
            await asyncio.sleep(sleep_for)
            await manager.cleanup_expired()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Conversation cleanup loop failed")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _cleanup_task

    settings = get_settings()
    logger.info("Starting LiteRT Session Server on port %s", settings.server_port)
    logger.info("Models directory: %s", settings.models_dir)
    logger.info("Default model profile: %s", settings.model_profile)

    await init_profile_store()
    await init_conversation_manager(None)
    _cleanup_task = asyncio.create_task(_cleanup_loop())

    if settings.preload_first_model:
        logger.info("PRELOAD_FIRST_MODEL is enabled. Checking models directory...")
        try:
            discovered = discover_models()
            if discovered:
                first_model_id = next(iter(discovered.keys()))
                logger.info("Preloading model '%s' at startup...", first_model_id)
                await init_engine(first_model_id)
                logger.info("Model '%s' successfully preloaded at startup.", first_model_id)
            else:
                logger.warning("PRELOAD_FIRST_MODEL=true, but no models found in %s", settings.models_dir)
        except Exception as exc:
            logger.warning(
                "Could not preload model at startup (%s). Will initialize on first request.",
                exc,
            )
    else:
        logger.info("Lazy-loading enabled: no model preloaded at startup. Initializing on first request.")

    try:
        yield
    finally:
        if _cleanup_task is not None:
            _cleanup_task.cancel()
            try:
                await _cleanup_task
            except asyncio.CancelledError:
                pass
            _cleanup_task = None

        await close_conversation_manager()
        await close_engine()


app = FastAPI(
    title="LiteRT Session Server",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(openai_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/internal/profile")
async def internal_profile() -> dict[str, object]:
    profile_store = get_profile_store()
    manager = get_conversation_manager()
    stats = await manager.stats()
    return {
        "current_model": get_current_model_id(),
        "profile": profile_store.as_debug_dict(),
        "active_conversations": stats["active_conversations"],
        "profile_initialized_conversations": stats["profile_initialized_conversations"],
    }
