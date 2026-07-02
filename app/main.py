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
from app.engine import close_engine, init_engine
from app.openai_routes import router as openai_router

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

    engine = await init_engine()
    await init_conversation_manager(engine)
    _cleanup_task = asyncio.create_task(_cleanup_loop())

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
