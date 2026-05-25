from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.agents.research_agent import run_agent
from app.config.logging import get_logger, log_event
from app.utils.core_utils.guard import check_query_safety
from app.health import router as health_router
import time


def create_app() -> FastAPI:
    app = FastAPI()

    app.include_router(health_router)

    logger = get_logger("api.main")

    @app.get("/ask")
    async def ask(q: str):
        started = time.perf_counter()
        safe, reason = check_query_safety(q)

        if not safe:
            log_event(logger, "query_blocked", query=q, reason=reason)

            async def blocked_stream():
                yield "Request blocked by safety guard.\n"
                yield reason

            return StreamingResponse(blocked_stream(), media_type="text/plain")

        log_event(logger, "request_started", query=q)

        async def stream():
            try:
                async for token in run_agent(q):
                    yield token
                duration_ms = int((time.perf_counter() - started) * 1000)
                log_event(logger, "request_completed", query=q, latency_ms=duration_ms)
            except Exception as e:
                yield f"\n\n[stream_error] {type(e).__name__}: {e}\n"
                duration_ms = int((time.perf_counter() - started) * 1000)
                log_event(
                    logger,
                    "request_failed",
                    query=q,
                    latency_ms=duration_ms,
                    error=type(e).__name__,
                )

        return StreamingResponse(stream(), media_type="text/plain")

    return app
