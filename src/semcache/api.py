"""FastAPI service in front of the cache (SPEC section 14)."""

from __future__ import annotations

import hmac
import json
import logging

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import BaseModel

from semcache.cache import SemanticCache
from semcache.types import ChatRequest, ChatResponse

log = logging.getLogger("semcache.api")


class InvalidateRequest(BaseModel):
    scope_key: str | None = None
    namespace: str | None = None
    entry_id: str | None = None


def create_app(cache: SemanticCache, *, admin_token: str = "") -> FastAPI:
    app = FastAPI(
        title="semantic-cache",
        version="0.1.0",
        summary="Embedding-similarity cache in front of Amazon Bedrock chat calls.",
    )

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        # Not real auth (a non-goal for v1); it stops accidental use of destructive endpoints.
        if admin_token and not hmac.compare_digest(x_admin_token or "", admin_token):
            raise HTTPException(status_code=401, detail="missing or wrong X-Admin-Token")

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        if request.stream:
            return StreamingResponse(_sse(cache, request), media_type="text/event-stream")
        try:
            return await cache.aget_or_call(request)
        except Exception as exc:
            log.exception("upstream call failed")
            raise HTTPException(
                status_code=502, detail=f"upstream error: {type(exc).__name__}"
            ) from None

    @app.get("/metrics")
    def metrics():
        return Response(cache.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "store": cache.store.name}

    @app.post("/admin/invalidate", dependencies=[Depends(require_admin)])
    def invalidate(body: InvalidateRequest):
        try:
            deleted = cache.invalidate(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"deleted": deleted}

    @app.post("/admin/purge-expired", dependencies=[Depends(require_admin)])
    def purge_expired():
        return {"deleted": cache.purge_expired()}

    @app.get("/admin/stats", dependencies=[Depends(require_admin)])
    def stats():
        return {"store": cache.store.stats(), "traffic": cache.metrics.snapshot()}

    return app


async def _sse(cache: SemanticCache, request: ChatRequest):
    """Server-Sent Events: {"delta": ...} chunks, then {"done": true, ...metadata}."""
    try:
        async for event in cache.astream(request):
            if "delta" in event:
                yield f"data: {json.dumps({'delta': event['delta']})}\n\n"
            else:
                meta = event["response"].model_dump(mode="json", exclude={"text"})
                yield f"data: {json.dumps({'done': True, **meta})}\n\n"
    except Exception as exc:
        log.exception("stream failed")
        yield f"data: {json.dumps({'error': f'upstream error: {type(exc).__name__}'})}\n\n"
