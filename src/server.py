"""FastAPI application assembly for TTS Inference API."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from pydantic import ValidationError as PydanticValidationError

from .auth import AuthMiddleware
from .config import ServerConfig
from .models import (
    ErrorResponse,
    HealthResponse,
    QueueBusyResponse,
    ReferenceAddResponse,
    ReferenceDeleteRequest,
    ReferenceListResponse,
    ReferenceUpdateRequest,
    TTSRequest,
)
from .reference_manager import ConflictError, NotFoundError, ReferenceManager
from .request_queue import QueueFullError, RequestQueue
from .tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# Content-Type mapping per format
CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "pcm": "application/octet-stream",
}

# Module-level state (populated during lifespan)
_engines: list[TTSEngine] = []
_queue: RequestQueue | None = None
_ref_manager: ReferenceManager | None = None
_worker_tasks: list[asyncio.Task] = []


def create_app(config: ServerConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engines, _queue, _ref_manager, _worker_tasks

        # Validate model path
        model_path = Path(config.model_path)
        if not model_path.exists():
            logger.error(f"Model path does not exist: {config.model_path}")
            print(
                f"ERROR: Model path does not exist: {config.model_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        num_workers = max(1, config.num_workers)

        # Print startup configuration
        print(f"Starting TTS Inference API Server")
        print(f"  Host:           {config.host}:{config.port}")
        print(f"  Model path:     {config.model_path}")
        print(f"  Workers:        {num_workers}")
        print(f"  Max queue size: {config.max_queue_size}")
        print(f"  Timeout:        {config.timeout}s")
        print(f"  API key:        {'configured' if config.api_key else 'not configured'}")

        # Load one engine per worker
        logger.info(f"Loading {num_workers} TTS engine instance(s)...")
        _engines = [TTSEngine(config.model_path) for _ in range(num_workers)]

        # Create queue with engine pool and reference manager
        _queue = RequestQueue(engines=_engines, max_size=config.max_queue_size, timeout=config.timeout)
        _ref_manager = ReferenceManager()

        # Start background workers
        _worker_tasks = _queue.start_workers()

        yield

        # Shutdown
        for task in _worker_tasks:
            task.cancel()
        for task in _worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="TTS Inference API", lifespan=lifespan)

    # Register auth middleware
    app.add_middleware(AuthMiddleware, api_key=config.api_key)

    # ── Exception handlers ────────────────────────────────────────────────────

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc.errors())},
        )

    @app.exception_handler(PydanticValidationError)
    async def pydantic_exception_handler(request: Request, exc: PydanticValidationError):
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(exc.errors())},
        )

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, exc: QueueFullError):
        return JSONResponse(
            status_code=503,
            content={"status": "busy", "message": str(exc)},
        )

    @app.exception_handler(asyncio.TimeoutError)
    async def timeout_handler(request: Request, exc: asyncio.TimeoutError):
        return JSONResponse(
            status_code=408,
            content={"success": False, "message": "Request timeout"},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception during request processing")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "Internal server error"},
        )

    # ── Health endpoint ───────────────────────────────────────────────────────

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    # ── TTS endpoint ──────────────────────────────────────────────────────────

    @app.post("/v1/tts")
    async def tts(req: TTSRequest):
        # Streaming + non-WAV check
        if req.streaming and req.format != "wav":
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Streaming is only supported for WAV format",
                },
            )

        # Resolve reference_id to path if provided
        ref_audio_path: str | None = None
        if req.reference_id:
            path = _ref_manager.get_path(req.reference_id)
            if path is None:
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "message": f"Reference '{req.reference_id}' not found",
                    },
                )
            ref_audio_path = str(path)

        # Build references list for engine
        references = (
            [r.model_dump() for r in req.references] if req.references else None
        )

        if req.streaming:
            # Streaming response — run inference directly (generator)
            def _run_streaming(engine):
                return engine.generate(
                    text=req.text,
                    references=references,
                    ref_audio_path=ref_audio_path,
                    format="wav",
                    chunk_length=req.chunk_length,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    max_new_tokens=req.max_new_tokens,
                    seed=req.seed,
                    normalize=req.normalize,
                    streaming=True,
                    cfg_scale=req.cfg_scale,
                    flow_steps=req.flow_steps,
                    sigma=req.sigma,
                    instruct=req.instruct,
                )

            try:
                audio_gen = await _queue.enqueue(_run_streaming)
            except QueueFullError:
                return JSONResponse(
                    status_code=503,
                    content={"status": "busy", "message": "Request queue is full"},
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=408,
                    content={"success": False, "message": "Request timeout"},
                )
            except Exception:
                logger.exception("Inference error during streaming TTS")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": "Internal server error"},
                )

            return StreamingResponse(audio_gen, media_type="audio/wav")

        else:
            # Non-streaming — enqueue and wait for complete audio bytes
            def _run_inference(engine):
                return engine.generate(
                    text=req.text,
                    references=references,
                    ref_audio_path=ref_audio_path,
                    format=req.format,
                    chunk_length=req.chunk_length,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    max_new_tokens=req.max_new_tokens,
                    seed=req.seed,
                    normalize=req.normalize,
                    streaming=False,
                    cfg_scale=req.cfg_scale,
                    flow_steps=req.flow_steps,
                    sigma=req.sigma,
                    instruct=req.instruct,
                )

            try:
                audio_bytes: bytes = await _queue.enqueue(_run_inference)
            except QueueFullError:
                return JSONResponse(
                    status_code=503,
                    content={"status": "busy", "message": "Request queue is full"},
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=408,
                    content={"success": False, "message": "Request timeout"},
                )
            except Exception:
                logger.exception("Inference error during TTS")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "message": "Internal server error"},
                )

            media_type = CONTENT_TYPE_MAP.get(req.format, "application/octet-stream")
            return Response(content=audio_bytes, media_type=media_type)

    # ── Reference management endpoints ────────────────────────────────────────

    @app.post("/v1/references/add", response_model=ReferenceAddResponse)
    async def references_add(
        id: str = Form(...),
        audio: UploadFile = Form(...),
        text: str = Form(...),
    ):
        if not _ref_manager.validate_id(id):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Invalid reference ID: must be alphanumeric, -, _, space, max 255 chars",
                },
            )

        audio_bytes = await audio.read()
        try:
            await _ref_manager.add(id, audio_bytes, text)
        except ConflictError as e:
            return JSONResponse(
                status_code=409,
                content={"success": False, "message": str(e)},
            )

        return ReferenceAddResponse(
            message=f"Reference '{id}' added successfully",
            reference_id=id,
        )

    @app.get("/v1/references/list", response_model=ReferenceListResponse)
    async def references_list():
        ids = await _ref_manager.list_all()
        return ReferenceListResponse(
            reference_ids=ids,
            message=f"Found {len(ids)} reference(s)",
        )

    @app.delete("/v1/references/delete")
    async def references_delete(req: ReferenceDeleteRequest):
        try:
            await _ref_manager.delete(req.reference_id)
        except NotFoundError as e:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": str(e)},
            )
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Reference '{req.reference_id}' deleted",
            },
        )

    @app.post("/v1/references/update")
    async def references_update(req: ReferenceUpdateRequest):
        if not _ref_manager.validate_id(req.new_reference_id):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "Invalid new reference ID",
                },
            )
        try:
            await _ref_manager.update(req.old_reference_id, req.new_reference_id)
        except NotFoundError as e:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": str(e)},
            )
        except ConflictError as e:
            return JSONResponse(
                status_code=409,
                content={"success": False, "message": str(e)},
            )
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Reference renamed from '{req.old_reference_id}' to '{req.new_reference_id}'",
            },
        )

    return app
