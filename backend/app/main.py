"""Main FastAPI application entry point."""
import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import router
from .ai_routes import router as ai_router
from .docker_routes import router as docker_router
from .ecg_routes import router as ecg_router
from .settings import settings, ensure_workspace
from .utils import ensure_temp_directory

# Configure logging
if settings.LOG_FORMAT == "json":
    try:
        from pythonjsonlogger import jsonlogger
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(jsonlogger.JsonFormatter())
        logging.root.handlers = [handler]
    except ImportError:
        # Fallback to standard logging if jsonlogger not available
        logging.basicConfig(
            level=settings.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
else:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"AI Engine URL: {settings.AI_ENGINE_URL}")
    logger.info(f"AI Engine Mode: {settings.AI_ENGINE_MODE}")
    
    # Ensure temp directory exists
    temp_dir = ensure_temp_directory(settings.TEMP_DIR)
    logger.info(f"Temporary directory: {temp_dir}")

    # Ensure workspace directory and subdirectories exist
    logger.info(f"Workspace path: {settings.WORKSPACE_PATH}")
    subdirs = ensure_workspace(settings.WORKSPACE_PATH)
    logger.info(f"Workspace subdirectories: {subdirs}")

    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Local medical AI application for PDF analysis",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request ID to request state for tracing."""
    import uuid
    
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"HTTP exception for request {request_id}: "
        f"{exc.status_code} - {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"Validation error for request {request_id}: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception for request {request_id}: {exc}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "request_id": request_id
        }
    )


# Include routers
app.include_router(router)
app.include_router(ai_router)
app.include_router(docker_router)
app.include_router(ecg_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "disabled in production"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )