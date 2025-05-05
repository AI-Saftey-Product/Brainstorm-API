"""Main application module."""
import logging
from urllib.request import Request

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from starlette.responses import JSONResponse

from brainstorm.api.v1.api import api_router, auth_router
from brainstorm.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url="/api/openapi.json"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation error: {exc.errors()}")
    logging.error(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": await request.body()},
    )

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
app.include_router(auth_router, prefix="/auth")

@app.get("/", tags=["status"])
async def root():
    """Root endpoint for health check."""
    return {"status": "ok", "message": "AI Safety Testing API is running"}


@app.get("/health", tags=["status"])
async def health():
    """Health check endpoint."""
    return {"status": "ok"} 
