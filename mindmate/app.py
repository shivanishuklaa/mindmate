"""
MindMate - Mental Health Multi-Agent System
Main Application Entry Point

This module provides the FastAPI server for the MindMate system,
exposing REST endpoints for conversation, tools, and system health.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

from config import config
from memory import SessionService, LongTermMemory
from workflows import MainRouter
from agents import AgentType

# ==================== Logging Setup ====================

def setup_logging() -> None:
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level)
    )

setup_logging()
logger = structlog.get_logger(__name__)


# ==================== Tracing Setup ====================

def setup_tracing() -> None:
    """Configure OpenTelemetry tracing."""
    if not config.enable_tracing:
        return
    
    resource = Resource.create({
        "service.name": config.service_name,
        "service.version": "1.0.0"
    })
    
    provider = TracerProvider(resource=resource)
    
    if config.otel_endpoint:
        exporter = OTLPSpanExporter(endpoint=config.otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    
    trace.set_tracer_provider(provider)

setup_tracing()
tracer = trace.get_tracer(__name__)


# ==================== Application State ====================

class AppState:
    """Application state container."""
    session_service: SessionService
    long_term_memory: LongTermMemory
    router: MainRouter
    start_time: datetime

app_state = AppState()


# ==================== Lifespan Management ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting MindMate application...")
    
    # Initialize services
    app_state.session_service = SessionService()
    app_state.long_term_memory = LongTermMemory()
    
    # Initialize memory databases
    await app_state.long_term_memory.initialize()
    
    # Initialize main router with all agents
    app_state.router = MainRouter(
        session_service=app_state.session_service,
        long_term_memory=app_state.long_term_memory
    )
    
    app_state.start_time = datetime.utcnow()
    
    logger.info(
        "MindMate initialized",
        agents=list(app_state.router.agents.keys()),
        model=config.gemini_model
    )
    
    yield
    
    # Cleanup
    logger.info("Shutting down MindMate...")
    await app_state.session_service.cleanup_expired_sessions()


# ==================== FastAPI Application ====================

app = FastAPI(
    title="MindMate",
    description="Mental Health Multi-Agent Support System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument with OpenTelemetry
if config.enable_tracing:
    FastAPIInstrumentor.instrument_app(app)


# ==================== Request/Response Models ====================

class ChatRequest(BaseModel):
    """Chat message request."""
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    user_id: str = Field(default="anonymous")

class ChatResponse(BaseModel):
    """Chat message response."""
    response: str
    session_id: str
    agents_used: List[str]
    workflow_pattern: str
    processing_time_ms: float
    metadata: Dict[str, Any] = {}

class SessionCreateRequest(BaseModel):
    """Session creation request."""
    user_id: str

class SessionResponse(BaseModel):
    """Session details response."""
    session_id: str
    user_id: str
    message_count: int
    created_at: str
    status: str

class ToolRequest(BaseModel):
    """Tool execution request."""
    tool_name: str
    action: str
    user_id: str
    session_id: Optional[str] = None
    parameters: Dict[str, Any] = {}

class MoodLogRequest(BaseModel):
    """Mood logging request."""
    user_id: str
    mood_rating: int = Field(..., ge=1, le=10)
    emotions: List[str] = []
    context: Optional[str] = None
    session_id: Optional[str] = None

class JournalEntryRequest(BaseModel):
    """Journal entry request."""
    user_id: str
    content: str
    entry_type: str = "freeform"
    title: Optional[str] = None
    mood_rating: Optional[int] = None
    tags: List[str] = []

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    agents_ready: bool


# ==================== Static Files ====================

# Get the static directory path
STATIC_DIR = Path(__file__).parent / "static"


# ==================== API Endpoints ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(content="""
        <html>
            <head><title>MindMate</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1>🧠 MindMate</h1>
                <p>Frontend not found. API is running.</p>
                <p><a href="/docs">Open API Documentation</a></p>
            </body>
        </html>
    """)


@app.get("/api", response_class=JSONResponse)
async def api_info():
    """API information endpoint."""
    return {
        "name": "MindMate",
        "description": "Mental Health Multi-Agent Support System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.utcnow() - app_state.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        agents_ready=len(app_state.router.agents) == 4
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for conversation.
    
    Automatically routes messages to appropriate agents and
    manages session state.
    """
    session_id = request.session_id
    
    try:
        # Check if API key is configured
        if not config.google_api_key:
            return ChatResponse(
                response=(
                    "⚠️ **API Key Not Configured**\n\n"
                    "Please set your Google API key:\n"
                    "```\n$env:MINDMATE_GOOGLE_API_KEY = 'your_key'\n```\n\n"
                    "Get a free key at: https://aistudio.google.com/app/apikey"
                ),
                session_id=session_id or "no-session",
                agents_used=[],
                workflow_pattern="error",
                processing_time_ms=0,
                metadata={"error": "api_key_missing"}
            )
        
        # Get or create session
        if not session_id:
            session = await app_state.session_service.create_session(
                user_id=request.user_id
            )
            session_id = session.id
        else:
            session = await app_state.session_service.get_session(session_id)
            if not session:
                session = await app_state.session_service.create_session(
                    user_id=request.user_id
                )
                session_id = session.id
        
        # Process message through router
        result = await app_state.router.process_message(
            user_input=request.message,
            session_id=session_id
        )
        
        return ChatResponse(
            response=result.response,
            session_id=session_id,
            agents_used=[a.value for a in result.agents_used],
            workflow_pattern=result.workflow_pattern.value,
            processing_time_ms=result.processing_time_ms,
            metadata=result.metadata
        )
        
    except Exception as e:
        error_str = str(e)
        logger.error("Chat error", error=error_str)
        
        # Provide helpful error messages
        if "429" in error_str or "quota" in error_str.lower():
            error_response = (
                "⏳ **Rate Limit Reached**\n\n"
                "The API quota has been exceeded. Please wait a minute and try again.\n\n"
                "If this persists, consider:\n"
                "• Waiting a few minutes\n"
                "• Using a different API key\n"
                "• Upgrading to a paid plan"
            )
        elif "401" in error_str or "invalid" in error_str.lower() or "api key" in error_str.lower():
            error_response = (
                "🔑 **Invalid API Key**\n\n"
                "Your Google API key appears to be invalid.\n"
                "Please check your key at: https://aistudio.google.com/app/apikey"
            )
        else:
            error_response = (
                "😔 **Something went wrong**\n\n"
                "I encountered an issue processing your message. "
                "I'm still here for you - please try again.\n\n"
                f"*Technical details: {error_str[:100]}*"
            )
        
        return ChatResponse(
            response=error_response,
            session_id=session_id or "error-session",
            agents_used=[],
            workflow_pattern="error",
            processing_time_ms=0,
            metadata={"error": error_str[:200]}
        )


@app.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new conversation session."""
    session = await app_state.session_service.create_session(
        user_id=request.user_id
    )
    
    return SessionResponse(
        session_id=session.id,
        user_id=session.user_id,
        message_count=session.message_count,
        created_at=session.created_at.isoformat(),
        status=session.status.value
    )


@app.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details."""
    session = await app_state.session_service.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session.id,
        user_id=session.user_id,
        message_count=session.message_count,
        created_at=session.created_at.isoformat(),
        status=session.status.value
    )


@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """End a conversation session."""
    summary = await app_state.session_service.end_session(
        session_id,
        generate_summary=True
    )
    
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"success": True, "summary": summary}


@app.post("/mood")
async def log_mood(request: MoodLogRequest):
    """Log a mood entry."""
    result = await app_state.router.mood_tracker.write(
        user_id=request.user_id,
        mood_rating=request.mood_rating,
        emotions=request.emotions,
        context=request.context
    )
    
    return result


@app.get("/mood/{user_id}")
async def get_mood_history(user_id: str, days: int = 7):
    """Get mood history for a user."""
    result = await app_state.router.mood_tracker.read(
        user_id=user_id,
        days=days
    )
    
    return result


@app.get("/mood/{user_id}/analyze")
async def analyze_mood(user_id: str, days: int = 30):
    """Analyze mood patterns for a user."""
    result = await app_state.router.mood_tracker.analyze(
        user_id=user_id,
        days=days
    )
    
    return result


@app.post("/journal")
async def create_journal_entry(request: JournalEntryRequest):
    """Create a journal entry."""
    result = await app_state.router.journal_tool.execute(
        action="write",
        user_id=request.user_id,
        content=request.content,
        entry_type=request.entry_type,
        title=request.title,
        mood_rating=request.mood_rating,
        tags=request.tags
    )
    
    return result


@app.get("/journal/{user_id}")
async def get_journal_entries(user_id: str, limit: int = 10):
    """Get journal entries for a user."""
    result = await app_state.router.journal_tool.execute(
        action="list",
        user_id=user_id,
        limit=limit
    )
    
    return result


@app.get("/journal/{user_id}/analyze")
async def analyze_journal(user_id: str, days: int = 7):
    """Analyze journal patterns for a user."""
    result = await app_state.router.journal_tool.execute(
        action="analyze",
        user_id=user_id,
        days=days
    )
    
    return result


@app.get("/memory/{user_id}")
async def get_user_memories(user_id: str, limit: int = 20):
    """Get long-term memories for a user."""
    memories = await app_state.long_term_memory.get_user_memories(
        user_id=user_id,
        limit=limit
    )
    
    return {
        "user_id": user_id,
        "memories": [
            {
                "id": m.id,
                "type": m.memory_type.value,
                "summary": m.summary,
                "importance": m.importance.value,
                "created_at": m.created_at.isoformat()
            }
            for m in memories
        ]
    }


@app.get("/metrics")
async def get_metrics():
    """Get system metrics for observability."""
    return app_state.router.get_metrics()


@app.get("/crisis/resources")
async def get_crisis_resources(region: str = "us"):
    """Get crisis resources for a region."""
    return app_state.router.emergency_api.get_crisis_resources(region)


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "An unexpected error occurred",
            "detail": str(exc) if config.debug else "Please try again later"
        }
    )


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower()
    )

