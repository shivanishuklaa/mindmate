"""
MindMate Observability Module

Provides logging, tracing, and metrics functionality for the system.
Implements OpenTelemetry instrumentation and structured logging.
"""

import logging
import sys
from functools import wraps
from typing import Any, Callable, Dict, Optional
import time

import structlog
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

from config import config


# ==================== Resource Configuration ====================

def get_resource() -> Resource:
    """Get OpenTelemetry resource with service info."""
    return Resource.create({
        "service.name": config.service_name,
        "service.version": "1.0.0",
        "deployment.environment": "development" if config.debug else "production"
    })


# ==================== Logging Setup ====================

def setup_logging() -> None:
    """Configure structured logging with structlog."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Use JSON in production, colored console in development
            structlog.processors.JSONRenderer() if not config.debug else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level)
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for the given name."""
    return structlog.get_logger(name)


# ==================== Tracing Setup ====================

_tracer_provider: Optional[TracerProvider] = None
_tracer: Optional[trace.Tracer] = None


def setup_tracing() -> None:
    """Configure OpenTelemetry tracing."""
    global _tracer_provider, _tracer
    
    if not config.enable_tracing:
        return
    
    resource = get_resource()
    _tracer_provider = TracerProvider(resource=resource)
    
    # Add console exporter for development
    if config.debug:
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
    
    # Add OTLP exporter if configured
    if config.otel_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=config.otel_endpoint)
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            pass
    
    trace.set_tracer_provider(_tracer_provider)
    _tracer = trace.get_tracer(__name__)


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer for the given name."""
    return trace.get_tracer(name)


# ==================== Metrics Setup ====================

_meter_provider: Optional[MeterProvider] = None
_meter: Optional[metrics.Meter] = None

# Custom metrics
_request_counter: Optional[metrics.Counter] = None
_response_time_histogram: Optional[metrics.Histogram] = None
_crisis_counter: Optional[metrics.Counter] = None
_mood_histogram: Optional[metrics.Histogram] = None


def setup_metrics() -> None:
    """Configure OpenTelemetry metrics."""
    global _meter_provider, _meter
    global _request_counter, _response_time_histogram, _crisis_counter, _mood_histogram
    
    if not config.enable_metrics:
        return
    
    resource = get_resource()
    
    # Create metric reader
    reader = PeriodicExportingMetricReader(
        ConsoleMetricExporter(),
        export_interval_millis=60000  # Export every minute
    )
    
    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[reader]
    )
    
    metrics.set_meter_provider(_meter_provider)
    _meter = metrics.get_meter(__name__)
    
    # Create custom metrics
    _request_counter = _meter.create_counter(
        name="mindmate.requests",
        description="Number of requests processed",
        unit="1"
    )
    
    _response_time_histogram = _meter.create_histogram(
        name="mindmate.response_time",
        description="Response time in milliseconds",
        unit="ms"
    )
    
    _crisis_counter = _meter.create_counter(
        name="mindmate.crisis_detections",
        description="Number of crisis situations detected",
        unit="1"
    )
    
    _mood_histogram = _meter.create_histogram(
        name="mindmate.mood_ratings",
        description="Distribution of mood ratings",
        unit="1"
    )


def record_request(agent: str, intent: str) -> None:
    """Record a request metric."""
    if _request_counter:
        _request_counter.add(1, {"agent": agent, "intent": intent})


def record_response_time(ms: float, agent: str) -> None:
    """Record response time metric."""
    if _response_time_histogram:
        _response_time_histogram.record(ms, {"agent": agent})


def record_crisis_detection(severity: str) -> None:
    """Record a crisis detection metric."""
    if _crisis_counter:
        _crisis_counter.add(1, {"severity": severity})


def record_mood(rating: int) -> None:
    """Record a mood rating metric."""
    if _mood_histogram:
        _mood_histogram.record(rating)


# ==================== Instrumentation Decorators ====================

def traced(name: Optional[str] = None):
    """
    Decorator to add tracing to a function.
    
    Args:
        name: Optional span name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def timed(metric_name: Optional[str] = None):
    """
    Decorator to measure and record function execution time.
    
    Args:
        metric_name: Optional metric name prefix
    """
    def decorator(func: Callable) -> Callable:
        name = metric_name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                record_response_time(duration_ms, name)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                record_response_time(duration_ms, name)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ==================== Initialization ====================

def init_observability() -> None:
    """Initialize all observability components."""
    setup_logging()
    setup_tracing()
    setup_metrics()


# Initialize on import if not in test mode
if not hasattr(sys, '_called_from_test'):
    init_observability()

