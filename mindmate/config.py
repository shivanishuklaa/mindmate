"""
MindMate Configuration Module

Centralized configuration management for the Mental Health Multi-Agent System.
Uses Pydantic Settings for type-safe environment variable parsing.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class MindMateConfig(BaseSettings):
    """
    Application configuration loaded from environment variables.
    
    All settings can be overridden via environment variables or .env file.
    Prefix: MINDMATE_ (e.g., MINDMATE_DEBUG=true)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MINDMATE_",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==================== API Keys ====================
    google_api_key: str = ""
    google_cloud_project: str = ""
    google_cloud_region: str = "us-central1"
    
    # ==================== Model Configuration ====================
    # Available models: gemini-2.0-flash, gemini-2.5-flash, gemini-2.0-flash-lite
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 2048
    
    # ==================== Server Configuration ====================
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    
    # ==================== Memory Configuration ====================
    memory_db_path: Path = Path("./data/mindmate.db")
    vector_db_path: Path = Path("./data/chromadb")
    context_window_size: int = 10  # Messages to keep in active context
    compaction_threshold: int = 20  # When to trigger context compaction
    
    # ==================== Observability ====================
    otel_endpoint: str = ""  # Set to enable OTEL export (e.g., "http://localhost:4317")
    service_name: str = "mindmate"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_tracing: bool = False  # Disabled by default, enable with OTEL endpoint
    enable_metrics: bool = False  # Disabled by default
    
    # ==================== Safety Configuration ====================
    crisis_threshold: float = 0.7  # Probability threshold for crisis escalation
    emergency_contact_enabled: bool = True
    emergency_webhook_url: str = "https://api.example.com/emergency"
    
    # Safety guardrails
    max_session_duration_minutes: int = 120
    require_crisis_confirmation: bool = True
    
    # ==================== Rate Limiting ====================
    max_requests_per_minute: int = 60
    session_timeout_minutes: int = 30
    
    # ==================== Agent Configuration ====================
    emotion_agent_personality: str = "warm, empathetic, supportive"
    cbt_agent_style: str = "gentle, Socratic, analytical"
    knowledge_agent_depth: str = "accessible, evidence-based"
    
    def ensure_directories(self) -> None:
        """Create necessary data directories if they don't exist."""
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> MindMateConfig:
    """
    Get cached configuration instance.
    
    Uses LRU cache to ensure configuration is loaded only once.
    """
    config = MindMateConfig()
    config.ensure_directories()
    return config


# Export commonly used config values
config = get_config()

