"""
MindMate - Mental Health Multi-Agent Support System

A production-ready multi-agent system providing compassionate mental health support
using Google's Agent Development Kit (ADK) and Gemini models.

Features:
- 4 Specialized Agents (Emotion, CBT, Crisis, Knowledge)
- MCP-based Journaling Tool
- Mood Tracking & Analysis
- Crisis Detection & Escalation
- Session & Long-term Memory
- Full Observability (Logging, Tracing, Metrics)

⚠️ IMPORTANT: MindMate is an AI support tool, NOT a replacement for professional care.
If you're in crisis, please contact:
- US: 988 (Suicide & Crisis Lifeline) or text HOME to 741741
- UK: 116 123 (Samaritans) or text SHOUT to 85258
- Emergency: Call your local emergency services
"""

__version__ = "1.0.0"
__author__ = "MindMate Team"
__license__ = "Apache-2.0"

from .config import config, get_config
from .agents import EmotionAgent, CBTAgent, CrisisAgent, KnowledgeAgent
from .memory import SessionService, LongTermMemory
from .tools import JournalMCPTool, MoodTracker, EmergencyAPI
from .workflows import MainRouter

__all__ = [
    # Configuration
    "config",
    "get_config",
    
    # Agents
    "EmotionAgent",
    "CBTAgent", 
    "CrisisAgent",
    "KnowledgeAgent",
    
    # Memory
    "SessionService",
    "LongTermMemory",
    
    # Tools
    "JournalMCPTool",
    "MoodTracker",
    "EmergencyAPI",
    
    # Workflows
    "MainRouter",
]

