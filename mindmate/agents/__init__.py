"""
MindMate Agents Package

This package contains all specialized agents for the Mental Health Support System:
- EmotionAgent: Empathetic listening and emotional support
- CBTAgent: Cognitive-behavioral therapy techniques
- CrisisAgent: Risk detection and emergency escalation
- KnowledgeAgent: Psychoeducation and mental health information
"""

try:
    from agents.base_agent import BaseAgent, AgentMessage, AgentResponse, AgentType
    from agents.emotion_agent import EmotionAgent
    from agents.cbt_agent import CBTAgent
    from agents.crisis_agent import CrisisAgent
    from agents.knowledge_agent import KnowledgeAgent
except ImportError:
    from .base_agent import BaseAgent, AgentMessage, AgentResponse, AgentType
    from .emotion_agent import EmotionAgent
    from .cbt_agent import CBTAgent
    from .crisis_agent import CrisisAgent
    from .knowledge_agent import KnowledgeAgent

__all__ = [
    "BaseAgent",
    "AgentMessage", 
    "AgentResponse",
    "AgentType",
    "EmotionAgent",
    "CBTAgent",
    "CrisisAgent",
    "KnowledgeAgent",
]

