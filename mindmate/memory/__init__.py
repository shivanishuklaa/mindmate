"""
MindMate Memory Package

This package provides memory management for the multi-agent system:
- SessionService: Per-session conversation tracking
- LongTermMemory: Persistent memory bank for patterns and insights
- ContextCompactor: Conversation compression for long sessions
"""

try:
    from memory.session import SessionService, Session, SessionMessage, MessageRole
    from memory.long_term_memory import LongTermMemory, MemoryEntry, MemoryType
except ImportError:
    from .session import SessionService, Session, SessionMessage, MessageRole
    from .long_term_memory import LongTermMemory, MemoryEntry, MemoryType

__all__ = [
    "SessionService",
    "Session",
    "SessionMessage",
    "MessageRole",
    "LongTermMemory",
    "MemoryEntry",
    "MemoryType",
]

