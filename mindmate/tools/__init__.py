"""
MindMate Tools Package

This package contains all tools available to agents:
- JournalMCPTool: MCP-based journaling storage and retrieval
- MoodTracker: Mood logging and pattern analysis
- EmergencyAPI: Emergency escalation endpoint integration
"""

try:
    from tools.journal_mcp import JournalMCPTool, JournalEntry
    from tools.mood_tracker import MoodTracker, MoodEntry, MoodAnalysis
    from tools.emergency_api import EmergencyAPI, EmergencyAlert
except ImportError:
    from .journal_mcp import JournalMCPTool, JournalEntry
    from .mood_tracker import MoodTracker, MoodEntry, MoodAnalysis
    from .emergency_api import EmergencyAPI, EmergencyAlert

__all__ = [
    "JournalMCPTool",
    "JournalEntry",
    "MoodTracker", 
    "MoodEntry",
    "MoodAnalysis",
    "EmergencyAPI",
    "EmergencyAlert",
]

