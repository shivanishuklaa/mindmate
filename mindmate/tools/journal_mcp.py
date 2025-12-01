"""
Journal MCP Tool

Implements a Model Context Protocol (MCP) based journaling tool
for storing and retrieving mental health journal entries.
Supports tagging, searching, and pattern analysis.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

import aiosqlite
from opentelemetry import trace

try:
    from config import config
except ImportError:
    from ..config import config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class JournalEntryType(Enum):
    """Types of journal entries."""
    FREEFORM = "freeform"
    GRATITUDE = "gratitude"
    REFLECTION = "reflection"
    CBT_THOUGHT_RECORD = "cbt_thought_record"
    MOOD_LOG = "mood_log"
    GOAL = "goal"
    ACCOMPLISHMENT = "accomplishment"


@dataclass
class JournalEntry:
    """
    A journal entry structure.
    
    Attributes:
        id: Unique entry identifier
        user_id: User who created the entry
        session_id: Session during which entry was created
        entry_type: Type of journal entry
        content: Main content of the entry
        title: Optional title
        mood_rating: Optional mood rating (1-10)
        tags: Tags for categorization
        metadata: Additional data
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    user_id: str
    content: str
    entry_type: JournalEntryType = JournalEntryType.FREEFORM
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    title: Optional[str] = None
    mood_rating: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['entry_type'] = self.entry_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['tags'] = json.dumps(self.tags)
        data['metadata'] = json.dumps(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JournalEntry':
        """Create from dictionary."""
        data['entry_type'] = JournalEntryType(data['entry_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if isinstance(data['tags'], str):
            data['tags'] = json.loads(data['tags'])
        if isinstance(data['metadata'], str):
            data['metadata'] = json.loads(data['metadata'])
        return cls(**data)


class JournalMCPTool:
    """
    MCP-based journaling tool for storing and analyzing journal entries.
    
    Provides:
    - Create, read, update, delete journal entries
    - Tag-based searching
    - Pattern analysis over time
    - Mood correlation
    - Export capabilities
    
    MCP Protocol:
    This tool follows the Model Context Protocol specification,
    providing structured tool definitions for LLM integration.
    """
    
    TOOL_NAME = "journal"
    TOOL_DESCRIPTION = "Store and retrieve mental health journal entries"
    
    # MCP Tool Schema
    TOOL_SCHEMA = {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["write", "read", "search", "analyze", "list"],
                    "description": "The action to perform"
                },
                "content": {
                    "type": "string",
                    "description": "Journal entry content (for write action)"
                },
                "entry_id": {
                    "type": "string",
                    "description": "Entry ID (for read action)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to filter by"
                },
                "entry_type": {
                    "type": "string",
                    "enum": ["freeform", "gratitude", "reflection", "cbt_thought_record", "mood_log"],
                    "description": "Type of journal entry"
                },
                "mood_rating": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Mood rating for the entry"
                },
                "days": {
                    "type": "integer",
                    "description": "Number of days to analyze (for analyze action)"
                }
            },
            "required": ["action"]
        }
    }

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the Journal MCP Tool.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or config.memory_db_path
        self._initialized = False
        self._db_lock = asyncio.Lock()
        
        logger.info(f"JournalMCPTool initialized with db: {self.db_path}")

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS journal_entries (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        entry_type TEXT NOT NULL,
                        title TEXT,
                        content TEXT NOT NULL,
                        mood_rating INTEGER,
                        tags TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_journal_user 
                    ON journal_entries(user_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_journal_created 
                    ON journal_entries(created_at)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_journal_type 
                    ON journal_entries(entry_type)
                """)
                
                await db.commit()
                
            self._initialized = True
            logger.info("Journal database initialized")

    async def execute(
        self,
        action: str,
        user_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a journal action (MCP entry point).
        
        Args:
            action: The action to perform
            user_id: User performing the action
            **kwargs: Action-specific parameters
            
        Returns:
            Action result
        """
        await self.initialize()
        
        with tracer.start_as_current_span(f"journal_{action}") as span:
            span.set_attribute("action", action)
            span.set_attribute("user_id", user_id)
            
            if action == "write":
                return await self.write(user_id=user_id, **kwargs)
            elif action == "read":
                return await self.read(user_id=user_id, **kwargs)
            elif action == "search":
                return await self.search(user_id=user_id, **kwargs)
            elif action == "analyze":
                return await self.analyze(user_id=user_id, **kwargs)
            elif action == "list":
                return await self.list_entries(user_id=user_id, **kwargs)
            else:
                raise ValueError(f"Unknown action: {action}")

    async def write(
        self,
        user_id: str,
        content: str,
        entry_type: str = "freeform",
        title: Optional[str] = None,
        mood_rating: Optional[int] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Write a new journal entry.
        
        Args:
            user_id: User creating the entry
            content: Entry content
            entry_type: Type of entry
            title: Optional title
            mood_rating: Optional mood rating (1-10)
            tags: Tags for categorization
            session_id: Session ID if in a session
            metadata: Additional metadata
            
        Returns:
            Created entry details
        """
        entry = JournalEntry(
            user_id=user_id,
            content=content,
            entry_type=JournalEntryType(entry_type),
            title=title,
            mood_rating=mood_rating,
            tags=tags or [],
            session_id=session_id,
            metadata=metadata or {}
        )
        
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                data = entry.to_dict()
                await db.execute("""
                    INSERT INTO journal_entries 
                    (id, user_id, session_id, entry_type, title, content, 
                     mood_rating, tags, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['id'], data['user_id'], data['session_id'],
                    data['entry_type'], data['title'], data['content'],
                    data['mood_rating'], data['tags'], data['metadata'],
                    data['created_at'], data['updated_at']
                ))
                await db.commit()
        
        logger.info(f"Journal entry created: {entry.id}")
        
        return {
            "success": True,
            "entry_id": entry.id,
            "message": "Journal entry saved successfully",
            "entry_type": entry_type,
            "created_at": entry.created_at.isoformat()
        }

    async def read(
        self,
        user_id: str,
        entry_id: str
    ) -> Dict[str, Any]:
        """
        Read a specific journal entry.
        
        Args:
            user_id: User requesting the entry
            entry_id: Entry to retrieve
            
        Returns:
            Entry details
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM journal_entries 
                WHERE id = ? AND user_id = ?
            """, (entry_id, user_id)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return {"success": False, "error": "Entry not found"}
                
                entry = JournalEntry.from_dict(dict(row))
                return {
                    "success": True,
                    "entry": {
                        "id": entry.id,
                        "title": entry.title,
                        "content": entry.content,
                        "entry_type": entry.entry_type.value,
                        "mood_rating": entry.mood_rating,
                        "tags": entry.tags,
                        "created_at": entry.created_at.isoformat()
                    }
                }

    async def search(
        self,
        user_id: str,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        entry_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search journal entries.
        
        Args:
            user_id: User searching
            query: Text search query
            tags: Filter by tags
            entry_type: Filter by entry type
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum results
            
        Returns:
            Search results
        """
        conditions = ["user_id = ?"]
        params: List[Any] = [user_id]
        
        if query:
            conditions.append("(content LIKE ? OR title LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if entry_type:
            conditions.append("entry_type = ?")
            params.append(entry_type)
        
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date.isoformat())
        
        where_clause = " AND ".join(conditions)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(f"""
                SELECT * FROM journal_entries 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
            """, params + [limit]) as cursor:
                rows = await cursor.fetchall()
                
                entries = []
                for row in rows:
                    entry_data = dict(row)
                    # Check tag filter
                    if tags:
                        entry_tags = json.loads(entry_data.get('tags', '[]'))
                        if not any(t in entry_tags for t in tags):
                            continue
                    
                    entries.append({
                        "id": entry_data['id'],
                        "title": entry_data['title'],
                        "content": entry_data['content'][:100] + "..." if len(entry_data['content']) > 100 else entry_data['content'],
                        "entry_type": entry_data['entry_type'],
                        "mood_rating": entry_data['mood_rating'],
                        "created_at": entry_data['created_at']
                    })
                
                return {
                    "success": True,
                    "count": len(entries),
                    "entries": entries
                }

    async def analyze(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze journal patterns over a time period.
        
        Args:
            user_id: User to analyze
            days: Number of days to analyze
            
        Returns:
            Pattern analysis
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Get entries in period
            async with db.execute("""
                SELECT * FROM journal_entries 
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at ASC
            """, (user_id, start_date.isoformat())) as cursor:
                rows = await cursor.fetchall()
                
                if not rows:
                    return {
                        "success": True,
                        "period_days": days,
                        "total_entries": 0,
                        "message": "No entries found in this period"
                    }
                
                # Analyze patterns
                entries_by_type: Dict[str, int] = {}
                mood_ratings: List[int] = []
                all_tags: Dict[str, int] = {}
                
                for row in rows:
                    entry_data = dict(row)
                    
                    # Count by type
                    entry_type = entry_data['entry_type']
                    entries_by_type[entry_type] = entries_by_type.get(entry_type, 0) + 1
                    
                    # Collect mood ratings
                    if entry_data['mood_rating']:
                        mood_ratings.append(entry_data['mood_rating'])
                    
                    # Count tags
                    tags = json.loads(entry_data.get('tags', '[]'))
                    for tag in tags:
                        all_tags[tag] = all_tags.get(tag, 0) + 1
                
                # Calculate statistics
                avg_mood = sum(mood_ratings) / len(mood_ratings) if mood_ratings else None
                mood_trend = None
                if len(mood_ratings) >= 3:
                    first_half = mood_ratings[:len(mood_ratings)//2]
                    second_half = mood_ratings[len(mood_ratings)//2:]
                    avg_first = sum(first_half) / len(first_half)
                    avg_second = sum(second_half) / len(second_half)
                    if avg_second > avg_first + 0.5:
                        mood_trend = "improving"
                    elif avg_second < avg_first - 0.5:
                        mood_trend = "declining"
                    else:
                        mood_trend = "stable"
                
                # Top tags
                top_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)[:5]
                
                return {
                    "success": True,
                    "period_days": days,
                    "total_entries": len(rows),
                    "entries_per_day": round(len(rows) / days, 2),
                    "entries_by_type": entries_by_type,
                    "mood_analysis": {
                        "average_rating": round(avg_mood, 2) if avg_mood else None,
                        "trend": mood_trend,
                        "total_mood_entries": len(mood_ratings)
                    },
                    "top_themes": [{"tag": t, "count": c} for t, c in top_tags],
                    "journaling_consistency": "high" if len(rows) >= days * 0.7 else "moderate" if len(rows) >= days * 0.4 else "developing"
                }

    async def list_entries(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List recent journal entries.
        
        Args:
            user_id: User to list for
            limit: Maximum entries
            offset: Pagination offset
            
        Returns:
            List of entries
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT id, title, entry_type, mood_rating, created_at 
                FROM journal_entries 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset)) as cursor:
                rows = await cursor.fetchall()
                
                entries = [
                    {
                        "id": row['id'],
                        "title": row['title'] or "Untitled",
                        "entry_type": row['entry_type'],
                        "mood_rating": row['mood_rating'],
                        "created_at": row['created_at']
                    }
                    for row in rows
                ]
                
                return {
                    "success": True,
                    "entries": entries,
                    "count": len(entries)
                }

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return MCP tool definition."""
        return self.TOOL_SCHEMA

    def as_callable(self) -> Callable:
        """Return the execute method as a callable for agent registration."""
        return self.execute

