"""
Long-Term Memory Module (Memory Bank)

Provides persistent memory storage for mental health patterns, insights,
and user preferences. Supports vector-based semantic search for
relevant context retrieval.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import aiosqlite
from opentelemetry import trace

try:
    from config import config
except ImportError:
    from ..config import config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class MemoryType(Enum):
    """Types of memories stored."""
    INSIGHT = "insight"  # Therapeutic insights
    PATTERN = "pattern"  # Behavioral/emotional patterns
    PREFERENCE = "preference"  # User preferences
    TRIGGER = "trigger"  # Identified triggers
    COPING = "coping"  # Coping strategies that work
    GOAL = "goal"  # Therapeutic goals
    BREAKTHROUGH = "breakthrough"  # Significant progress
    CONCERN = "concern"  # Ongoing concerns
    SESSION_SUMMARY = "session_summary"  # Session summaries


class MemoryImportance(Enum):
    """Importance levels for memory prioritization."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryEntry:
    """
    A long-term memory entry.
    
    Attributes:
        id: Unique memory identifier
        user_id: User this memory belongs to
        memory_type: Type of memory
        content: Memory content
        summary: Brief summary for quick retrieval
        importance: How important this memory is
        tags: Tags for categorization
        source_session: Session where this was learned
        metadata: Additional data
        created_at: When the memory was created
        last_accessed: Last time this memory was retrieved
        access_count: How many times this memory was accessed
    """
    user_id: str
    memory_type: MemoryType
    content: str
    summary: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    importance: MemoryImportance = MemoryImportance.NORMAL
    tags: List[str] = field(default_factory=list)
    source_session: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    
    # Embedding for semantic search (stored separately)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['importance'] = self.importance.value
        data['tags'] = json.dumps(self.tags)
        data['metadata'] = json.dumps(self.metadata)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        data.pop('embedding', None)  # Store separately
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        data['memory_type'] = MemoryType(data['memory_type'])
        data['importance'] = MemoryImportance(data['importance'])
        data['tags'] = json.loads(data['tags']) if isinstance(data['tags'], str) else data['tags']
        data['metadata'] = json.loads(data['metadata']) if isinstance(data['metadata'], str) else data['metadata']
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class LongTermMemory:
    """
    Long-term memory bank for persistent user insights.
    
    Provides:
    - Persistent storage of therapeutic insights and patterns
    - Semantic search for relevant memories
    - Memory importance ranking and decay
    - Cross-session context retrieval
    - Pattern detection over time
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the long-term memory system.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or config.memory_db_path
        self._initialized = False
        self._db_lock = asyncio.Lock()
        
        # Simple in-memory cache for frequently accessed memories
        self._cache: Dict[str, MemoryEntry] = {}
        self._cache_limit = 100
        
        # Metrics
        self._memories_created = 0
        self._memories_retrieved = 0
        self._searches_performed = 0
        
        logger.info(f"LongTermMemory initialized with db: {self.db_path}")

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        importance INTEGER NOT NULL,
                        tags TEXT,
                        source_session TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        last_accessed TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0
                    )
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_user 
                    ON memories(user_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_type 
                    ON memories(memory_type)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_importance 
                    ON memories(importance DESC)
                """)
                
                # Full-text search index for content
                await db.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts 
                    USING fts5(id, content, summary, tags)
                """)
                
                await db.commit()
                
            self._initialized = True
            logger.info("Long-term memory database initialized")

    async def store(
        self,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        summary: str,
        importance: MemoryImportance = MemoryImportance.NORMAL,
        tags: Optional[List[str]] = None,
        source_session: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        Store a new memory.
        
        Args:
            user_id: User to store memory for
            memory_type: Type of memory
            content: Full memory content
            summary: Brief summary
            importance: Memory importance level
            tags: Tags for categorization
            source_session: Session where this originated
            metadata: Additional metadata
            
        Returns:
            Created memory entry
        """
        await self.initialize()
        
        with tracer.start_as_current_span("memory_store") as span:
            span.set_attribute("memory_type", memory_type.value)
            span.set_attribute("importance", importance.value)
            
            memory = MemoryEntry(
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                summary=summary,
                importance=importance,
                tags=tags or [],
                source_session=source_session,
                metadata=metadata or {}
            )
            
            async with self._db_lock:
                async with aiosqlite.connect(self.db_path) as db:
                    data = memory.to_dict()
                    await db.execute("""
                        INSERT INTO memories 
                        (id, user_id, memory_type, content, summary, importance,
                         tags, source_session, metadata, created_at, last_accessed,
                         access_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['id'], data['user_id'], data['memory_type'],
                        data['content'], data['summary'], data['importance'],
                        data['tags'], data['source_session'], data['metadata'],
                        data['created_at'], data['last_accessed'], data['access_count']
                    ))
                    
                    # Add to FTS index
                    await db.execute("""
                        INSERT INTO memories_fts (id, content, summary, tags)
                        VALUES (?, ?, ?, ?)
                    """, (memory.id, content, summary, json.dumps(tags or [])))
                    
                    await db.commit()
            
            # Update cache
            self._update_cache(memory)
            
            self._memories_created += 1
            logger.info(f"Stored memory {memory.id} for user {user_id}")
            
            return memory

    async def retrieve(
        self,
        memory_id: str,
        update_access: bool = True
    ) -> Optional[MemoryEntry]:
        """
        Retrieve a specific memory.
        
        Args:
            memory_id: Memory to retrieve
            update_access: Whether to update access stats
            
        Returns:
            Memory entry or None
        """
        await self.initialize()
        
        # Check cache first
        if memory_id in self._cache:
            memory = self._cache[memory_id]
            if update_access:
                await self._update_access(memory_id)
            return memory
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                memory = MemoryEntry.from_dict(dict(row))
                
                if update_access:
                    await self._update_access(memory_id)
                
                self._update_cache(memory)
                self._memories_retrieved += 1
                
                return memory

    def _sanitize_fts_query(self, query: str) -> str:
        """
        Sanitize a query string for FTS5 to prevent syntax errors.
        Escapes special characters that have meaning in FTS5.
        """
        # Remove or escape FTS5 special characters
        special_chars = ['"', "'", '*', '-', '+', '(', ')', ':', '^', '~', '@', '<', '>', '[', ']', '{', '}']
        sanitized = query
        for char in special_chars:
            sanitized = sanitized.replace(char, ' ')
        # Remove extra whitespace and return
        return ' '.join(sanitized.split())

    async def search(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        min_importance: MemoryImportance = MemoryImportance.LOW,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search memories using full-text search.
        
        Args:
            user_id: User to search for
            query: Search query
            memory_types: Filter by memory types
            tags: Filter by tags
            min_importance: Minimum importance level
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        await self.initialize()
        
        with tracer.start_as_current_span("memory_search") as span:
            span.set_attribute("query", query)
            span.set_attribute("limit", limit)
            
            self._searches_performed += 1
            
            # Sanitize query for FTS5
            safe_query = self._sanitize_fts_query(query)
            
            # If query becomes empty after sanitization, do a simple LIKE search instead
            if not safe_query.strip():
                return await self.get_user_memories(user_id, memory_types, limit)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                try:
                    # Build query with FTS
                    sql = """
                        SELECT m.* FROM memories m
                        JOIN memories_fts fts ON m.id = fts.id
                        WHERE m.user_id = ? 
                        AND m.importance >= ?
                        AND memories_fts MATCH ?
                    """
                    params: List[Any] = [user_id, min_importance.value, safe_query]
                
                    if memory_types:
                        placeholders = ','.join(['?' for _ in memory_types])
                        sql += f" AND m.memory_type IN ({placeholders})"
                        params.extend([mt.value for mt in memory_types])
                    
                    sql += " ORDER BY m.importance DESC, m.last_accessed DESC LIMIT ?"
                    params.append(limit)
                    
                    async with db.execute(sql, params) as cursor:
                        rows = await cursor.fetchall()
                        
                        memories = []
                        for row in rows:
                            memory = MemoryEntry.from_dict(dict(row))
                            
                            # Filter by tags if specified
                            if tags and not any(t in memory.tags for t in tags):
                                continue
                            
                            memories.append(memory)
                        
                        return memories
                        
                except Exception as e:
                    # If FTS fails, fall back to simple search
                    logger.warning(f"FTS search failed, falling back to simple search: {e}")
                    return await self.get_user_memories(user_id, memory_types, limit)

    async def get_user_memories(
        self,
        user_id: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 50,
        order_by: str = "importance"
    ) -> List[MemoryEntry]:
        """
        Get all memories for a user.
        
        Args:
            user_id: User to get memories for
            memory_types: Filter by types
            limit: Maximum memories
            order_by: Sort field (importance, created_at, access_count)
            
        Returns:
            List of memories
        """
        await self.initialize()
        
        order_clause = {
            "importance": "importance DESC, last_accessed DESC",
            "created_at": "created_at DESC",
            "access_count": "access_count DESC",
            "recent": "last_accessed DESC"
        }.get(order_by, "importance DESC")
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            sql = f"SELECT * FROM memories WHERE user_id = ?"
            params: List[Any] = [user_id]
            
            if memory_types:
                placeholders = ','.join(['?' for _ in memory_types])
                sql += f" AND memory_type IN ({placeholders})"
                params.extend([mt.value for mt in memory_types])
            
            sql += f" ORDER BY {order_clause} LIMIT ?"
            params.append(limit)
            
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [MemoryEntry.from_dict(dict(row)) for row in rows]

    async def get_patterns(self, user_id: str) -> List[MemoryEntry]:
        """Get all pattern memories for a user."""
        return await self.get_user_memories(
            user_id,
            memory_types=[MemoryType.PATTERN],
            limit=20
        )

    async def get_coping_strategies(self, user_id: str) -> List[MemoryEntry]:
        """Get all coping strategy memories for a user."""
        return await self.get_user_memories(
            user_id,
            memory_types=[MemoryType.COPING],
            limit=20
        )

    async def get_triggers(self, user_id: str) -> List[MemoryEntry]:
        """Get all trigger memories for a user."""
        return await self.get_user_memories(
            user_id,
            memory_types=[MemoryType.TRIGGER],
            limit=20
        )

    async def get_context_for_session(
        self,
        user_id: str,
        current_topic: Optional[str] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Get relevant context for a new session.
        
        Args:
            user_id: User starting session
            current_topic: Topic if known
            limit: Max memories per category
            
        Returns:
            Context dictionary with relevant memories
        """
        await self.initialize()
        
        context = {
            "patterns": [],
            "coping_strategies": [],
            "recent_insights": [],
            "active_goals": [],
            "concerns": []
        }
        
        # Get relevant patterns
        patterns = await self.get_patterns(user_id)
        context["patterns"] = [
            {"summary": p.summary, "importance": p.importance.value}
            for p in patterns[:limit]
        ]
        
        # Get coping strategies that work
        coping = await self.get_coping_strategies(user_id)
        context["coping_strategies"] = [
            {"summary": c.summary, "tags": c.tags}
            for c in coping[:limit]
        ]
        
        # Get recent insights
        insights = await self.get_user_memories(
            user_id,
            memory_types=[MemoryType.INSIGHT, MemoryType.BREAKTHROUGH],
            limit=limit,
            order_by="created_at"
        )
        context["recent_insights"] = [
            {"summary": i.summary, "type": i.memory_type.value}
            for i in insights
        ]
        
        # Get active goals
        goals = await self.get_user_memories(
            user_id,
            memory_types=[MemoryType.GOAL],
            limit=limit
        )
        context["active_goals"] = [
            {"summary": g.summary, "metadata": g.metadata}
            for g in goals
        ]
        
        # Get ongoing concerns
        concerns = await self.get_user_memories(
            user_id,
            memory_types=[MemoryType.CONCERN],
            limit=limit
        )
        context["concerns"] = [
            {"summary": c.summary, "importance": c.importance.value}
            for c in concerns
        ]
        
        # If topic provided, search for related memories
        if current_topic:
            related = await self.search(
                user_id,
                current_topic,
                limit=3
            )
            context["topic_related"] = [
                {"summary": r.summary, "type": r.memory_type.value}
                for r in related
            ]
        
        return context

    async def _update_access(self, memory_id: str) -> None:
        """Update access statistics for a memory."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE memories 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), memory_id))
            await db.commit()

    def _update_cache(self, memory: MemoryEntry) -> None:
        """Update the memory cache."""
        if len(self._cache) >= self._cache_limit:
            # Remove least accessed
            sorted_cache = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )
            for key, _ in sorted_cache[:10]:
                del self._cache[key]
        
        self._cache[memory.id] = memory

    async def decay_memories(
        self,
        user_id: str,
        days_threshold: int = 90
    ) -> int:
        """
        Apply decay to old, unused memories.
        
        Reduces importance of memories not accessed in a while.
        
        Args:
            user_id: User to process
            days_threshold: Days without access before decay
            
        Returns:
            Number of memories decayed
        """
        cutoff = datetime.utcnow() - timedelta(days=days_threshold)
        
        async with aiosqlite.connect(self.db_path) as db:
            # Reduce importance of stale low-access memories
            result = await db.execute("""
                UPDATE memories 
                SET importance = MAX(1, importance - 1)
                WHERE user_id = ? 
                AND last_accessed < ?
                AND access_count < 3
                AND importance > 1
            """, (user_id, cutoff.isoformat()))
            
            await db.commit()
            
            decayed = result.rowcount
            if decayed:
                logger.info(f"Decayed {decayed} memories for user {user_id}")
            
            return decayed

    def get_metrics(self) -> Dict[str, Any]:
        """Return memory bank metrics for observability."""
        return {
            "service": "long_term_memory",
            "memories_created": self._memories_created,
            "memories_retrieved": self._memories_retrieved,
            "searches_performed": self._searches_performed,
            "cache_size": len(self._cache)
        }

