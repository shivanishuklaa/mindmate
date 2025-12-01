"""
Session Management Module

Provides in-memory session tracking for conversations, including:
- Session creation and management
- Message history with context windowing
- Session state and metadata
- Context compaction for long conversations
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import google.generativeai as genai
from opentelemetry import trace

try:
    from config import config
except ImportError:
    from ..config import config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class MessageRole(Enum):
    """Roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionStatus(Enum):
    """Session lifecycle status."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    EXPIRED = "expired"


@dataclass
class SessionMessage:
    """
    A message in a session conversation.
    
    Attributes:
        id: Unique message identifier
        role: Who sent the message
        content: Message content
        agent_type: Which agent processed/generated this
        metadata: Additional message data
        timestamp: When the message was created
    """
    role: MessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for LLM context."""
        return {
            "role": self.role.value,
            "content": self.content,
            "agent": self.agent_type,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Session:
    """
    A user session containing conversation state.
    
    Attributes:
        id: Unique session identifier
        user_id: User this session belongs to
        status: Current session status
        messages: Conversation history
        context: Session context/state
        created_at: When the session started
        updated_at: Last activity time
        compacted_summary: Summary of compacted history
    """
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: SessionStatus = SessionStatus.ACTIVE
    messages: List[SessionMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    compacted_summary: Optional[str] = None
    
    # Tracking
    emotion_history: List[str] = field(default_factory=list)
    mood_trend: Optional[str] = None
    topics_covered: List[str] = field(default_factory=list)
    risk_level: str = "none"
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        agent_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionMessage:
        """Add a message to the session."""
        message = SessionMessage(
            role=role,
            content=content,
            agent_type=agent_type,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message
    
    def get_conversation_history(
        self,
        limit: Optional[int] = None,
        include_system: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for LLM context.
        
        Args:
            limit: Maximum messages to return
            include_system: Whether to include system messages
            
        Returns:
            List of message dictionaries
        """
        messages = self.messages
        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        
        if limit:
            messages = messages[-limit:]
        
        return [m.to_dict() for m in messages]
    
    def get_context_with_summary(self) -> List[Dict[str, str]]:
        """Get context including compacted summary if available."""
        history = []
        
        if self.compacted_summary:
            history.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.compacted_summary}"
            })
        
        # Add recent messages within context window
        recent = self.get_conversation_history(limit=config.context_window_size)
        history.extend(recent)
        
        return history
    
    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)
    
    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status == SessionStatus.ACTIVE
    
    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / 60


class SessionService:
    """
    In-memory session management service.
    
    Provides:
    - Session creation and retrieval
    - Message management
    - Context compaction
    - Session lifecycle management
    """
    
    def __init__(self):
        """Initialize the session service."""
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_sessions = 0
        self._active_sessions = 0
        self._total_messages = 0
        self._compactions_performed = 0
        
        logger.info("SessionService initialized")

    async def create_session(
        self,
        user_id: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Create a new session for a user.
        
        Args:
            user_id: User to create session for
            initial_context: Optional initial context
            
        Returns:
            New session
        """
        async with self._lock:
            session = Session(
                user_id=user_id,
                context=initial_context or {}
            )
            
            self._sessions[session.id] = session
            
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session.id)
            
            self._total_sessions += 1
            self._active_sessions += 1
            
            logger.info(f"Created session {session.id} for user {user_id}")
            
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session to retrieve
            
        Returns:
            Session or None if not found
        """
        return self._sessions.get(session_id)

    async def get_user_sessions(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[Session]:
        """
        Get all sessions for a user.
        
        Args:
            user_id: User to get sessions for
            active_only: Only return active sessions
            
        Returns:
            List of sessions
        """
        session_ids = self._user_sessions.get(user_id, [])
        sessions = [self._sessions[sid] for sid in session_ids if sid in self._sessions]
        
        if active_only:
            sessions = [s for s in sessions if s.is_active]
        
        return sessions

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        agent_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SessionMessage]:
        """
        Add a message to a session.
        
        Args:
            session_id: Session to add message to
            role: Message role
            content: Message content
            agent_type: Agent that processed/generated the message
            metadata: Additional metadata
            
        Returns:
            Created message or None if session not found
        """
        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return None
        
        message = session.add_message(role, content, agent_type, metadata)
        self._total_messages += 1
        
        # Check if compaction is needed
        if session.message_count >= config.compaction_threshold:
            await self.compact_context(session_id)
        
        return message

    async def update_context(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update session context.
        
        Args:
            session_id: Session to update
            updates: Context updates to apply
            
        Returns:
            True if updated successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session.context.update(updates)
        session.updated_at = datetime.utcnow()
        
        return True

    async def compact_context(self, session_id: str) -> bool:
        """
        Compact conversation history to reduce context size.
        
        Uses LLM to summarize older messages while keeping recent ones.
        
        Args:
            session_id: Session to compact
            
        Returns:
            True if compaction successful
        """
        with tracer.start_as_current_span("session_compact") as span:
            span.set_attribute("session_id", session_id)
            
            session = await self.get_session(session_id)
            if not session or session.message_count < config.compaction_threshold:
                return False
            
            # Keep recent messages, summarize older ones
            keep_count = config.context_window_size
            to_summarize = session.messages[:-keep_count]
            
            if not to_summarize:
                return False
            
            # Build summary prompt
            conversation_text = "\n".join([
                f"{m.role.value}: {m.content}" for m in to_summarize
            ])
            
            summary_prompt = f"""
Summarize this conversation for context continuity. Focus on:
1. Key emotional themes discussed
2. Important insights or breakthroughs
3. Any safety concerns mentioned
4. Coping strategies or techniques explored
5. User's current emotional state

Conversation:
{conversation_text}

Provide a concise summary (2-3 paragraphs) that captures the essential context.
"""
            
            try:
                # Generate summary using Gemini
                genai.configure(api_key=config.google_api_key)
                model = genai.GenerativeModel(config.gemini_model)
                response = await asyncio.to_thread(
                    model.generate_content,
                    summary_prompt
                )
                
                summary = response.text
                
                # Update session
                session.compacted_summary = summary
                session.messages = session.messages[-keep_count:]
                session.updated_at = datetime.utcnow()
                
                self._compactions_performed += 1
                
                logger.info(
                    f"Compacted session {session_id}: "
                    f"summarized {len(to_summarize)} messages"
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Context compaction failed: {e}")
                return False

    async def end_session(
        self,
        session_id: str,
        generate_summary: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        End a session and optionally generate a final summary.
        
        Args:
            session_id: Session to end
            generate_summary: Whether to generate a session summary
            
        Returns:
            Session summary if generated
        """
        session = await self.get_session(session_id)
        if not session:
            return None
        
        session.status = SessionStatus.ENDED
        self._active_sessions -= 1
        
        summary = None
        if generate_summary and session.message_count > 2:
            summary = await self._generate_session_summary(session)
        
        logger.info(f"Ended session {session_id} (duration: {session.duration_minutes:.1f} min)")
        
        return summary

    async def _generate_session_summary(self, session: Session) -> Dict[str, Any]:
        """Generate a summary of the session."""
        try:
            conversation = "\n".join([
                f"{m.role.value}: {m.content}" for m in session.messages
            ])
            
            summary_prompt = f"""
Generate a session summary for mental health record keeping.
Include:
1. Primary topics discussed
2. User's emotional state (beginning vs end)
3. Key insights or progress
4. Any concerning patterns
5. Suggested follow-up topics

Conversation:
{conversation}

Format as structured summary.
"""
            
            genai.configure(api_key=config.google_api_key)
            model = genai.GenerativeModel(config.gemini_model)
            response = await asyncio.to_thread(
                model.generate_content,
                summary_prompt
            )
            
            return {
                "session_id": session.id,
                "user_id": session.user_id,
                "duration_minutes": session.duration_minutes,
                "message_count": session.message_count,
                "summary": response.text,
                "emotion_history": session.emotion_history,
                "topics": session.topics_covered,
                "ended_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Session summary generation failed: {e}")
            return {
                "session_id": session.id,
                "error": str(e)
            }

    async def cleanup_expired_sessions(
        self,
        max_age_minutes: Optional[int] = None
    ) -> int:
        """
        Clean up expired sessions.
        
        Args:
            max_age_minutes: Maximum session age (defaults to config)
            
        Returns:
            Number of sessions cleaned up
        """
        max_age = max_age_minutes or config.session_timeout_minutes
        cutoff = datetime.utcnow() - timedelta(minutes=max_age)
        
        cleaned = 0
        async with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.updated_at < cutoff and session.is_active
            ]
            
            for session_id in expired:
                session = self._sessions[session_id]
                session.status = SessionStatus.EXPIRED
                self._active_sessions -= 1
                cleaned += 1
        
        if cleaned:
            logger.info(f"Cleaned up {cleaned} expired sessions")
        
        return cleaned

    def get_metrics(self) -> Dict[str, Any]:
        """Return service metrics for observability."""
        return {
            "service": "session_service",
            "total_sessions": self._total_sessions,
            "active_sessions": self._active_sessions,
            "total_messages": self._total_messages,
            "compactions_performed": self._compactions_performed
        }

