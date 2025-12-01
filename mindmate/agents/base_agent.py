"""
Base Agent Module

Provides the foundational class for all MindMate agents.
Implements common functionality for LLM interaction, A2A messaging,
observability, and safety guardrails.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable

import google.generativeai as genai
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

try:
    from config import config
except ImportError:
    from ..config import config

# Configure logging
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class AgentType(Enum):
    """Enumeration of available agent types."""
    EMOTION = "emotion"
    CBT = "cbt"
    CRISIS = "crisis"
    KNOWLEDGE = "knowledge"
    ROUTER = "router"


class MessagePriority(Enum):
    """Priority levels for A2A messaging."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4  # For crisis situations


@dataclass
class AgentMessage:
    """
    Message structure for Agent-to-Agent (A2A) communication.
    
    Attributes:
        id: Unique message identifier
        source: Agent type that sent the message
        target: Agent type that should receive the message
        content: Message content/payload
        priority: Message priority level
        metadata: Additional context and data
        timestamp: When the message was created
        requires_response: Whether a response is expected
        correlation_id: ID linking related messages
    """
    source: AgentType
    target: AgentType
    content: str
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AgentResponse:
    """
    Structured response from an agent.
    
    Attributes:
        agent_type: Which agent generated this response
        content: The main response text
        confidence: Agent's confidence in the response (0-1)
        suggested_actions: Recommended follow-up actions
        escalate: Whether to escalate to crisis handling
        metadata: Additional response data
        tools_used: List of tools that were invoked
        processing_time_ms: Time taken to generate response
    """
    agent_type: AgentType
    content: str
    confidence: float = 1.0
    suggested_actions: List[str] = field(default_factory=list)
    escalate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all MindMate agents.
    
    Provides:
    - LLM integration with Gemini models
    - A2A messaging infrastructure
    - Observability (logging, tracing, metrics)
    - Safety guardrails for mental health context
    - Tool registration and execution
    """
    
    # Safety disclaimer to append to all responses
    SAFETY_DISCLAIMER = (
        "\n\n---\n"
        "💡 *I'm an AI assistant providing emotional support, not a licensed "
        "mental health professional. If you're experiencing a crisis, please "
        "contact emergency services or a crisis helpline.*"
    )
    
    # Crisis keywords that trigger immediate escalation
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die",
        "self-harm", "hurt myself", "cutting", "overdose",
        "no reason to live", "better off dead", "can't go on"
    ]
    
    def __init__(
        self,
        agent_type: AgentType,
        system_prompt: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Callable]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_type: The type of this agent
            system_prompt: System instructions for the LLM
            model_name: Gemini model to use (defaults to config)
            temperature: Model temperature (defaults to config)
            tools: List of tool functions to register
        """
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.model_name = model_name or config.gemini_model
        self.temperature = temperature or config.gemini_temperature
        self.tools = tools or []
        
        # Initialize Gemini
        genai.configure(api_key=config.google_api_key)
        
        # Create model with safety settings for mental health context
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self._build_system_instruction(),
            safety_settings=self._get_safety_settings(),
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=config.gemini_max_tokens,
                top_p=0.95,
            )
        )
        
        # A2A message queue
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._message_handlers: Dict[AgentType, Callable[[AgentMessage], Awaitable[None]]] = {}
        
        # State management
        self._is_paused = False
        self._current_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized {agent_type.value} agent with model {self.model_name}")
    
    def _build_system_instruction(self) -> str:
        """Build the complete system instruction with safety context."""
        safety_context = """
CRITICAL SAFETY GUIDELINES:
1. You are an AI providing emotional support, NOT a licensed therapist or medical professional
2. Never provide medical diagnoses or medication advice
3. Always encourage professional help for serious concerns
4. If user expresses suicidal ideation or self-harm, immediately flag for crisis escalation
5. Maintain a warm, non-judgmental, empathetic tone
6. Respect user boundaries and autonomy
7. Do not make promises you cannot keep
8. Acknowledge the limitations of AI support
"""
        return f"{safety_context}\n\n{self.system_prompt}"
    
    def _get_safety_settings(self) -> List[Dict[str, Any]]:
        """Configure Gemini safety settings appropriate for mental health support."""
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"  # Allow discussion of mental health topics
            }
        ]
    
    def check_for_crisis(self, text: str) -> bool:
        """
        Quick check for crisis keywords in user input.
        
        Args:
            text: User's message text
            
        Returns:
            True if crisis keywords detected
        """
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.CRISIS_KEYWORDS)
    
    @abstractmethod
    async def process(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> AgentResponse:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's message
            session_context: Current session data
            conversation_history: Previous messages in the conversation
            
        Returns:
            AgentResponse with the agent's reply
        """
        pass
    
    async def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response using the Gemini model.
        
        Args:
            prompt: The prompt to send to the model
            conversation_history: Optional conversation context
            
        Returns:
            Generated response text
        """
        with tracer.start_as_current_span(f"{self.agent_type.value}_generate") as span:
            span.set_attribute("agent.type", self.agent_type.value)
            span.set_attribute("prompt.length", len(prompt))
            
            try:
                # Build conversation context
                messages = []
                if conversation_history:
                    for msg in conversation_history[-config.context_window_size:]:
                        role = "user" if msg.get("role") == "user" else "model"
                        messages.append({"role": role, "parts": [msg["content"]]})
                
                messages.append({"role": "user", "parts": [prompt]})
                
                # Generate response
                chat = self.model.start_chat(history=messages[:-1])
                response = await asyncio.to_thread(
                    chat.send_message,
                    prompt
                )
                
                result = response.text
                span.set_attribute("response.length", len(result))
                span.set_status(Status(StatusCode.OK))
                
                logger.debug(f"{self.agent_type.value} generated response: {result[:100]}...")
                return result
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Error generating response: {e}")
                raise
    
    # ==================== A2A Messaging ====================
    
    async def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to another agent.
        
        Args:
            message: The message to send
        """
        logger.info(
            f"A2A: {message.source.value} -> {message.target.value} "
            f"(priority: {message.priority.name})"
        )
        
        # If we have a handler for the target, invoke it
        if message.target in self._message_handlers:
            await self._message_handlers[message.target](message)
        else:
            # Queue for later processing
            await self._message_queue.put(message)
    
    def register_message_handler(
        self,
        target_type: AgentType,
        handler: Callable[[AgentMessage], Awaitable[None]]
    ) -> None:
        """
        Register a handler for messages to a specific agent type.
        
        Args:
            target_type: Agent type to handle messages for
            handler: Async function to process messages
        """
        self._message_handlers[target_type] = handler
    
    async def receive_message(self, timeout: float = 5.0) -> Optional[AgentMessage]:
        """
        Receive a message from the queue.
        
        Args:
            timeout: How long to wait for a message
            
        Returns:
            Received message or None if timeout
        """
        try:
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    # ==================== Pause/Resume (Long-running operations) ====================
    
    async def pause(self) -> None:
        """Pause the agent's current processing."""
        self._is_paused = True
        logger.info(f"{self.agent_type.value} agent paused")
    
    async def resume(self) -> None:
        """Resume the agent's processing."""
        self._is_paused = False
        logger.info(f"{self.agent_type.value} agent resumed")
    
    @property
    def is_paused(self) -> bool:
        """Check if agent is currently paused."""
        return self._is_paused
    
    async def check_pause(self) -> None:
        """Check and wait if paused. Call during long operations."""
        while self._is_paused:
            await asyncio.sleep(0.1)
    
    # ==================== Tool Execution ====================
    
    async def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        Execute a registered tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        with tracer.start_as_current_span(f"tool_{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            
            for tool in self.tools:
                if tool.__name__ == tool_name:
                    logger.info(f"Executing tool: {tool_name}")
                    result = await tool(**kwargs) if asyncio.iscoroutinefunction(tool) else tool(**kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
            
            span.set_status(Status(StatusCode.ERROR, "Tool not found"))
            raise ValueError(f"Tool '{tool_name}' not found")
    
    def add_safety_disclaimer(self, response: str) -> str:
        """Add safety disclaimer to response if not already present."""
        if "I'm an AI assistant" not in response:
            return response + self.SAFETY_DISCLAIMER
        return response

