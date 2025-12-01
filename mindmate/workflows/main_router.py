"""
Main Router Module

Provides intelligent message routing and workflow orchestration for
the MindMate multi-agent system. Implements sequential, parallel,
and loop patterns for agent coordination.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import google.generativeai as genai
from opentelemetry import trace

try:
    from agents import (
        BaseAgent, AgentType, AgentResponse,
        EmotionAgent, CBTAgent, CrisisAgent, KnowledgeAgent
    )
    from memory import SessionService, LongTermMemory, Session, MessageRole
    from tools import JournalMCPTool, MoodTracker, EmergencyAPI
    from config import config
except ImportError:
    from ..agents import (
        BaseAgent, AgentType, AgentResponse,
        EmotionAgent, CBTAgent, CrisisAgent, KnowledgeAgent
    )
    from ..memory import SessionService, LongTermMemory, Session, MessageRole
    from ..tools import JournalMCPTool, MoodTracker, EmergencyAPI
    from ..config import config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class WorkflowPattern(Enum):
    """Workflow execution patterns."""
    SINGLE = "single"  # Single agent handles the request
    SEQUENTIAL = "sequential"  # Multiple agents in sequence
    PARALLEL = "parallel"  # Multiple agents in parallel
    LOOP = "loop"  # Iterative processing until condition met


class IntentType(Enum):
    """User intent classifications."""
    EMOTIONAL_SUPPORT = "emotional_support"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    INFORMATION_SEEKING = "information_seeking"
    CRISIS = "crisis"
    JOURNALING = "journaling"
    MOOD_TRACKING = "mood_tracking"
    GENERAL_CHECK_IN = "general_check_in"
    UNCLEAR = "unclear"


@dataclass
class RoutingDecision:
    """
    Represents a routing decision made by the router.
    
    Attributes:
        primary_agent: Main agent to handle the request
        supporting_agents: Additional agents to consult
        workflow_pattern: How to execute the workflow
        intent: Detected user intent
        confidence: Confidence in the routing decision
        reasoning: Why this routing was chosen
    """
    primary_agent: AgentType
    intent: IntentType
    confidence: float
    reasoning: str
    supporting_agents: List[AgentType] = field(default_factory=list)
    workflow_pattern: WorkflowPattern = WorkflowPattern.SINGLE


@dataclass
class WorkflowResult:
    """
    Result of workflow execution.
    
    Attributes:
        response: Final response to user
        agents_used: Agents that participated
        workflow_pattern: Pattern used
        processing_time_ms: Total processing time
        metadata: Additional result data
    """
    response: str
    agents_used: List[AgentType]
    workflow_pattern: WorkflowPattern
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MainRouter:
    """
    Intelligent message router and workflow orchestrator.
    
    Responsibilities:
    - Analyze user messages to determine intent
    - Route to appropriate agent(s)
    - Coordinate multi-agent workflows
    - Handle crisis escalation
    - Manage session and memory context
    
    Workflow Patterns:
    - SINGLE: Direct routing to one agent
    - SEQUENTIAL: Chain of agents (e.g., Crisis -> Emotion)
    - PARALLEL: Multiple agents consulted simultaneously
    - LOOP: Iterative refinement (e.g., CBT thought record)
    """
    
    # Intent detection keywords
    INTENT_KEYWORDS = {
        IntentType.EMOTIONAL_SUPPORT: [
            "feeling", "feel", "sad", "happy", "angry", "anxious", "scared",
            "lonely", "overwhelmed", "stressed", "hurt", "upset", "crying"
        ],
        IntentType.COGNITIVE_RESTRUCTURING: [
            "think", "thought", "believe", "should", "must", "always", "never",
            "can't", "worthless", "stupid", "failure", "perspective", "reframe"
        ],
        IntentType.INFORMATION_SEEKING: [
            "what is", "how does", "explain", "tell me about", "learn",
            "information", "understand", "therapy", "technique", "cbt", "dbt"
        ],
        IntentType.JOURNALING: [
            "journal", "write", "record", "entry", "diary", "log my",
            "note", "document"
        ],
        IntentType.MOOD_TRACKING: [
            "mood", "track", "rate", "scale", "today i feel", "my mood",
            "energy level", "how i'm doing"
        ],
        IntentType.CRISIS: [
            "suicide", "kill", "die", "end my life", "hurt myself",
            "self-harm", "cutting", "overdose", "can't go on", "no point"
        ]
    }

    def __init__(
        self,
        session_service: SessionService,
        long_term_memory: LongTermMemory
    ):
        """
        Initialize the main router.
        
        Args:
            session_service: Session management service
            long_term_memory: Long-term memory bank
        """
        self.session_service = session_service
        self.long_term_memory = long_term_memory
        
        # Initialize tools
        self.journal_tool = JournalMCPTool()
        self.mood_tracker = MoodTracker()
        self.emergency_api = EmergencyAPI()
        
        # Initialize agents with tools
        self.agents: Dict[AgentType, BaseAgent] = {
            AgentType.EMOTION: EmotionAgent(tools=[
                self.mood_tracker.write,
                self.journal_tool.execute
            ]),
            AgentType.CBT: CBTAgent(tools=[
                self.journal_tool.execute
            ]),
            AgentType.CRISIS: CrisisAgent(tools=[
                self.emergency_api.create_alert,
                self.emergency_api.get_crisis_resources
            ]),
            AgentType.KNOWLEDGE: KnowledgeAgent(tools=[
                self.journal_tool.execute
            ])
        }
        
        # Register A2A message handlers
        self._setup_a2a_handlers()
        
        # Initialize routing model
        genai.configure(api_key=config.google_api_key)
        self.router_model = genai.GenerativeModel(
            model_name=config.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,  # Lower for consistent routing
                max_output_tokens=256
            )
        )
        
        # Metrics
        self._requests_routed = 0
        self._routes_by_agent: Dict[str, int] = {}
        self._patterns_used: Dict[str, int] = {}
        self._crisis_escalations = 0
        
        logger.info("MainRouter initialized with all agents")

    def _setup_a2a_handlers(self) -> None:
        """Set up Agent-to-Agent message handlers."""
        # Crisis agent can receive escalations from any agent
        for agent_type, agent in self.agents.items():
            if agent_type != AgentType.CRISIS:
                agent.register_message_handler(
                    AgentType.CRISIS,
                    self._handle_crisis_message
                )

    async def _handle_crisis_message(self, message) -> None:
        """Handle crisis escalation messages."""
        logger.warning(f"Crisis escalation received from {message.source.value}")
        self._crisis_escalations += 1
        
        # Process through crisis agent
        crisis_agent = self.agents[AgentType.CRISIS]
        # The crisis agent will handle the message content

    def detect_intent(self, text: str) -> Tuple[IntentType, float]:
        """
        Detect user intent from message text.
        
        Args:
            text: User's message
            
        Returns:
            Tuple of (intent, confidence)
        """
        text_lower = text.lower()
        scores: Dict[IntentType, int] = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return IntentType.UNCLEAR, 0.3
        
        # Crisis always takes priority
        if IntentType.CRISIS in scores:
            return IntentType.CRISIS, 0.95
        
        best_intent = max(scores, key=scores.get)
        max_score = scores[best_intent]
        
        # Normalize confidence
        confidence = min(0.9, 0.4 + (max_score * 0.15))
        
        return best_intent, confidence

    async def route(
        self,
        user_input: str,
        session_id: str
    ) -> RoutingDecision:
        """
        Make a routing decision for a user message.
        
        Args:
            user_input: User's message
            session_id: Current session ID
            
        Returns:
            RoutingDecision
        """
        with tracer.start_as_current_span("router_route") as span:
            span.set_attribute("session_id", session_id)
            
            # Quick intent detection
            intent, confidence = self.detect_intent(user_input)
            span.set_attribute("intent", intent.value)
            span.set_attribute("confidence", confidence)
            
            # Crisis always goes to crisis agent first
            if intent == IntentType.CRISIS:
                return RoutingDecision(
                    primary_agent=AgentType.CRISIS,
                    supporting_agents=[AgentType.EMOTION],
                    workflow_pattern=WorkflowPattern.SEQUENTIAL,
                    intent=intent,
                    confidence=0.95,
                    reasoning="Crisis signals detected - prioritizing safety assessment"
                )
            
            # Map intents to agents
            intent_agent_map = {
                IntentType.EMOTIONAL_SUPPORT: AgentType.EMOTION,
                IntentType.COGNITIVE_RESTRUCTURING: AgentType.CBT,
                IntentType.INFORMATION_SEEKING: AgentType.KNOWLEDGE,
                IntentType.JOURNALING: AgentType.EMOTION,
                IntentType.MOOD_TRACKING: AgentType.EMOTION,
                IntentType.GENERAL_CHECK_IN: AgentType.EMOTION,
                IntentType.UNCLEAR: AgentType.EMOTION
            }
            
            primary_agent = intent_agent_map.get(intent, AgentType.EMOTION)
            
            # Determine supporting agents and pattern
            supporting_agents = []
            pattern = WorkflowPattern.SINGLE
            
            if intent == IntentType.COGNITIVE_RESTRUCTURING:
                supporting_agents = [AgentType.EMOTION]
                pattern = WorkflowPattern.SEQUENTIAL
            elif intent == IntentType.GENERAL_CHECK_IN:
                supporting_agents = [AgentType.KNOWLEDGE]
                pattern = WorkflowPattern.PARALLEL
            
            return RoutingDecision(
                primary_agent=primary_agent,
                supporting_agents=supporting_agents,
                workflow_pattern=pattern,
                intent=intent,
                confidence=confidence,
                reasoning=f"Intent '{intent.value}' mapped to {primary_agent.value} agent"
            )

    async def execute_workflow(
        self,
        user_input: str,
        session: Session,
        routing: RoutingDecision
    ) -> WorkflowResult:
        """
        Execute the workflow based on routing decision.
        
        Args:
            user_input: User's message
            session: Current session
            routing: Routing decision
            
        Returns:
            WorkflowResult
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("execute_workflow") as span:
            span.set_attribute("pattern", routing.workflow_pattern.value)
            span.set_attribute("primary_agent", routing.primary_agent.value)
            
            # Get conversation history
            history = session.get_context_with_summary()
            context = session.context.copy()
            
            # Get long-term memory context
            memory_context = await self.long_term_memory.get_context_for_session(
                session.user_id,
                current_topic=user_input[:50]
            )
            context["memory_context"] = memory_context
            
            agents_used = []
            responses: List[AgentResponse] = []
            
            if routing.workflow_pattern == WorkflowPattern.SINGLE:
                # Single agent handles the request
                response = await self._execute_single(
                    routing.primary_agent,
                    user_input,
                    context,
                    history
                )
                responses.append(response)
                agents_used.append(routing.primary_agent)
                
            elif routing.workflow_pattern == WorkflowPattern.SEQUENTIAL:
                # Execute agents in sequence
                responses, agents_used = await self._execute_sequential(
                    [routing.primary_agent] + routing.supporting_agents,
                    user_input,
                    context,
                    history
                )
                
            elif routing.workflow_pattern == WorkflowPattern.PARALLEL:
                # Execute agents in parallel
                responses, agents_used = await self._execute_parallel(
                    [routing.primary_agent] + routing.supporting_agents,
                    user_input,
                    context,
                    history
                )
                
            elif routing.workflow_pattern == WorkflowPattern.LOOP:
                # Execute iterative workflow
                responses, agents_used = await self._execute_loop(
                    routing.primary_agent,
                    user_input,
                    context,
                    history,
                    max_iterations=3
                )
            
            # Synthesize final response
            final_response = await self._synthesize_response(
                responses,
                routing
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self._requests_routed += 1
            for agent in agents_used:
                self._routes_by_agent[agent.value] = \
                    self._routes_by_agent.get(agent.value, 0) + 1
            self._patterns_used[routing.workflow_pattern.value] = \
                self._patterns_used.get(routing.workflow_pattern.value, 0) + 1
            
            return WorkflowResult(
                response=final_response,
                agents_used=agents_used,
                workflow_pattern=routing.workflow_pattern,
                processing_time_ms=processing_time,
                metadata={
                    "intent": routing.intent.value,
                    "confidence": routing.confidence,
                    "escalated": any(r.escalate for r in responses),
                    "suggested_actions": [
                        a for r in responses for a in r.suggested_actions
                    ]
                }
            )

    async def _execute_single(
        self,
        agent_type: AgentType,
        user_input: str,
        context: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> AgentResponse:
        """Execute a single agent."""
        agent = self.agents[agent_type]
        return await agent.process(user_input, context, history)

    async def _execute_sequential(
        self,
        agent_types: List[AgentType],
        user_input: str,
        context: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> Tuple[List[AgentResponse], List[AgentType]]:
        """Execute agents in sequence, passing context between them."""
        responses = []
        accumulated_context = context.copy()
        
        for agent_type in agent_types:
            agent = self.agents[agent_type]
            
            # Check for pause
            await agent.check_pause()
            
            response = await agent.process(
                user_input,
                accumulated_context,
                history
            )
            responses.append(response)
            
            # Pass insights to next agent
            accumulated_context["previous_agent"] = agent_type.value
            accumulated_context["previous_response"] = response.content[:500]
            accumulated_context["escalate"] = response.escalate
            
            # If escalation, break the chain
            if response.escalate:
                break
        
        return responses, agent_types[:len(responses)]

    async def _execute_parallel(
        self,
        agent_types: List[AgentType],
        user_input: str,
        context: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> Tuple[List[AgentResponse], List[AgentType]]:
        """Execute agents in parallel using asyncio.gather()."""
        tasks = []
        
        for agent_type in agent_types:
            agent = self.agents[agent_type]
            task = agent.process(user_input, context, history)
            tasks.append(task)
        
        # Execute all agents in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        valid_agents = []
        
        for i, response in enumerate(responses):
            if isinstance(response, AgentResponse):
                valid_responses.append(response)
                valid_agents.append(agent_types[i])
            else:
                logger.error(f"Agent {agent_types[i].value} failed: {response}")
        
        return valid_responses, valid_agents

    async def _execute_loop(
        self,
        agent_type: AgentType,
        user_input: str,
        context: Dict[str, Any],
        history: List[Dict[str, str]],
        max_iterations: int = 3
    ) -> Tuple[List[AgentResponse], List[AgentType]]:
        """Execute agent in a loop until condition is met."""
        agent = self.agents[agent_type]
        responses = []
        
        for i in range(max_iterations):
            # Check for pause
            await agent.check_pause()
            
            response = await agent.process(user_input, context, history)
            responses.append(response)
            
            # Check termination conditions
            if response.confidence > 0.9:
                break
            if response.escalate:
                break
            if "complete" in response.suggested_actions:
                break
            
            # Update context for next iteration
            context["iteration"] = i + 1
            context["previous_response"] = response.content
        
        return responses, [agent_type]

    async def _synthesize_response(
        self,
        responses: List[AgentResponse],
        routing: RoutingDecision
    ) -> str:
        """
        Synthesize a final response from multiple agent responses.
        
        For single agent, returns the response directly.
        For multiple agents, combines insights appropriately.
        """
        if not responses:
            return "I'm here to help. Could you tell me more about what's on your mind?"
        
        if len(responses) == 1:
            return responses[0].content
        
        # For multiple responses, use the primary agent's response
        # but incorporate insights from supporting agents
        primary_response = responses[0].content
        
        # Check if any agent escalated
        escalated = any(r.escalate for r in responses)
        if escalated:
            # Crisis response takes priority
            crisis_responses = [r for r in responses if r.escalate]
            if crisis_responses:
                return crisis_responses[0].content
        
        # For parallel execution, synthesize
        if routing.workflow_pattern == WorkflowPattern.PARALLEL:
            # Combine insights
            combined = primary_response
            for response in responses[1:]:
                if response.metadata.get("add_insight"):
                    combined += f"\n\n{response.content}"
            return combined
        
        return primary_response

    async def process_message(
        self,
        user_input: str,
        session_id: str
    ) -> WorkflowResult:
        """
        Main entry point for processing a user message.
        
        Args:
            user_input: User's message
            session_id: Current session ID
            
        Returns:
            WorkflowResult
        """
        with tracer.start_as_current_span("process_message") as span:
            span.set_attribute("session_id", session_id)
            
            # Get or create session
            session = await self.session_service.get_session(session_id)
            if not session:
                logger.warning(f"Session {session_id} not found")
                return WorkflowResult(
                    response="I'm sorry, there was an issue with your session. Please try again.",
                    agents_used=[],
                    workflow_pattern=WorkflowPattern.SINGLE,
                    processing_time_ms=0
                )
            
            # Add user message to session
            await self.session_service.add_message(
                session_id,
                MessageRole.USER,
                user_input
            )
            
            # Route the message
            routing = await self.route(user_input, session_id)
            
            logger.info(
                f"Routing: {routing.intent.value} -> "
                f"{routing.primary_agent.value} ({routing.workflow_pattern.value})"
            )
            
            # Execute workflow
            result = await self.execute_workflow(user_input, session, routing)
            
            # Add assistant response to session
            await self.session_service.add_message(
                session_id,
                MessageRole.ASSISTANT,
                result.response,
                agent_type=routing.primary_agent.value,
                metadata=result.metadata
            )
            
            # Update session context
            await self.session_service.update_context(
                session_id,
                {
                    "last_intent": routing.intent.value,
                    "last_agent": routing.primary_agent.value,
                    "risk_level": "high" if result.metadata.get("escalated") else "normal"
                }
            )
            
            return result

    async def handle_tool_request(
        self,
        tool_name: str,
        user_id: str,
        session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle a direct tool request.
        
        Args:
            tool_name: Name of the tool to execute
            user_id: User making the request
            session_id: Current session ID
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        if tool_name == "journal":
            return await self.journal_tool.execute(
                user_id=user_id,
                **kwargs
            )
        elif tool_name == "mood":
            if kwargs.get("action") == "write":
                return await self.mood_tracker.write(user_id=user_id, **kwargs)
            elif kwargs.get("action") == "read":
                return await self.mood_tracker.read(user_id=user_id, **kwargs)
            elif kwargs.get("action") == "analyze":
                return await self.mood_tracker.analyze(user_id=user_id, **kwargs)
        elif tool_name == "emergency":
            return await self.emergency_api.create_alert(
                user_id=user_id,
                session_id=session_id,
                **kwargs
            )
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def get_metrics(self) -> Dict[str, Any]:
        """Return router metrics for observability."""
        agent_metrics = {
            agent_type.value: agent.get_metrics() if hasattr(agent, 'get_metrics') else {}
            for agent_type, agent in self.agents.items()
        }
        
        return {
            "router": {
                "requests_routed": self._requests_routed,
                "routes_by_agent": self._routes_by_agent,
                "patterns_used": self._patterns_used,
                "crisis_escalations": self._crisis_escalations
            },
            "agents": agent_metrics,
            "session_service": self.session_service.get_metrics(),
            "memory": self.long_term_memory.get_metrics(),
            "tools": {
                "journal": self.journal_tool.get_tool_definition(),
                "mood_tracker": self.mood_tracker.get_metrics(),
                "emergency": self.emergency_api.get_metrics()
            }
        }

