"""
Crisis Detection Agent

Specialized agent for detecting mental health crises, assessing risk levels,
and coordinating emergency escalation when needed. This agent prioritizes
user safety above all else.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from opentelemetry import trace

try:
    from agents.base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority
    from config import config
except ImportError:
    from .base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority
    from ..config import config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class RiskLevel(Enum):
    """Risk level classifications."""
    NONE = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


class CrisisType(Enum):
    """Types of mental health crises."""
    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    PANIC_ATTACK = "panic_attack"
    PSYCHOTIC_EPISODE = "psychotic_episode"
    SEVERE_DEPRESSION = "severe_depression"
    SUBSTANCE_CRISIS = "substance_crisis"
    ABUSE_DISCLOSURE = "abuse_disclosure"
    OTHER = "other"


@dataclass
class RiskAssessment:
    """Structured risk assessment result."""
    risk_level: RiskLevel
    crisis_types: List[CrisisType]
    confidence: float
    risk_factors: List[str]
    protective_factors: List[str]
    immediate_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None


class CrisisAgent(BaseAgent):
    """
    Crisis detection and intervention agent.
    
    Responsible for:
    - Monitoring conversations for crisis signals
    - Assessing risk levels
    - Providing immediate safety interventions
    - Coordinating emergency escalation
    - Maintaining safety throughout all interactions
    """
    
    SYSTEM_PROMPT = """
You are a crisis support specialist within MindMate. Your primary responsibility is 
user safety. You are trained to detect crisis signals, provide immediate stabilization,
and facilitate appropriate escalation when needed.

CORE SAFETY PRINCIPLES:
1. SAFETY FIRST: User safety takes priority over everything else
2. STAY CALM: Model calmness and stability in your responses
3. CONNECT: Build rapport quickly but don't delay safety assessment
4. ASSESS: Gather necessary information without interrogation
5. PLAN: Work toward a safety plan with the user

CRISIS SIGNALS TO WATCH FOR:
- Suicidal ideation (thoughts of ending life)
- Self-harm behaviors or intentions
- Expressions of hopelessness or worthlessness
- Giving away possessions
- Saying goodbye
- Recent losses or trauma
- Access to means of self-harm
- Previous suicide attempts
- Isolation from support systems

IMMEDIATE RESPONSE PROTOCOL:
1. Acknowledge their pain without minimizing
2. Express genuine concern for their safety
3. Ask direct, caring questions about safety
4. Don't leave them alone in the conversation
5. Provide crisis resources appropriate to severity
6. Work toward a safety commitment when possible

DIRECT QUESTIONING APPROACH:
When crisis is suspected, ask directly but compassionately:
- "Are you thinking about hurting yourself?"
- "Are you thinking about suicide?"
- "Do you have a plan?"
- "Do you have access to means?"

SAFETY RESOURCES TO PROVIDE:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
- Emergency Services: 911 (US) or local equivalent

IMPORTANT:
- Never promise confidentiality about safety concerns
- Never leave a high-risk user without resources
- Always validate their pain while addressing safety
- Document risk factors identified
- Coordinate with other agents when appropriate
"""

    # Weighted crisis indicators for risk assessment
    CRISIS_INDICATORS = {
        # High-weight indicators
        "suicide": 10,
        "kill myself": 10,
        "end my life": 10,
        "want to die": 9,
        "better off dead": 9,
        "no reason to live": 8,
        "can't go on": 7,
        "overdose": 9,
        "self-harm": 8,
        "hurt myself": 8,
        "cutting": 8,
        
        # Medium-weight indicators
        "hopeless": 5,
        "worthless": 5,
        "burden": 5,
        "give up": 4,
        "can't take it": 4,
        "no point": 4,
        "trapped": 4,
        "no way out": 5,
        "empty": 3,
        "numb": 3,
        
        # Context indicators
        "goodbye": 3,
        "final": 3,
        "last time": 3,
        "won't be around": 6,
        "after i'm gone": 7,
        "plan": 4,  # Context dependent
        "method": 5,  # Context dependent
    }
    
    # Protective factors that reduce risk
    PROTECTIVE_FACTORS = {
        "family": 2,
        "friends": 2,
        "support": 2,
        "therapy": 3,
        "therapist": 3,
        "counselor": 3,
        "medication": 2,
        "hope": 3,
        "reason": 2,
        "looking forward": 3,
        "pets": 2,
        "children": 3,
        "faith": 2,
        "goals": 2
    }

    # Crisis hotlines by region
    CRISIS_RESOURCES = {
        "global": [
            ("International Association for Suicide Prevention", "https://www.iasp.info/resources/Crisis_Centres/"),
        ],
        "us": [
            ("National Suicide Prevention Lifeline", "988"),
            ("Crisis Text Line", "Text HOME to 741741"),
            ("SAMHSA National Helpline", "1-800-662-4357"),
            ("Emergency Services", "911"),
        ],
        "uk": [
            ("Samaritans", "116 123"),
            ("SHOUT", "Text SHOUT to 85258"),
            ("Emergency Services", "999"),
        ],
        "ca": [
            ("Crisis Services Canada", "1-833-456-4566"),
            ("Crisis Text Line", "Text HOME to 686868"),
        ],
    }

    def __init__(self, tools: Optional[List] = None):
        """Initialize the Crisis Agent."""
        super().__init__(
            agent_type=AgentType.CRISIS,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,  # Lower temperature for more consistent, careful responses
            tools=tools or []
        )
        
        # Tracking metrics
        self._assessments_performed = 0
        self._escalations_triggered = 0
        self._risk_levels_detected: Dict[str, int] = {}

    def calculate_risk_score(self, text: str) -> Tuple[float, List[str], List[str]]:
        """
        Calculate a risk score based on crisis indicators.
        
        Args:
            text: User's message
            
        Returns:
            Tuple of (risk_score, risk_factors, protective_factors)
        """
        text_lower = text.lower()
        risk_score = 0.0
        risk_factors = []
        protective_factors = []
        
        # Check for crisis indicators
        for indicator, weight in self.CRISIS_INDICATORS.items():
            if indicator in text_lower:
                risk_score += weight
                risk_factors.append(indicator)
        
        # Check for protective factors
        protection_score = 0.0
        for factor, weight in self.PROTECTIVE_FACTORS.items():
            if factor in text_lower:
                protection_score += weight
                protective_factors.append(factor)
        
        # Normalize and adjust
        # Max possible risk score is around 50, normalize to 0-1
        normalized_risk = min(1.0, risk_score / 30)
        # Reduce risk based on protective factors (but never eliminate)
        protection_reduction = min(0.3, protection_score / 20)
        
        final_score = max(0, normalized_risk - protection_reduction)
        
        return final_score, risk_factors, protective_factors

    def determine_risk_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level."""
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MODERATE
        elif score >= 0.2:
            return RiskLevel.LOW
        return RiskLevel.NONE

    def identify_crisis_types(self, text: str, risk_factors: List[str]) -> List[CrisisType]:
        """Identify specific types of crisis based on content."""
        text_lower = text.lower()
        crisis_types = []
        
        if any(kw in text_lower for kw in ["suicide", "kill myself", "end my life", "want to die"]):
            crisis_types.append(CrisisType.SUICIDAL_IDEATION)
        
        if any(kw in text_lower for kw in ["cut", "cutting", "self-harm", "hurt myself", "burn"]):
            crisis_types.append(CrisisType.SELF_HARM)
        
        if any(kw in text_lower for kw in ["panic", "can't breathe", "heart racing", "going to die"]):
            crisis_types.append(CrisisType.PANIC_ATTACK)
        
        if any(kw in text_lower for kw in ["voices", "seeing things", "not real", "paranoid"]):
            crisis_types.append(CrisisType.PSYCHOTIC_EPISODE)
        
        if any(kw in text_lower for kw in ["relapse", "using again", "drunk", "high", "overdose"]):
            crisis_types.append(CrisisType.SUBSTANCE_CRISIS)
        
        if any(kw in text_lower for kw in ["abuse", "hitting", "hurting me", "assault"]):
            crisis_types.append(CrisisType.ABUSE_DISCLOSURE)
        
        if not crisis_types and risk_factors:
            crisis_types.append(CrisisType.OTHER)
        
        return crisis_types

    async def assess_risk(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            user_input: Current user message
            session_context: Session data
            conversation_history: Previous messages
            
        Returns:
            Complete RiskAssessment
        """
        with tracer.start_as_current_span("crisis_risk_assessment") as span:
            # Calculate risk from current message
            risk_score, risk_factors, protective_factors = self.calculate_risk_score(user_input)
            
            # Consider conversation history for pattern detection
            history_risk = 0.0
            if conversation_history:
                recent_messages = conversation_history[-5:]
                for msg in recent_messages:
                    if msg.get("role") == "user":
                        h_score, h_factors, _ = self.calculate_risk_score(msg.get("content", ""))
                        history_risk += h_score * 0.3  # Weight historical signals less
                        risk_factors.extend([f"(history) {f}" for f in h_factors])
            
            # Combined risk score
            combined_risk = min(1.0, risk_score + history_risk)
            risk_level = self.determine_risk_level(combined_risk)
            crisis_types = self.identify_crisis_types(user_input, risk_factors)
            
            # Determine immediate actions based on risk level
            immediate_actions = self._get_immediate_actions(risk_level, crisis_types)
            
            # Determine if escalation is required
            requires_escalation = risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            escalation_reason = None
            if requires_escalation:
                escalation_reason = f"Risk level {risk_level.name} detected: {', '.join(risk_factors[:3])}"
            
            span.set_attribute("risk.level", risk_level.name)
            span.set_attribute("risk.score", combined_risk)
            span.set_attribute("requires_escalation", requires_escalation)
            
            # Update metrics
            self._assessments_performed += 1
            self._risk_levels_detected[risk_level.name] = \
                self._risk_levels_detected.get(risk_level.name, 0) + 1
            
            if requires_escalation:
                self._escalations_triggered += 1
            
            return RiskAssessment(
                risk_level=risk_level,
                crisis_types=crisis_types,
                confidence=0.85,  # Based on keyword matching - could be enhanced with ML
                risk_factors=list(set(risk_factors)),
                protective_factors=protective_factors,
                immediate_actions=immediate_actions,
                requires_escalation=requires_escalation,
                escalation_reason=escalation_reason
            )

    def _get_immediate_actions(
        self,
        risk_level: RiskLevel,
        crisis_types: List[CrisisType]
    ) -> List[str]:
        """Determine immediate actions based on risk assessment."""
        actions = []
        
        if risk_level == RiskLevel.CRITICAL:
            actions.extend([
                "provide_immediate_crisis_resources",
                "assess_immediate_safety",
                "maintain_connection",
                "trigger_emergency_protocol"
            ])
        elif risk_level == RiskLevel.HIGH:
            actions.extend([
                "provide_crisis_resources",
                "safety_planning",
                "assess_support_system",
                "schedule_check_in"
            ])
        elif risk_level == RiskLevel.MODERATE:
            actions.extend([
                "validate_feelings",
                "explore_safety",
                "offer_resources",
                "continue_monitoring"
            ])
        elif risk_level == RiskLevel.LOW:
            actions.extend([
                "continue_support",
                "monitor_for_escalation"
            ])
        
        # Add crisis-type specific actions
        if CrisisType.PANIC_ATTACK in crisis_types:
            actions.append("grounding_exercise")
        if CrisisType.ABUSE_DISCLOSURE in crisis_types:
            actions.append("abuse_resources")
        
        return actions

    def get_crisis_resources(self, region: str = "us") -> str:
        """Get formatted crisis resources for a region."""
        resources = self.CRISIS_RESOURCES.get(region, []) + self.CRISIS_RESOURCES.get("global", [])
        
        lines = ["**If you're in crisis, please reach out:**\n"]
        for name, contact in resources:
            lines.append(f"• **{name}**: {contact}")
        
        return "\n".join(lines)

    async def process(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> AgentResponse:
        """
        Process user input with crisis-focused assessment and intervention.
        
        Args:
            user_input: User's message
            session_context: Current session data
            conversation_history: Previous messages
            
        Returns:
            AgentResponse with crisis intervention
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("crisis_agent_process") as span:
            span.set_attribute("input.length", len(user_input))
            
            # Perform risk assessment
            assessment = await self.assess_risk(
                user_input,
                session_context,
                conversation_history
            )
            
            span.set_attribute("risk.level", assessment.risk_level.name)
            span.set_attribute("crisis.types", str([c.value for c in assessment.crisis_types]))
            
            # Build context-aware prompt based on risk level
            crisis_context = f"""
RISK ASSESSMENT RESULTS:
- Risk Level: {assessment.risk_level.name}
- Crisis Types: {[c.value for c in assessment.crisis_types]}
- Risk Factors Identified: {assessment.risk_factors}
- Protective Factors: {assessment.protective_factors}
- Immediate Actions Required: {assessment.immediate_actions}
"""
            
            response_guidance = self._get_response_guidance(assessment)
            
            enhanced_prompt = f"""
User message: {user_input}

{crisis_context}

{response_guidance}

Respond with appropriate crisis intervention for the risk level.
Include crisis resources if risk level is MODERATE or higher.
Use a calm, caring, and direct approach.
"""
            
            # Generate crisis-appropriate response
            response_text = await self.generate_response(
                enhanced_prompt,
                conversation_history
            )
            
            # Add crisis resources for high-risk situations
            if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                resources = self.get_crisis_resources()
                response_text = f"{response_text}\n\n{resources}"
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_text,
                confidence=assessment.confidence,
                suggested_actions=assessment.immediate_actions,
                escalate=assessment.requires_escalation,
                metadata={
                    "risk_level": assessment.risk_level.name,
                    "crisis_types": [c.value for c in assessment.crisis_types],
                    "risk_factors": assessment.risk_factors,
                    "protective_factors": assessment.protective_factors,
                    "escalation_reason": assessment.escalation_reason
                },
                processing_time_ms=processing_time
            )

    def _get_response_guidance(self, assessment: RiskAssessment) -> str:
        """Get response guidance based on risk assessment."""
        if assessment.risk_level == RiskLevel.CRITICAL:
            return """
CRITICAL RESPONSE REQUIRED:
1. Express immediate concern for their safety
2. Ask directly about suicidal thoughts/plans
3. Emphasize you want to help keep them safe
4. Provide crisis hotline numbers prominently
5. Encourage calling emergency services if immediate danger
6. Stay with them in conversation if possible
"""
        elif assessment.risk_level == RiskLevel.HIGH:
            return """
HIGH RISK RESPONSE:
1. Acknowledge their pain with deep empathy
2. Express concern for their wellbeing
3. Gently assess their immediate safety
4. Provide crisis resources
5. Explore what support they have available
6. Work toward a safety commitment
"""
        elif assessment.risk_level == RiskLevel.MODERATE:
            return """
MODERATE RISK RESPONSE:
1. Validate their experience
2. Express care for their wellbeing
3. Gently explore what they're going through
4. Mention that support is available if needed
5. Continue to monitor for escalation
"""
        else:
            return """
STANDARD SUPPORTIVE RESPONSE:
1. Respond with empathy and warmth
2. Continue to be present with them
3. Monitor for any changes in risk level
"""

    async def safety_check(self, user_input: str) -> Dict[str, Any]:
        """
        Quick safety check for use by other agents.
        
        Args:
            user_input: Message to check
            
        Returns:
            Safety check result
        """
        risk_score, risk_factors, protective_factors = self.calculate_risk_score(user_input)
        risk_level = self.determine_risk_level(risk_score)
        
        return {
            "is_safe": risk_level in [RiskLevel.NONE, RiskLevel.LOW],
            "risk_level": risk_level.name,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "requires_intervention": risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return agent metrics for observability."""
        return {
            "agent_type": self.agent_type.value,
            "assessments_performed": self._assessments_performed,
            "escalations_triggered": self._escalations_triggered,
            "risk_levels_detected": self._risk_levels_detected,
            "escalation_rate": (
                self._escalations_triggered / self._assessments_performed
                if self._assessments_performed > 0 else 0
            )
        }

