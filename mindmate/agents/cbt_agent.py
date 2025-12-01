"""
CBT Restructuring Agent

Implements Cognitive-Behavioral Therapy techniques for thought analysis
and cognitive restructuring. Uses Socratic questioning to help users
identify and challenge unhelpful thinking patterns.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from opentelemetry import trace

try:
    from agents.base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority
except ImportError:
    from .base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class CognitiveDistortion(Enum):
    """Common cognitive distortions identified in CBT."""
    ALL_OR_NOTHING = "all_or_nothing"  # Black and white thinking
    CATASTROPHIZING = "catastrophizing"  # Assuming the worst
    MIND_READING = "mind_reading"  # Assuming others' thoughts
    FORTUNE_TELLING = "fortune_telling"  # Predicting negative outcomes
    OVERGENERALIZATION = "overgeneralization"  # One event = pattern
    MENTAL_FILTER = "mental_filter"  # Focus only on negatives
    DISQUALIFYING = "disqualifying_positive"  # Dismissing positives
    SHOULD_STATEMENTS = "should_statements"  # Rigid rules
    LABELING = "labeling"  # Global negative labels
    PERSONALIZATION = "personalization"  # Taking blame inappropriately
    EMOTIONAL_REASONING = "emotional_reasoning"  # Feelings = facts


@dataclass
class ThoughtRecord:
    """Structure for CBT thought analysis."""
    situation: str
    automatic_thought: str
    emotion: str
    emotion_intensity: int  # 0-100
    distortions: List[CognitiveDistortion]
    evidence_for: List[str]
    evidence_against: List[str]
    balanced_thought: Optional[str] = None
    new_emotion_intensity: Optional[int] = None


class CBTAgent(BaseAgent):
    """
    Cognitive-Behavioral Therapy agent using Socratic questioning.
    
    Helps users:
    - Identify automatic negative thoughts
    - Recognize cognitive distortions
    - Challenge unhelpful thinking patterns
    - Develop more balanced perspectives
    - Build cognitive flexibility
    """
    
    SYSTEM_PROMPT = """
You are a supportive CBT (Cognitive-Behavioral Therapy) companion. Your role is to 
gently help users examine their thoughts and develop more balanced perspectives using
Socratic questioning techniques.

CORE CBT PRINCIPLES:
1. Thoughts influence emotions and behaviors
2. We can learn to identify and modify unhelpful thought patterns
3. Change is possible through awareness and practice
4. The goal is balanced thinking, not positive thinking

SOCRATIC QUESTIONING APPROACH:
- Ask open-ended questions that promote reflection
- Guide discovery rather than providing answers
- Be curious and collaborative, not directive
- Help users examine evidence for their thoughts
- Encourage consideration of alternative perspectives

KEY QUESTION TYPES:
1. Clarifying: "What do you mean when you say...?"
2. Evidence: "What evidence supports that thought?"
3. Alternative: "Is there another way to look at this?"
4. Consequences: "What's the effect of thinking this way?"
5. Reality testing: "Has this always been true in the past?"

COGNITIVE DISTORTIONS TO WATCH FOR:
- All-or-nothing thinking (black and white)
- Catastrophizing (worst-case assumptions)
- Mind reading (assuming others' thoughts)
- Overgeneralization (one event = always)
- Should statements (rigid rules)
- Emotional reasoning (I feel it, so it's true)

IMPORTANT GUIDELINES:
- Never force a perspective change
- Validate the emotion before examining the thought
- Move at the user's pace
- Celebrate small insights
- Don't diagnose or label the user
- Frame distortions gently, not critically

RESPONSE STRUCTURE:
1. Acknowledge the user's experience
2. Gently identify a thought to examine (if present)
3. Ask ONE Socratic question to explore further
4. Offer support regardless of their response
"""

    # Patterns that may indicate cognitive distortions
    DISTORTION_PATTERNS = {
        CognitiveDistortion.ALL_OR_NOTHING: [
            "always", "never", "everyone", "no one", "completely", 
            "totally", "nothing", "everything", "ruined", "perfect"
        ],
        CognitiveDistortion.CATASTROPHIZING: [
            "disaster", "terrible", "awful", "worst", "can't handle",
            "end of the world", "unbearable", "impossible"
        ],
        CognitiveDistortion.MIND_READING: [
            "they think", "he thinks", "she thinks", "they must think",
            "everyone thinks", "they hate", "they don't like"
        ],
        CognitiveDistortion.SHOULD_STATEMENTS: [
            "should", "shouldn't", "must", "mustn't", "have to",
            "ought to", "supposed to"
        ],
        CognitiveDistortion.LABELING: [
            "i'm a failure", "i'm stupid", "i'm worthless", "i'm an idiot",
            "i'm a loser", "i'm pathetic", "i'm weak"
        ],
        CognitiveDistortion.EMOTIONAL_REASONING: [
            "i feel like", "i feel that", "feels like i'm",
            "it feels true", "i just know"
        ]
    }

    def __init__(self, tools: Optional[List] = None):
        """Initialize the CBT Agent."""
        super().__init__(
            agent_type=AgentType.CBT,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            tools=tools or []
        )
        
        # Track distortions identified
        self._distortions_identified: Dict[str, int] = {}
        self._thought_records_created = 0

    def detect_distortions(self, text: str) -> List[CognitiveDistortion]:
        """
        Detect potential cognitive distortions in text.
        
        Args:
            text: User's message
            
        Returns:
            List of detected cognitive distortions
        """
        text_lower = text.lower()
        detected = []
        
        for distortion, patterns in self.DISTORTION_PATTERNS.items():
            if any(pattern in text_lower for pattern in patterns):
                detected.append(distortion)
                # Track for metrics
                name = distortion.value
                self._distortions_identified[name] = self._distortions_identified.get(name, 0) + 1
        
        return detected

    def get_distortion_explanation(self, distortion: CognitiveDistortion) -> str:
        """Get a gentle explanation of a cognitive distortion."""
        explanations = {
            CognitiveDistortion.ALL_OR_NOTHING: 
                "Sometimes our minds see things in black and white, when reality often "
                "has many shades of gray.",
            CognitiveDistortion.CATASTROPHIZING:
                "Our minds sometimes jump to the worst possible outcome, even when "
                "other outcomes are just as likely.",
            CognitiveDistortion.MIND_READING:
                "We sometimes assume we know what others are thinking, when we can't "
                "actually know for certain.",
            CognitiveDistortion.SHOULD_STATEMENTS:
                "We often have rules about how things 'should' be, which can create "
                "pressure and frustration.",
            CognitiveDistortion.LABELING:
                "We sometimes put global labels on ourselves based on specific situations.",
            CognitiveDistortion.EMOTIONAL_REASONING:
                "Our emotions can feel like facts, but feelings and reality don't "
                "always match up.",
            CognitiveDistortion.OVERGENERALIZATION:
                "One experience can sometimes feel like it represents everything.",
            CognitiveDistortion.FORTUNE_TELLING:
                "We sometimes predict the future negatively without full evidence.",
            CognitiveDistortion.MENTAL_FILTER:
                "We might focus on negative details while filtering out positive ones.",
            CognitiveDistortion.DISQUALIFYING:
                "Sometimes we dismiss positive experiences as not counting.",
            CognitiveDistortion.PERSONALIZATION:
                "We might take responsibility for things outside our control."
        }
        return explanations.get(distortion, "This is a common thinking pattern.")

    def generate_socratic_questions(
        self,
        distortions: List[CognitiveDistortion],
        context: str
    ) -> List[str]:
        """
        Generate appropriate Socratic questions based on detected distortions.
        
        Args:
            distortions: Detected cognitive distortions
            context: User's message for context
            
        Returns:
            List of relevant Socratic questions
        """
        questions = []
        
        for distortion in distortions[:2]:  # Limit to top 2 to not overwhelm
            if distortion == CognitiveDistortion.ALL_OR_NOTHING:
                questions.append("Are there any exceptions to this that you can think of?")
                questions.append("What would it look like if this were partially true instead?")
            
            elif distortion == CognitiveDistortion.CATASTROPHIZING:
                questions.append("What's the most likely outcome, realistically?")
                questions.append("If this did happen, how might you cope?")
            
            elif distortion == CognitiveDistortion.MIND_READING:
                questions.append("What evidence do you have for what they're thinking?")
                questions.append("Could there be other explanations for their behavior?")
            
            elif distortion == CognitiveDistortion.SHOULD_STATEMENTS:
                questions.append("Where does this 'should' come from?")
                questions.append("What happens when you replace 'should' with 'could' or 'would like to'?")
            
            elif distortion == CognitiveDistortion.LABELING:
                questions.append("If a friend described themselves this way, what would you tell them?")
                questions.append("Does this label capture all of who you are?")
            
            elif distortion == CognitiveDistortion.EMOTIONAL_REASONING:
                questions.append("Is it possible to feel something strongly and have it not be the full picture?")
                questions.append("What would the situation look like from an outside observer's view?")
        
        return questions

    async def process(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> AgentResponse:
        """
        Process user input with CBT-informed analysis and questioning.
        
        Args:
            user_input: User's message
            session_context: Current session data
            conversation_history: Previous messages
            
        Returns:
            AgentResponse with CBT intervention
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("cbt_agent_process") as span:
            span.set_attribute("input.length", len(user_input))
            
            # Check for crisis signals first
            if self.check_for_crisis(user_input):
                crisis_message = AgentMessage(
                    source=AgentType.CBT,
                    target=AgentType.CRISIS,
                    content=user_input,
                    priority=MessagePriority.CRITICAL,
                    metadata={"session_context": session_context}
                )
                await self.send_message(crisis_message)
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    content=(
                        "I notice you're going through something really difficult. "
                        "Right now, let's focus on keeping you safe. "
                        "Can you tell me more about how you're feeling in this moment?"
                    ),
                    confidence=0.9,
                    escalate=True,
                    metadata={"crisis_detected": True},
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Detect cognitive distortions
            distortions = self.detect_distortions(user_input)
            socratic_questions = self.generate_socratic_questions(distortions, user_input)
            
            span.set_attribute("distortions.count", len(distortions))
            span.set_attribute("distortions.types", str([d.value for d in distortions]))
            
            # Build context-aware prompt
            distortion_context = ""
            if distortions:
                distortion_names = [d.value.replace("_", " ") for d in distortions]
                distortion_context = f"\n[Potential thought patterns: {', '.join(distortion_names)}]"
                distortion_context += f"\n[Suggested questions: {socratic_questions[:2]}]"
            
            # Check session for ongoing thought record
            thought_record = session_context.get("current_thought_record")
            thought_context = ""
            if thought_record:
                thought_context = f"\n[Ongoing thought record - Stage: {thought_record.get('stage', 'initial')}]"
            
            enhanced_prompt = f"""
User message: {user_input}
{distortion_context}
{thought_context}

Previous CBT work in session: {session_context.get('cbt_insights', [])}

Using Socratic questioning and CBT principles:
1. First acknowledge their experience with empathy
2. If thought patterns are detected, gently explore one
3. Ask ONE thoughtful Socratic question
4. Keep the tone warm and collaborative
"""
            
            # Generate CBT-informed response
            response_text = await self.generate_response(
                enhanced_prompt,
                conversation_history
            )
            
            # Determine suggested actions
            suggested_actions = []
            if distortions:
                suggested_actions.append("continue_thought_examination")
            if len(conversation_history) > 6:
                suggested_actions.append("summarize_insights")
            if "success" in user_input.lower() or "better" in user_input.lower():
                suggested_actions.append("reinforce_progress")
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_text,
                confidence=0.85,
                suggested_actions=suggested_actions,
                escalate=False,
                metadata={
                    "distortions_detected": [d.value for d in distortions],
                    "socratic_questions": socratic_questions,
                    "thought_record_active": thought_record is not None
                },
                processing_time_ms=processing_time
            )

    async def create_thought_record(
        self,
        situation: str,
        automatic_thought: str,
        emotion: str,
        intensity: int
    ) -> ThoughtRecord:
        """
        Create a new thought record for structured CBT work.
        
        Args:
            situation: What happened
            automatic_thought: The automatic thought that arose
            emotion: Primary emotion felt
            intensity: Emotion intensity (0-100)
            
        Returns:
            Initialized ThoughtRecord
        """
        distortions = self.detect_distortions(automatic_thought)
        
        record = ThoughtRecord(
            situation=situation,
            automatic_thought=automatic_thought,
            emotion=emotion,
            emotion_intensity=intensity,
            distortions=distortions,
            evidence_for=[],
            evidence_against=[]
        )
        
        self._thought_records_created += 1
        logger.info(f"Created thought record for situation: {situation[:50]}...")
        
        return record

    async def complete_thought_record(
        self,
        record: ThoughtRecord,
        evidence_for: List[str],
        evidence_against: List[str],
        balanced_thought: str,
        new_intensity: int
    ) -> ThoughtRecord:
        """
        Complete a thought record with evidence and balanced perspective.
        
        Args:
            record: The thought record to complete
            evidence_for: Evidence supporting the automatic thought
            evidence_against: Evidence against the automatic thought  
            balanced_thought: More balanced alternative thought
            new_intensity: New emotion intensity after reframing
            
        Returns:
            Completed ThoughtRecord
        """
        record.evidence_for = evidence_for
        record.evidence_against = evidence_against
        record.balanced_thought = balanced_thought
        record.new_emotion_intensity = new_intensity
        
        return record

    def get_metrics(self) -> Dict[str, Any]:
        """Return agent metrics for observability."""
        return {
            "agent_type": self.agent_type.value,
            "distortions_identified": self._distortions_identified,
            "thought_records_created": self._thought_records_created,
            "total_distortions": sum(self._distortions_identified.values())
        }

