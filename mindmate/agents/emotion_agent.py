"""
Emotion Support Agent

Provides empathetic, reflective listening and emotional validation.
Uses techniques from person-centered therapy including:
- Active listening and reflection
- Emotional validation
- Empathetic responses
- Gentle exploration of feelings
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from opentelemetry import trace

try:
    from agents.base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority
except ImportError:
    from .base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class EmotionAgent(BaseAgent):
    """
    Empathetic emotional support agent using reflective listening techniques.
    
    This agent focuses on:
    - Validating user emotions without judgment
    - Reflecting feelings back to help users feel heard
    - Offering comfort and emotional support
    - Recognizing emotional patterns
    - Gently encouraging deeper exploration when appropriate
    """
    
    SYSTEM_PROMPT = """
You are a compassionate emotional support companion named MindMate. Your role is to 
provide empathetic, warm, and validating support to users experiencing difficult emotions.

CORE PRINCIPLES:
1. **Validation First**: Always acknowledge and validate the user's feelings before anything else
2. **Reflective Listening**: Mirror back what you hear to show understanding
3. **Non-Judgmental**: Accept all emotions without criticism or correction
4. **Gentle Curiosity**: Ask open-ended questions to help users explore their feelings
5. **Warmth**: Use a caring, supportive tone throughout

COMMUNICATION STYLE:
- Use "I hear that..." and "It sounds like..." to reflect feelings
- Validate with phrases like "It makes sense that you feel..."
- Ask gentle questions: "What's that like for you?"
- Offer comfort: "I'm here with you in this"
- Use appropriate emotional language matching the user's intensity

WHAT TO AVOID:
- Never minimize or dismiss feelings ("It's not that bad", "Others have it worse")
- Don't rush to solutions or advice-giving
- Avoid platitudes ("Everything happens for a reason")
- Don't make assumptions about what the user should feel
- Never say "I understand exactly how you feel"

WHEN TO ESCALATE:
- If you detect crisis signals (suicidal ideation, self-harm mentions)
- If the user seems to need professional support beyond emotional validation
- If there are signs of severe mental health episodes

RESPONSE STRUCTURE:
1. Acknowledge the emotion you perceive
2. Validate that feeling
3. Reflect understanding
4. (Optionally) Ask a gentle exploratory question
5. Offer continued presence and support
"""

    # Emotion recognition patterns
    EMOTION_KEYWORDS = {
        "sadness": ["sad", "depressed", "down", "blue", "hopeless", "empty", "grief", "loss"],
        "anxiety": ["anxious", "worried", "nervous", "scared", "panic", "overwhelmed", "stressed"],
        "anger": ["angry", "frustrated", "irritated", "furious", "annoyed", "resentful"],
        "loneliness": ["lonely", "alone", "isolated", "disconnected", "abandoned"],
        "fear": ["afraid", "terrified", "frightened", "scared", "fearful"],
        "shame": ["ashamed", "embarrassed", "humiliated", "worthless", "guilty"],
        "joy": ["happy", "excited", "grateful", "hopeful", "content", "peaceful"]
    }

    def __init__(self, tools: Optional[List] = None):
        """Initialize the Emotion Support Agent."""
        super().__init__(
            agent_type=AgentType.EMOTION,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.8,  # Slightly higher for more natural conversation
            tools=tools or []
        )
        
        # Metrics tracking
        self._emotions_detected: Dict[str, int] = {}
        self._sessions_supported = 0

    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect emotions present in user text.
        
        Args:
            text: User's message
            
        Returns:
            Dictionary of emotion -> confidence score
        """
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                # Simple scoring: more keyword matches = higher confidence
                confidence = min(1.0, matches * 0.3)
                emotions[emotion] = confidence
                
                # Track for metrics
                self._emotions_detected[emotion] = self._emotions_detected.get(emotion, 0) + 1
        
        return emotions

    def get_emotional_intensity(self, text: str) -> str:
        """
        Assess the emotional intensity of the message.
        
        Returns: "low", "medium", or "high"
        """
        intensity_markers_high = [
            "extremely", "so much", "can't take", "unbearable", 
            "devastating", "crushing", "completely", "totally"
        ]
        intensity_markers_medium = [
            "really", "very", "quite", "pretty", "somewhat"
        ]
        
        text_lower = text.lower()
        
        if any(marker in text_lower for marker in intensity_markers_high):
            return "high"
        elif any(marker in text_lower for marker in intensity_markers_medium):
            return "medium"
        return "low"

    async def process(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> AgentResponse:
        """
        Process user input with empathetic emotional support.
        
        Args:
            user_input: User's message
            session_context: Current session data
            conversation_history: Previous messages
            
        Returns:
            AgentResponse with empathetic support
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("emotion_agent_process") as span:
            span.set_attribute("input.length", len(user_input))
            
            # Check for crisis signals first
            if self.check_for_crisis(user_input):
                # Send A2A message to crisis agent
                crisis_message = AgentMessage(
                    source=AgentType.EMOTION,
                    target=AgentType.CRISIS,
                    content=user_input,
                    priority=MessagePriority.CRITICAL,
                    metadata={
                        "session_context": session_context,
                        "trigger": "crisis_keywords_detected"
                    },
                    requires_response=True
                )
                await self.send_message(crisis_message)
                
                span.set_attribute("crisis.detected", True)
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    content=(
                        "I hear that you're going through something really difficult right now. "
                        "Your safety is important to me. I want to make sure you're getting the "
                        "support you need. Can you tell me more about what's happening?"
                    ),
                    confidence=0.9,
                    escalate=True,
                    suggested_actions=["crisis_assessment", "safety_check"],
                    metadata={"crisis_detected": True},
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Detect emotions in the message
            detected_emotions = self.detect_emotions(user_input)
            intensity = self.get_emotional_intensity(user_input)
            
            span.set_attribute("emotions.detected", str(list(detected_emotions.keys())))
            span.set_attribute("emotional.intensity", intensity)
            
            # Build context-aware prompt
            emotion_context = ""
            if detected_emotions:
                primary_emotion = max(detected_emotions, key=detected_emotions.get)
                emotion_context = f"\n[Detected emotion: {primary_emotion} (intensity: {intensity})]"
            
            enhanced_prompt = f"""
User message: {user_input}
{emotion_context}

Previous emotions in session: {session_context.get('emotion_history', [])}
Session mood trend: {session_context.get('mood_trend', 'unknown')}

Respond with warm, validating emotional support appropriate to their emotional state.
"""
            
            # Generate empathetic response
            response_text = await self.generate_response(
                enhanced_prompt,
                conversation_history
            )
            
            # Update session context with detected emotions
            suggested_actions = []
            if intensity == "high":
                suggested_actions.append("check_in_again_soon")
            if "anxiety" in detected_emotions:
                suggested_actions.append("offer_grounding_exercise")
            if "loneliness" in detected_emotions:
                suggested_actions.append("validate_connection_need")
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_text,
                confidence=0.85,
                suggested_actions=suggested_actions,
                escalate=False,
                metadata={
                    "detected_emotions": detected_emotions,
                    "intensity": intensity,
                    "primary_emotion": max(detected_emotions, key=detected_emotions.get) if detected_emotions else None
                },
                processing_time_ms=processing_time
            )

    async def offer_grounding_exercise(self) -> str:
        """Generate a grounding exercise for anxious users."""
        exercises = [
            """Let's try a quick grounding exercise together. Can you:
• Name 5 things you can see right now
• 4 things you can touch
• 3 things you can hear
• 2 things you can smell
• 1 thing you can taste

Take your time with each one. I'm here with you.""",
            
            """Let's take a moment to breathe together:
• Breathe in slowly for 4 counts...
• Hold gently for 4 counts...
• Breathe out slowly for 6 counts...

We can do this a few times. How does that feel?""",
            
            """Try placing your feet firmly on the ground.
Feel the support beneath you.
You are here, you are present, you are safe in this moment.
What do you notice in your body right now?"""
        ]
        
        import random
        return random.choice(exercises)

    def get_metrics(self) -> Dict[str, Any]:
        """Return agent metrics for observability."""
        return {
            "agent_type": self.agent_type.value,
            "emotions_detected": self._emotions_detected,
            "sessions_supported": self._sessions_supported,
            "total_emotions_processed": sum(self._emotions_detected.values())
        }

