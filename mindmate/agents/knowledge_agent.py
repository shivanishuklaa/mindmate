"""
Knowledge Therapy Agent

Provides psychoeducation and evidence-based mental health information.
Uses search capabilities to find relevant resources and summarizes
complex psychological concepts in accessible language.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from opentelemetry import trace

try:
    from agents.base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority
except ImportError:
    from .base_agent import BaseAgent, AgentType, AgentResponse, AgentMessage, MessagePriority

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class TopicCategory(Enum):
    """Categories of mental health topics."""
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    STRESS = "stress"
    RELATIONSHIPS = "relationships"
    SELF_ESTEEM = "self_esteem"
    TRAUMA = "trauma"
    GRIEF = "grief"
    SLEEP = "sleep"
    MINDFULNESS = "mindfulness"
    COPING_SKILLS = "coping_skills"
    THERAPY_TYPES = "therapy_types"
    MEDICATIONS = "medications"
    GENERAL = "general"


@dataclass
class PsychoeducationResource:
    """Structure for educational resources."""
    topic: TopicCategory
    title: str
    summary: str
    key_points: List[str]
    techniques: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    difficulty_level: str = "accessible"  # accessible, intermediate, advanced


class KnowledgeAgent(BaseAgent):
    """
    Psychoeducation and knowledge agent.
    
    Provides:
    - Evidence-based mental health information
    - Explanations of psychological concepts
    - Coping techniques and strategies
    - Therapy type explanations
    - Research-backed insights
    """
    
    SYSTEM_PROMPT = """
You are a mental health educator within MindMate. Your role is to provide clear,
accurate, and accessible information about mental health topics, coping strategies,
and psychological concepts.

CORE PRINCIPLES:
1. ACCURACY: Only share evidence-based information
2. ACCESSIBILITY: Explain complex concepts simply
3. EMPOWERMENT: Help users understand themselves better
4. BOUNDARIES: Educational info, not diagnosis or treatment
5. SOURCES: Reference reputable sources when possible

WHAT YOU CAN PROVIDE:
- Explanations of mental health conditions (general info, not diagnosis)
- Description of therapy types (CBT, DBT, ACT, etc.)
- Coping techniques and strategies
- Mindfulness and relaxation exercises
- Information about how the brain processes emotions
- Research-backed wellness practices
- Understanding of common experiences (grief, stress, anxiety)

WHAT YOU CANNOT PROVIDE:
- Medical diagnoses
- Medication recommendations
- Specific treatment plans
- Predictions about prognosis
- Advice that should come from a doctor

COMMUNICATION STYLE:
- Use clear, jargon-free language
- Provide examples to illustrate concepts
- Break down complex ideas into steps
- Validate that learning about mental health is valuable
- Encourage professional support when appropriate

RESPONSE STRUCTURE:
1. Address the topic clearly
2. Provide key educational points
3. Offer practical takeaways when relevant
4. Suggest related topics if appropriate
5. Remind that professional support is valuable
"""

    # Built-in knowledge base for common topics
    KNOWLEDGE_BASE: Dict[TopicCategory, PsychoeducationResource] = {
        TopicCategory.ANXIETY: PsychoeducationResource(
            topic=TopicCategory.ANXIETY,
            title="Understanding Anxiety",
            summary="Anxiety is a natural response to stress, involving feelings of worry, nervousness, or unease about uncertain outcomes.",
            key_points=[
                "Anxiety is a normal part of life and serves a protective function",
                "It becomes a concern when it's persistent, excessive, or interferes with daily life",
                "Physical symptoms can include rapid heartbeat, sweating, and muscle tension",
                "Cognitive symptoms include racing thoughts, worry, and difficulty concentrating",
                "Anxiety disorders are highly treatable with therapy and/or medication"
            ],
            techniques=[
                "Deep breathing exercises",
                "Progressive muscle relaxation",
                "Grounding techniques (5-4-3-2-1)",
                "Cognitive restructuring",
                "Gradual exposure to fears"
            ],
            sources=["American Psychological Association", "National Institute of Mental Health"]
        ),
        TopicCategory.DEPRESSION: PsychoeducationResource(
            topic=TopicCategory.DEPRESSION,
            title="Understanding Depression",
            summary="Depression is more than feeling sad—it's a persistent condition affecting mood, thoughts, behavior, and physical health.",
            key_points=[
                "Depression is a medical condition, not a character flaw or weakness",
                "Symptoms include persistent sadness, loss of interest, changes in sleep/appetite",
                "It can affect anyone regardless of circumstances",
                "Treatment is effective for most people",
                "Recovery is possible with proper support"
            ],
            techniques=[
                "Behavioral activation (scheduling enjoyable activities)",
                "Cognitive restructuring",
                "Social connection",
                "Physical exercise",
                "Sleep hygiene"
            ],
            sources=["World Health Organization", "Mayo Clinic"]
        ),
        TopicCategory.MINDFULNESS: PsychoeducationResource(
            topic=TopicCategory.MINDFULNESS,
            title="Introduction to Mindfulness",
            summary="Mindfulness is the practice of paying attention to the present moment without judgment.",
            key_points=[
                "Mindfulness is about awareness, not clearing your mind",
                "It can reduce stress, anxiety, and depression symptoms",
                "Regular practice creates lasting changes in the brain",
                "It can be practiced formally (meditation) or informally (daily activities)",
                "There's no 'right' way to be mindful"
            ],
            techniques=[
                "Body scan meditation",
                "Mindful breathing",
                "Loving-kindness meditation",
                "Mindful walking",
                "Mindful eating"
            ],
            sources=["Mindfulness-Based Stress Reduction (MBSR)", "Jon Kabat-Zinn research"]
        ),
        TopicCategory.COPING_SKILLS: PsychoeducationResource(
            topic=TopicCategory.COPING_SKILLS,
            title="Healthy Coping Skills",
            summary="Coping skills are strategies we use to deal with stress, difficult emotions, and challenging situations.",
            key_points=[
                "Healthy coping helps manage stress without causing additional problems",
                "Different situations may call for different coping strategies",
                "Coping skills can be learned and strengthened with practice",
                "Both emotion-focused and problem-focused coping have value",
                "Having a variety of coping tools increases resilience"
            ],
            techniques=[
                "Deep breathing",
                "Journaling",
                "Physical exercise",
                "Talking to supportive people",
                "Creative expression",
                "Time in nature",
                "Progressive muscle relaxation",
                "Setting boundaries"
            ],
            sources=["American Psychological Association", "Dialectical Behavior Therapy skills"]
        ),
        TopicCategory.GRIEF: PsychoeducationResource(
            topic=TopicCategory.GRIEF,
            title="Understanding Grief",
            summary="Grief is a natural response to loss, encompassing a range of emotions and experiences.",
            key_points=[
                "Grief is not linear—there's no 'right' way or timeline",
                "It can be triggered by many types of loss (death, relationships, identity)",
                "Physical symptoms are common (fatigue, appetite changes)",
                "Grief can come in waves, even years later",
                "Support and connection are important during grief"
            ],
            techniques=[
                "Allow yourself to feel without judgment",
                "Create rituals to honor what was lost",
                "Connect with others who understand",
                "Take care of basic needs (sleep, food, movement)",
                "Consider grief counseling if struggling"
            ],
            sources=["Kübler-Ross model", "Worden's Tasks of Mourning"]
        ),
        TopicCategory.STRESS: PsychoeducationResource(
            topic=TopicCategory.STRESS,
            title="Understanding Stress",
            summary="Stress is the body's response to demands or threats, activating the fight-or-flight response.",
            key_points=[
                "Some stress is normal and can be motivating",
                "Chronic stress can harm physical and mental health",
                "The stress response involves hormones like cortisol and adrenaline",
                "Perception plays a key role in how we experience stress",
                "Stress management skills can be learned"
            ],
            techniques=[
                "Regular physical activity",
                "Adequate sleep",
                "Time management",
                "Setting realistic expectations",
                "Social support",
                "Relaxation techniques",
                "Healthy boundaries"
            ],
            sources=["American Institute of Stress", "Hans Selye stress research"]
        ),
        TopicCategory.THERAPY_TYPES: PsychoeducationResource(
            topic=TopicCategory.THERAPY_TYPES,
            title="Types of Therapy",
            summary="Various therapeutic approaches can help with mental health, each with different focuses and techniques.",
            key_points=[
                "CBT (Cognitive-Behavioral Therapy): Focuses on thoughts, feelings, behaviors",
                "DBT (Dialectical Behavior Therapy): Combines CBT with mindfulness and acceptance",
                "ACT (Acceptance and Commitment Therapy): Emphasizes acceptance and values",
                "Psychodynamic: Explores unconscious patterns from past experiences",
                "Humanistic: Person-centered, focuses on self-actualization",
                "EMDR: Processes traumatic memories through eye movements"
            ],
            techniques=[
                "Finding the right fit may take time",
                "Different approaches work better for different issues",
                "The therapeutic relationship is key to effectiveness"
            ],
            sources=["American Psychological Association", "National Alliance on Mental Illness"]
        )
    }

    # Topic detection keywords
    TOPIC_KEYWORDS: Dict[TopicCategory, List[str]] = {
        TopicCategory.ANXIETY: ["anxiety", "anxious", "worried", "panic", "nervous", "fear"],
        TopicCategory.DEPRESSION: ["depression", "depressed", "hopeless", "empty", "sad"],
        TopicCategory.STRESS: ["stress", "stressed", "overwhelmed", "pressure", "burnout"],
        TopicCategory.RELATIONSHIPS: ["relationship", "partner", "friend", "family", "conflict"],
        TopicCategory.SELF_ESTEEM: ["self-esteem", "confidence", "worthless", "self-worth"],
        TopicCategory.TRAUMA: ["trauma", "ptsd", "flashback", "triggered", "abuse"],
        TopicCategory.GRIEF: ["grief", "loss", "death", "mourning", "bereaved"],
        TopicCategory.SLEEP: ["sleep", "insomnia", "tired", "exhausted", "nightmare"],
        TopicCategory.MINDFULNESS: ["mindfulness", "meditation", "present", "aware"],
        TopicCategory.COPING_SKILLS: ["cope", "coping", "deal with", "manage", "handle"],
        TopicCategory.THERAPY_TYPES: ["therapy", "therapist", "counseling", "treatment", "cbt", "dbt"]
    }

    def __init__(self, tools: Optional[List] = None):
        """Initialize the Knowledge Agent."""
        super().__init__(
            agent_type=AgentType.KNOWLEDGE,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.6,  # Balanced for accuracy and readability
            tools=tools or []
        )
        
        # Metrics
        self._topics_requested: Dict[str, int] = {}
        self._searches_performed = 0

    def detect_topic(self, text: str) -> Optional[TopicCategory]:
        """
        Detect the primary topic in user's message.
        
        Args:
            text: User's message
            
        Returns:
            Detected topic category or None
        """
        text_lower = text.lower()
        topic_scores: Dict[TopicCategory, int] = {}
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            primary_topic = max(topic_scores, key=topic_scores.get)
            self._topics_requested[primary_topic.value] = \
                self._topics_requested.get(primary_topic.value, 0) + 1
            return primary_topic
        
        return None

    def get_knowledge_resource(self, topic: TopicCategory) -> Optional[PsychoeducationResource]:
        """Get built-in knowledge resource for a topic."""
        return self.KNOWLEDGE_BASE.get(topic)

    async def search_external_resources(
        self,
        query: str,
        topic: Optional[TopicCategory] = None
    ) -> List[Dict[str, str]]:
        """
        Search for external resources (placeholder for Google Search integration).
        
        Args:
            query: Search query
            topic: Optional topic for filtering
            
        Returns:
            List of search results
        """
        self._searches_performed += 1
        
        # In production, this would use Google Search API
        # For now, return placeholder that indicates search would occur
        logger.info(f"Would search for: {query}")
        
        return [
            {
                "title": f"Evidence-based information about {topic.value if topic else 'mental health'}",
                "snippet": "This would contain relevant search results from reputable sources.",
                "source": "Google Search API"
            }
        ]

    def format_resource_response(self, resource: PsychoeducationResource) -> str:
        """Format a knowledge resource into a readable response."""
        lines = [
            f"## {resource.title}\n",
            f"{resource.summary}\n",
            "\n**Key Points:**"
        ]
        
        for point in resource.key_points:
            lines.append(f"• {point}")
        
        if resource.techniques:
            lines.append("\n**Helpful Techniques:**")
            for technique in resource.techniques[:5]:  # Limit to 5
                lines.append(f"• {technique}")
        
        if resource.sources:
            lines.append(f"\n*Sources: {', '.join(resource.sources)}*")
        
        return "\n".join(lines)

    async def process(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> AgentResponse:
        """
        Process user input with psychoeducation and information.
        
        Args:
            user_input: User's message
            session_context: Current session data
            conversation_history: Previous messages
            
        Returns:
            AgentResponse with educational content
        """
        start_time = time.time()
        
        with tracer.start_as_current_span("knowledge_agent_process") as span:
            span.set_attribute("input.length", len(user_input))
            
            # Check for crisis signals first (all agents should do this)
            if self.check_for_crisis(user_input):
                crisis_message = AgentMessage(
                    source=AgentType.KNOWLEDGE,
                    target=AgentType.CRISIS,
                    content=user_input,
                    priority=MessagePriority.CRITICAL,
                    metadata={"session_context": session_context}
                )
                await self.send_message(crisis_message)
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    content=(
                        "I hear that you're going through something really difficult. "
                        "While I can share information about mental health, right now "
                        "I want to make sure you're getting the support you need. "
                        "Can you tell me more about how you're feeling?"
                    ),
                    confidence=0.9,
                    escalate=True,
                    metadata={"crisis_detected": True},
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Detect topic of interest
            topic = self.detect_topic(user_input)
            span.set_attribute("topic.detected", topic.value if topic else "none")
            
            # Check for built-in resource
            resource = self.get_knowledge_resource(topic) if topic else None
            resource_context = ""
            
            if resource:
                resource_context = f"""
RELEVANT KNOWLEDGE RESOURCE:
Topic: {resource.title}
Summary: {resource.summary}
Key Points: {resource.key_points}
Techniques: {resource.techniques}
"""
            
            # Build context-aware prompt
            enhanced_prompt = f"""
User question: {user_input}

Topic detected: {topic.value if topic else 'general mental health'}
{resource_context}

Previous topics discussed: {session_context.get('topics_covered', [])}

Provide educational, evidence-based information about the topic.
Be clear, accessible, and empowering.
Include practical takeaways when relevant.
Remind that professional support is valuable for personal concerns.
"""
            
            # Generate educational response
            response_text = await self.generate_response(
                enhanced_prompt,
                conversation_history
            )
            
            # Add formatted resource if available and relevant
            tools_used = []
            if resource and len(response_text) < 500:
                # Add structured resource for shorter responses
                formatted_resource = self.format_resource_response(resource)
                response_text = f"{response_text}\n\n---\n\n{formatted_resource}"
            
            # Determine suggested follow-up topics
            suggested_actions = []
            if topic:
                related_topics = self._get_related_topics(topic)
                if related_topics:
                    suggested_actions.append(f"explore_related: {related_topics}")
            
            suggested_actions.append("offer_practical_exercise")
            
            processing_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                content=response_text,
                confidence=0.85,
                suggested_actions=suggested_actions,
                escalate=False,
                metadata={
                    "topic": topic.value if topic else None,
                    "resource_used": resource is not None,
                    "sources": resource.sources if resource else []
                },
                tools_used=tools_used,
                processing_time_ms=processing_time
            )

    def _get_related_topics(self, topic: TopicCategory) -> List[str]:
        """Get topics related to the detected topic."""
        related_map = {
            TopicCategory.ANXIETY: [TopicCategory.MINDFULNESS, TopicCategory.COPING_SKILLS],
            TopicCategory.DEPRESSION: [TopicCategory.COPING_SKILLS, TopicCategory.THERAPY_TYPES],
            TopicCategory.STRESS: [TopicCategory.MINDFULNESS, TopicCategory.COPING_SKILLS],
            TopicCategory.GRIEF: [TopicCategory.COPING_SKILLS, TopicCategory.THERAPY_TYPES],
            TopicCategory.TRAUMA: [TopicCategory.THERAPY_TYPES, TopicCategory.COPING_SKILLS],
            TopicCategory.MINDFULNESS: [TopicCategory.STRESS, TopicCategory.ANXIETY],
            TopicCategory.COPING_SKILLS: [TopicCategory.MINDFULNESS, TopicCategory.STRESS],
        }
        
        related = related_map.get(topic, [])
        return [t.value for t in related[:2]]

    async def explain_concept(self, concept: str) -> str:
        """
        Explain a mental health concept in accessible language.
        
        Args:
            concept: The concept to explain
            
        Returns:
            Clear explanation
        """
        prompt = f"""
Explain the following mental health concept in clear, accessible language 
that someone without a psychology background could understand:

Concept: {concept}

Include:
1. A simple definition
2. Why it matters
3. A relatable example
4. One practical takeaway
"""
        return await self.generate_response(prompt, [])

    async def suggest_coping_technique(
        self,
        issue: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Suggest an appropriate coping technique based on the issue.
        
        Args:
            issue: The issue the user is facing
            user_preferences: User's preferences if known
            
        Returns:
            Suggested technique with instructions
        """
        topic = self.detect_topic(issue)
        resource = self.get_knowledge_resource(topic) if topic else None
        
        techniques = []
        if resource:
            techniques = resource.techniques
        
        if not techniques:
            techniques = self.KNOWLEDGE_BASE[TopicCategory.COPING_SKILLS].techniques
        
        prompt = f"""
The user is dealing with: {issue}
Available techniques: {techniques}
User preferences: {user_preferences or 'Not specified'}

Suggest ONE appropriate coping technique with:
1. Clear step-by-step instructions
2. Why it might help
3. Tips for making it effective
"""
        return await self.generate_response(prompt, [])

    def get_metrics(self) -> Dict[str, Any]:
        """Return agent metrics for observability."""
        return {
            "agent_type": self.agent_type.value,
            "topics_requested": self._topics_requested,
            "searches_performed": self._searches_performed,
            "total_topics_served": sum(self._topics_requested.values())
        }

