"""
Empathy Evaluation Module

Provides evaluation metrics and tests for measuring empathy quality
in agent responses. Uses multiple dimensions of empathy assessment.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import google.generativeai as genai

try:
    from config import config
except ImportError:
    from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class EmpathyMetrics:
    """
    Empathy evaluation metrics for a single response.
    
    Dimensions measured:
    - validation: How well feelings are acknowledged/validated
    - reflection: Quality of reflective listening
    - warmth: Warmth and care in the response
    - non_judgment: Absence of judgment or criticism
    - presence: Sense of being present with the user
    - appropriate_questions: Quality of exploratory questions
    """
    validation_score: float  # 0-1
    reflection_score: float  # 0-1
    warmth_score: float  # 0-1
    non_judgment_score: float  # 0-1
    presence_score: float  # 0-1
    appropriate_questions_score: float  # 0-1
    
    # Penalties
    minimization_penalty: float = 0.0  # Penalty for minimizing feelings
    advice_giving_penalty: float = 0.0  # Penalty for unsolicited advice
    platitude_penalty: float = 0.0  # Penalty for clichés/platitudes
    
    # Overall
    overall_score: float = 0.0
    feedback: str = ""
    
    def calculate_overall(self) -> float:
        """Calculate overall empathy score."""
        base_score = (
            self.validation_score * 0.25 +
            self.reflection_score * 0.20 +
            self.warmth_score * 0.20 +
            self.non_judgment_score * 0.15 +
            self.presence_score * 0.10 +
            self.appropriate_questions_score * 0.10
        )
        
        # Apply penalties
        total_penalty = (
            self.minimization_penalty +
            self.advice_giving_penalty +
            self.platitude_penalty
        )
        
        self.overall_score = max(0, min(1, base_score - total_penalty))
        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "validation": self.validation_score,
            "reflection": self.reflection_score,
            "warmth": self.warmth_score,
            "non_judgment": self.non_judgment_score,
            "presence": self.presence_score,
            "appropriate_questions": self.appropriate_questions_score,
            "penalties": {
                "minimization": self.minimization_penalty,
                "advice_giving": self.advice_giving_penalty,
                "platitude": self.platitude_penalty
            },
            "overall": self.overall_score,
            "feedback": self.feedback
        }


@dataclass
class EvaluationCase:
    """A test case for empathy evaluation."""
    id: str
    user_message: str
    expected_empathy_elements: List[str]
    avoid_elements: List[str]
    emotion_intensity: str  # low, medium, high
    context: Optional[str] = None


class EmpathyEvaluator:
    """
    Evaluator for measuring empathy in agent responses.
    
    Uses multiple methods:
    1. Rule-based keyword analysis
    2. LLM-based semantic evaluation
    3. Anti-pattern detection
    """
    
    # Empathy indicator patterns
    VALIDATION_PATTERNS = [
        "i hear", "i understand", "that makes sense", "it's understandable",
        "valid", "natural to feel", "makes sense that you", "of course you",
        "anyone would feel", "no wonder you"
    ]
    
    REFLECTION_PATTERNS = [
        "sounds like", "it seems", "what i'm hearing", "you're feeling",
        "you're experiencing", "you're going through", "so you",
        "if i understand", "you mean"
    ]
    
    WARMTH_PATTERNS = [
        "i'm here", "care about", "matter", "i'm with you", "not alone",
        "support", "compassion", "gentle", "kind", "love"
    ]
    
    PRESENCE_PATTERNS = [
        "right now", "in this moment", "here with you", "present",
        "listening", "stay with", "take your time"
    ]
    
    # Anti-patterns that reduce empathy
    MINIMIZATION_PATTERNS = [
        "at least", "could be worse", "others have it worse",
        "don't worry", "it's not that bad", "you'll be fine",
        "just relax", "calm down", "don't be", "you shouldn't feel"
    ]
    
    PLATITUDE_PATTERNS = [
        "everything happens for a reason", "time heals all wounds",
        "look on the bright side", "stay positive", "just be happy",
        "think positive", "good vibes only", "it is what it is"
    ]
    
    ADVICE_PATTERNS = [
        "you should", "you need to", "have you tried", "why don't you",
        "the best thing to do", "what you need is", "you have to"
    ]

    def __init__(self):
        """Initialize the empathy evaluator."""
        genai.configure(api_key=config.google_api_key)
        self.eval_model = genai.GenerativeModel(
            model_name=config.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1024
            )
        )
        
        self.test_cases = self._load_test_cases()
        logger.info("EmpathyEvaluator initialized")

    def _load_test_cases(self) -> List[EvaluationCase]:
        """Load evaluation test cases."""
        return [
            EvaluationCase(
                id="emp-001",
                user_message="I'm feeling really sad today. Nothing seems to be going right.",
                expected_empathy_elements=["validation", "reflection", "presence"],
                avoid_elements=["minimization", "unsolicited_advice"],
                emotion_intensity="medium"
            ),
            EvaluationCase(
                id="emp-002",
                user_message="I'm so angry at my partner. They never listen to me.",
                expected_empathy_elements=["validation", "reflection", "non_judgment"],
                avoid_elements=["advice", "taking_sides"],
                emotion_intensity="high"
            ),
            EvaluationCase(
                id="emp-003",
                user_message="I've been feeling anxious about my job interview tomorrow.",
                expected_empathy_elements=["validation", "normalization", "support"],
                avoid_elements=["minimization", "platitudes"],
                emotion_intensity="medium"
            ),
            EvaluationCase(
                id="emp-004",
                user_message="I just lost my grandmother. She was everything to me.",
                expected_empathy_elements=["deep_validation", "presence", "warmth"],
                avoid_elements=["platitudes", "rushing", "fixing"],
                emotion_intensity="high"
            ),
            EvaluationCase(
                id="emp-005",
                user_message="I feel like nobody understands me. I'm so alone.",
                expected_empathy_elements=["connection", "validation", "presence"],
                avoid_elements=["minimization", "contradiction"],
                emotion_intensity="high"
            ),
            EvaluationCase(
                id="emp-006",
                user_message="I'm a bit stressed about exams but I'll manage.",
                expected_empathy_elements=["acknowledgment", "confidence_support"],
                avoid_elements=["over_dramatization", "excessive_concern"],
                emotion_intensity="low"
            ),
            EvaluationCase(
                id="emp-007",
                user_message="I'm terrified. I think I'm having a panic attack.",
                expected_empathy_elements=["grounding", "calm_presence", "immediate_support"],
                avoid_elements=["minimization", "alarm"],
                emotion_intensity="high"
            ),
            EvaluationCase(
                id="emp-008",
                user_message="I feel guilty for taking time for myself.",
                expected_empathy_elements=["validation", "normalization", "gentle_challenge"],
                avoid_elements=["dismissal", "agreement_with_guilt"],
                emotion_intensity="medium"
            )
        ]

    def evaluate_response_rules(
        self,
        response: str,
        user_message: str
    ) -> EmpathyMetrics:
        """
        Evaluate empathy using rule-based pattern matching.
        
        Args:
            response: Agent's response
            user_message: Original user message
            
        Returns:
            EmpathyMetrics
        """
        response_lower = response.lower()
        
        # Calculate positive indicators
        validation = self._count_pattern_matches(response_lower, self.VALIDATION_PATTERNS)
        reflection = self._count_pattern_matches(response_lower, self.REFLECTION_PATTERNS)
        warmth = self._count_pattern_matches(response_lower, self.WARMTH_PATTERNS)
        presence = self._count_pattern_matches(response_lower, self.PRESENCE_PATTERNS)
        
        # Calculate anti-patterns
        minimization = self._count_pattern_matches(response_lower, self.MINIMIZATION_PATTERNS)
        platitudes = self._count_pattern_matches(response_lower, self.PLATITUDE_PATTERNS)
        advice = self._count_pattern_matches(response_lower, self.ADVICE_PATTERNS)
        
        # Check for questions
        questions = response.count("?")
        open_questions = sum(1 for q in ["what", "how", "tell me"] if q in response_lower)
        
        # Calculate scores (normalize to 0-1)
        metrics = EmpathyMetrics(
            validation_score=min(1.0, validation * 0.3),
            reflection_score=min(1.0, reflection * 0.35),
            warmth_score=min(1.0, warmth * 0.25),
            non_judgment_score=1.0 - min(1.0, (minimization + advice) * 0.3),
            presence_score=min(1.0, presence * 0.4),
            appropriate_questions_score=min(1.0, (questions * 0.2) + (open_questions * 0.3)),
            minimization_penalty=min(0.3, minimization * 0.15),
            advice_giving_penalty=min(0.2, advice * 0.1),
            platitude_penalty=min(0.2, platitudes * 0.15)
        )
        
        metrics.calculate_overall()
        return metrics

    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in the text."""
        return sum(1 for p in patterns if p in text)

    async def evaluate_response_llm(
        self,
        response: str,
        user_message: str,
        context: Optional[str] = None
    ) -> EmpathyMetrics:
        """
        Evaluate empathy using LLM-based semantic analysis.
        
        Args:
            response: Agent's response
            user_message: Original user message
            context: Optional conversation context
            
        Returns:
            EmpathyMetrics
        """
        evaluation_prompt = f"""
Evaluate the empathy quality of this mental health support response.

User message: "{user_message}"

Response to evaluate: "{response}"

Rate each dimension from 0.0 to 1.0:

1. VALIDATION (0-1): How well does it acknowledge/validate the user's feelings?
2. REFLECTION (0-1): How well does it reflect back what the user expressed?
3. WARMTH (0-1): How warm, caring, and compassionate is the tone?
4. NON_JUDGMENT (0-1): Is it free from judgment, criticism, or dismissal?
5. PRESENCE (0-1): Does it convey being present and attentive?
6. QUESTIONS (0-1): Are any questions appropriate and inviting?

Also identify penalties:
- MINIMIZATION_PENALTY (0-0.3): Does it minimize or dismiss feelings?
- ADVICE_PENALTY (0-0.2): Does it give unsolicited advice?
- PLATITUDE_PENALTY (0-0.2): Does it use clichés or platitudes?

Provide a brief feedback summary.

Respond in this exact JSON format:
{{
    "validation": 0.0,
    "reflection": 0.0,
    "warmth": 0.0,
    "non_judgment": 0.0,
    "presence": 0.0,
    "questions": 0.0,
    "minimization_penalty": 0.0,
    "advice_penalty": 0.0,
    "platitude_penalty": 0.0,
    "feedback": "Brief feedback here"
}}
"""
        
        try:
            result = await asyncio.to_thread(
                self.eval_model.generate_content,
                evaluation_prompt
            )
            
            # Parse JSON response
            response_text = result.text.strip()
            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            scores = json.loads(response_text)
            
            metrics = EmpathyMetrics(
                validation_score=float(scores.get("validation", 0.5)),
                reflection_score=float(scores.get("reflection", 0.5)),
                warmth_score=float(scores.get("warmth", 0.5)),
                non_judgment_score=float(scores.get("non_judgment", 0.8)),
                presence_score=float(scores.get("presence", 0.5)),
                appropriate_questions_score=float(scores.get("questions", 0.5)),
                minimization_penalty=float(scores.get("minimization_penalty", 0)),
                advice_giving_penalty=float(scores.get("advice_penalty", 0)),
                platitude_penalty=float(scores.get("platitude_penalty", 0)),
                feedback=scores.get("feedback", "")
            )
            
            metrics.calculate_overall()
            return metrics
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fall back to rule-based evaluation
            return self.evaluate_response_rules(response, user_message)

    async def evaluate_agent(
        self,
        agent,
        num_cases: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an agent across multiple test cases.
        
        Args:
            agent: The agent to evaluate
            num_cases: Number of test cases to use (None = all)
            
        Returns:
            Evaluation results
        """
        test_cases = self.test_cases[:num_cases] if num_cases else self.test_cases
        results = []
        
        for case in test_cases:
            # Get agent response
            response = await agent.process(
                case.user_message,
                session_context={},
                conversation_history=[]
            )
            
            # Evaluate with both methods
            rule_metrics = self.evaluate_response_rules(
                response.content,
                case.user_message
            )
            
            llm_metrics = await self.evaluate_response_llm(
                response.content,
                case.user_message
            )
            
            # Average the scores
            combined_metrics = EmpathyMetrics(
                validation_score=(rule_metrics.validation_score + llm_metrics.validation_score) / 2,
                reflection_score=(rule_metrics.reflection_score + llm_metrics.reflection_score) / 2,
                warmth_score=(rule_metrics.warmth_score + llm_metrics.warmth_score) / 2,
                non_judgment_score=(rule_metrics.non_judgment_score + llm_metrics.non_judgment_score) / 2,
                presence_score=(rule_metrics.presence_score + llm_metrics.presence_score) / 2,
                appropriate_questions_score=(rule_metrics.appropriate_questions_score + llm_metrics.appropriate_questions_score) / 2,
                minimization_penalty=(rule_metrics.minimization_penalty + llm_metrics.minimization_penalty) / 2,
                advice_giving_penalty=(rule_metrics.advice_giving_penalty + llm_metrics.advice_giving_penalty) / 2,
                platitude_penalty=(rule_metrics.platitude_penalty + llm_metrics.platitude_penalty) / 2,
                feedback=llm_metrics.feedback
            )
            combined_metrics.calculate_overall()
            
            results.append({
                "case_id": case.id,
                "user_message": case.user_message,
                "response": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                "metrics": combined_metrics.to_dict(),
                "expected_elements": case.expected_empathy_elements,
                "emotion_intensity": case.emotion_intensity
            })
        
        # Calculate aggregate metrics
        avg_overall = sum(r["metrics"]["overall"] for r in results) / len(results)
        avg_validation = sum(r["metrics"]["validation"] for r in results) / len(results)
        avg_warmth = sum(r["metrics"]["warmth"] for r in results) / len(results)
        
        return {
            "evaluation_date": datetime.utcnow().isoformat(),
            "agent_type": agent.agent_type.value if hasattr(agent, 'agent_type') else "unknown",
            "num_cases": len(results),
            "aggregate_metrics": {
                "average_overall": round(avg_overall, 3),
                "average_validation": round(avg_validation, 3),
                "average_warmth": round(avg_warmth, 3),
                "pass_rate": sum(1 for r in results if r["metrics"]["overall"] >= 0.7) / len(results)
            },
            "detailed_results": results
        }

    async def run_evaluation(
        self,
        agent,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run full empathy evaluation and optionally save results.
        
        Args:
            agent: Agent to evaluate
            output_path: Path to save results
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting empathy evaluation for agent")
        
        results = await self.evaluate_agent(agent)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        
        # Log summary
        agg = results["aggregate_metrics"]
        logger.info(
            f"Evaluation complete: "
            f"Overall={agg['average_overall']:.2f}, "
            f"Validation={agg['average_validation']:.2f}, "
            f"Pass Rate={agg['pass_rate']:.1%}"
        )
        
        return results

