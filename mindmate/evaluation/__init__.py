"""
MindMate Evaluation Package

This package provides evaluation tools for testing agent quality:
- EmpathyEvaluator: Measures empathy in agent responses
- CrisisEvaluator: Tests crisis detection accuracy
"""

try:
    from evaluation.empathy_eval import EmpathyEvaluator, EmpathyMetrics
    from evaluation.crisis_eval import CrisisEvaluator, CrisisMetrics
except ImportError:
    from .empathy_eval import EmpathyEvaluator, EmpathyMetrics
    from .crisis_eval import CrisisEvaluator, CrisisMetrics

__all__ = [
    "EmpathyEvaluator",
    "EmpathyMetrics",
    "CrisisEvaluator",
    "CrisisMetrics",
]

