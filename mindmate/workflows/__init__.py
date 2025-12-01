"""
MindMate Workflows Package

This package provides workflow orchestration for the multi-agent system:
- MainRouter: Intelligent routing of user messages to appropriate agents
- Workflow patterns: Sequential, parallel, and loop execution
- A2A coordination between agents
"""

try:
    from workflows.main_router import MainRouter, WorkflowResult, RoutingDecision
except ImportError:
    from .main_router import MainRouter, WorkflowResult, RoutingDecision

__all__ = [
    "MainRouter",
    "WorkflowResult",
    "RoutingDecision",
]

