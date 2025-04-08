"""Agent module for the AI agent system.

This module contains the core agent implementation including:
- Agent: Main agent class that coordinates all components
- Mind: Cognitive processing and state management
- Conscious: Conscious thought processing
- Subconscious: Background processing and pattern recognition
- Memory: Memory management systems
"""

from .mind import Mind
from .cognition import Conscious
from .cognition import Subconscious
from .agent import Agent
from .emotion import Emotion

__all__ = [
    'Agent',
    'Mind',
    'Conscious', 
    'Subconscious',
    'Memory',
    'Emotion',
    'EmotionCenter',
    'Task',
    'MotivationCenter',
]
