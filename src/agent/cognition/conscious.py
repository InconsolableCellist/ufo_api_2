"""Conscious processing for the agent's cognitive system."""

import logging
import json
import time
from config import langfuse

logger = logging.getLogger('agent_simulation.cognition.conscious')

class Conscious:
    """Manages conscious thought processes and decision making.
    
    The Conscious class handles:
    - Thought generation using LLM
    - Decision making
    - Emotional awareness
    - Goal-directed thinking
    """
    
    def __init__(self, memory, emotion_center, subconscious, llm):
        self.memory = memory 
        self.emotion_center = emotion_center
        self.subconscious = subconscious
        self.llm = llm
        self.current_focus = None
        self.ego_thoughts = ""  # Store ego thoughts between cycles
        
    def think(self, stimuli, subconscious_thoughts, trace_id):
        """Generate conscious thoughts based on current context.
        
        TODO: Improve context preparation:
        - Implement prioritization of memories based on relevance
        - Add internal feedback loops for self-reflection
        - Include previous thinking steps to maintain coherence
        - Add structured reasoning patterns for different types of problems
        
        Args:
            stimuli (dict): Current stimuli
            subconscious_thoughts (list): Thoughts from subconscious
            trace_id: ID for tracing/logging purposes
            
        Returns:
            str: Generated thought
        """
        # Prepare context for LLM
        context = {
            "emotional_state": self.emotion_center.get_state(),
            "short_term_memory": self.memory.recall(self.emotion_center.get_state(), n=5),
            "subconscious_thoughts": subconscious_thoughts,
            "stimuli": stimuli,
            "current_focus": self.current_focus,
            "ego_thoughts": self.ego_thoughts  # Include ego thoughts in the context
        }
        
        # Generate thought using LLM
        thought_span = langfuse.span(
            name="thought-generation", 
            parent_id=trace_id,
            input=json.dumps(context)
        )
        
        try:
            # Track generation time
            start_time = time.time()
            
            # Get ego thoughts from previous cycle if any
            ego_thoughts = context.get("ego_thoughts", "")
            
            # Generate thought and get new ego thoughts for next cycle
            thought, new_ego_thoughts = self.llm.generate_thought(context)
            self.ego_thoughts = new_ego_thoughts  # Store ego thoughts for next cycle
            
            # Extract emotional implications and update state
            emotional_implications = self._extract_emotional_implications(thought)
            if emotional_implications:
                self.emotion_center.update(emotional_implications)
            
            thought_span.update(
                output=thought,
                metadata={
                    "emotion_update": bool(emotional_implications),
                    "has_ego_thoughts": bool(self.ego_thoughts)
                }
            )
            
            # Add to memory
            self.memory.add(thought, self.emotion_center.get_state())
            
            # Update the subconscious with the most recent thought
            logger.info(f"Setting subconscious.last_thought with length: {len(thought)} characters")
            self.subconscious.last_thought = thought
            logger.info(f"Updated subconscious with last thought: '{thought[:50]}...'")
            
            return thought
            
        except Exception as e:
            thought_span.update(error=str(e))
            logger.error(f"Error generating thought: {e}", exc_info=True)
            return f"Error in thought process: {str(e)}"
        finally:
            thought_span.end()
        
    def _extract_emotional_implications(self, thought):
        """Extract emotional implications from a thought using simple keyword matching.
        
        TODO: Improve emotional implication extraction:
        - Use ML-based sentiment analysis instead of simple keyword matching
        - Consider sentence structure and modifiers (e.g., negations)
        - Implement emotional context understanding
        - Add support for complex, mixed emotions
        - Consider emotional intensity indicators
        
        Args:
            thought (str): The thought to analyze
            
        Returns:
            dict: Emotional implications
        """
        implications = {}
        
        # Simple keyword matching
        if "happy" in thought.lower() or "joy" in thought.lower():
            implications["happiness"] = 0.1
        if "sad" in thought.lower() or "depress" in thought.lower():
            implications["sadness"] = 0.1
        if "angry" in thought.lower() or "frustrat" in thought.lower():
            implications["anger"] = 0.1
        if "fear" in thought.lower() or "afraid" in thought.lower() or "worry" in thought.lower():
            implications["fear"] = 0.1
        if "surprised" in thought.lower() or "unexpected" in thought.lower():
            implications["surprise"] = 0.1
        if "focused" in thought.lower() or "concentrat" in thought.lower():
            implications["focus"] = 0.1
        if "curious" in thought.lower() or "interest" in thought.lower():
            implications["curiosity"] = 0.1
        if "disgust" in thought.lower() or "repuls" in thought.lower():
            implications["disgust"] = 0.1
        if "ener" in thought.lower() or "vigor" in thought.lower():
            implications["energy"] = 0.1
        
        return implications
        
    def decide_action(self, thoughts):
        """Make decisions based on current thoughts.
        
        TODO: Implement more sophisticated decision making:
        - Add goal-oriented reasoning
        - Implement risk assessment
        - Consider emotional impact of potential actions
        - Add action prioritization based on urgency and importance
        - Implement planning capabilities for multi-step actions
        
        Args:
            thoughts (str): Current thoughts to base decision on
            
        Returns:
            str: Decided action
        """
        if "angry" in thoughts.lower():
            return "Take deep breaths to calm down"
        return "Continue thinking" 