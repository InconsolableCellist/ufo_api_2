"""Conscious processing for the agent's cognitive system."""

import logging
import json
from config import langfuse
from datetime import datetime

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
        
    def think(self, stimuli, subconscious_thoughts, trace_id, last_tool_results=None, emotional_state=None):
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
            last_tool_results (dict, optional): Results from last tool invocations
            emotional_state (dict, optional): Current emotional state
            
        Returns:
            str: Generated thought
        """
        # Prepare context for LLM
        context = {
            "emotional_state": emotional_state or self.emotion_center.get_state(),
            "short_term_memory": self.memory.recall(n=5),
            "subconscious_thoughts": subconscious_thoughts,
            "stimuli": stimuli,
            "current_focus": self.current_focus,
            "last_tool_results": last_tool_results
        }
        
        # Generate thought using LLM
        thought_span = langfuse.span(
            name="thought-generation", 
            parent_id=trace_id,
            input=json.dumps(context)
        )
        
        try:
            # Generate thought
            thought = self.llm.generate_thought(context)
            
            # Extract emotional implications and update state
            emotional_implications = self._extract_emotional_implications(thought)
            if emotional_implications:
                self.emotion_center.update(emotional_implications)
            
            thought_span.update(
                output=thought,
                metadata={
                    "emotion_update": bool(emotional_implications)
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

    def run_ego_cycle(self, recent_memories, emotional_state):
        """Run an ego-focused cycle for self-reflection.
        
        Args:
            recent_memories (list): Recent memories to reflect upon
            emotional_state (dict): Current emotional state
            
        Returns:
            str: Generated ego thoughts
        """
        # Create a Langfuse trace for the ego cycle
        ego_cycle_trace = langfuse.trace(
            name="ego-cycle",
            metadata={
                "cycle_type": "ego",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Log current state of ego thoughts
            current_ego_thoughts = self.ego_thoughts
            logger.info(f"Before ego cycle - Current ego_thoughts length: {len(current_ego_thoughts)} chars")
            
            # Check if ego thoughts have already been processed
            processed_ego_marker = "[PROCESSED]"
            if current_ego_thoughts and processed_ego_marker in current_ego_thoughts:
                # Remove the processed marker to get the original thoughts
                current_ego_thoughts = current_ego_thoughts.replace(processed_ego_marker, "")
                logger.info("Removed [PROCESSED] marker from ego thoughts")
            
            # Prepare context for ego thoughts
            context = {
                "emotional_state": emotional_state,
                "short_term_memory": recent_memories,
                "previous_ego_thoughts": current_ego_thoughts if current_ego_thoughts else None
            }
            
            # Update Langfuse with context information
            ego_cycle_trace.update(
                input=json.dumps({
                    "previous_ego_thoughts_length": len(current_ego_thoughts) if current_ego_thoughts else 0,
                    "memory_count": len(recent_memories)
                })
            )
            
            # Generate ego thoughts
            ego_thoughts = self.llm._generate_ego_thoughts(context)
            
            # Format ego thoughts with the specified prefix
            formatted_ego_thoughts = f"\n\n!!!\n***Suddenly, the following thought(s) occur to you. You try to ignore them but cannot, they echo in your mind for a full minute, completely diverting your attention before they fade, and you can think again:\n{ego_thoughts}\n***\n!!!\n\n"
            
            # Store ego thoughts for next cycle
            logger.info(f"Setting ego_thoughts - New length: {len(formatted_ego_thoughts)} chars")
            # Mark as processed to prevent re-formatting in future cycles
            self.ego_thoughts = f"{processed_ego_marker}{formatted_ego_thoughts}"
            logger.info("Marked ego cycle thoughts as processed")
            
            # Store the ego thoughts as a memory
            memory_entry = f"[EGO THOUGHTS]: {ego_thoughts}"
            self.memory.short_term.append(memory_entry)
            logger.info(f"Added ego thoughts to short-term memory")
            
            # Update Langfuse with results
            ego_cycle_trace.update(
                output=formatted_ego_thoughts[:200] if formatted_ego_thoughts else "",
                metadata={
                    "ego_thoughts_length": len(formatted_ego_thoughts) if formatted_ego_thoughts else 0,
                    "status": "completed"
                }
            )
            
            return formatted_ego_thoughts
            
        except Exception as e:
            ego_cycle_trace.update(
                error=str(e),
                metadata={"status": "error"}
            )
            logger.error(f"Error in ego cycle: {e}", exc_info=True)
            return f"Error in ego cycle: {str(e)}"
        finally:
            # No need to explicitly end the trace as update() marks it as complete
            pass 