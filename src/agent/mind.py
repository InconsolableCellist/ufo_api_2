"""Mind module for managing the agent's cognitive processes."""

import logging
import pickle
import os
from .emotion import EmotionCenter
from .motivation import MotivationCenter
from .cognition.subconscious import Subconscious
from .cognition.conscious import Conscious
from .cognition.memory import Memory
from datetime import datetime

logger = logging.getLogger('agent_simulation.agent.mind')

class Mind:
    """Manages the agent's cognitive processes and mental state.
    
    The Mind class coordinates:
    - Subconscious processing
    - Conscious thinking
    - Emotional state
    - Memory management
    - Motivation and goals
    """
    
    def __init__(self, memory, emotion_center, motivation_center, subconscious, conscious):
        """Initialize the mind with its components.
        
        Args:
            memory: Memory manager instance
            emotion_center: Emotion center instance
            motivation_center: Motivation center instance
            subconscious: Subconscious processor instance
            conscious: Conscious processor instance
        """
        self.memory = memory
        self.emotion_center = emotion_center
        self.motivation_center = motivation_center
        self.subconscious = subconscious
        self.conscious = conscious
        
    def _update_motivation(self, thought):
        """Update motivation based on conscious thought.
        
        Args:
            thought: The conscious thought to process
        """
        # Extract goals and tasks from thought
        if hasattr(thought, 'goals'):
            for goal in thought.goals:
                self.motivation_center.add_goal(goal)
                
        if hasattr(thought, 'tasks'):
            for task in thought.tasks:
                self.motivation_center.add_task(task)
                
    def shutdown(self):
        """Shutdown all mind components."""
        self.memory.shutdown()
        self.emotion_center.shutdown()
        self.motivation_center.shutdown()
        self.subconscious.shutdown()
        self.conscious.shutdown()

    def run_conscious_thought(self, context):
        """Run the conscious thinking process with the provided context.
        
        This is part of the simplified execution cycle.
        
        Args:
            context (dict): The context for conscious thought, including:
                - recent_tool_results: Results from recently executed tools
                - short_term_memories: Recent memories for context
                - emotional_state: Current emotional state
                - physical_state: Current physical state
                - available_tools: Available tools for the agent
                
        Returns:
            dict: The result of conscious thought, including:
                - thought: The generated thought
                - tool_invocations: List of requested tool invocations
        """
        logger.info("Running conscious thought process")
        
        # Prepare prompt for the conscious thought
        system_message = """You are an autonomous AI agent's conscious mind.
        Based on the provided context, generate a coherent thought that reflects
        your current state, processing recent memories and determining what actions to take next.
        Consider your emotional state and physical state while thinking.
        
        If you need to use any tools, include them as tool invocations in your response.
        """
        
        # Construct a prompt with the context
        prompt = f"""
        CONTEXT:
        
        Recent memories:
        {context.get('short_term_memories', [])}
        
        Recent tool results:
        {context.get('recent_tool_results', [])}
        
        Emotional state:
        {context.get('emotional_state', {})}
        
        Physical state:
        {context.get('physical_state', {})}
        
        Available tools:
        {context.get('available_tools', [])}
        
        Based on this context, generate your current conscious thought.
        Think about your current situation, any recent experiences,
        and what you should do next.
        
        If you want to use any tools, specify them in the response.
        Tool usage template: [TOOL: tool_name(param1:value1, param2:value2)]
        """
        
        try:
            # Generate the conscious thought
            thought_response = self.conscious.think(
                context,
                self.subconscious.process(datetime.now().isoformat()),
                datetime.now().isoformat()
            )
            
            # Extract any tool invocations from the response
            tool_invocations = self._extract_tool_invocations(thought_response)
            
            # Store the thought in memory
            self.memory.add_memory(thought_response)
            
            # Update the subconscious with the most recent thought
            self.subconscious.last_thought = thought_response
            
            return {
                "thought": thought_response,
                "tool_invocations": tool_invocations
            }
            
        except Exception as e:
            logger.error(f"Error in conscious thought generation: {e}", exc_info=True)
            return {
                "thought": f"Error in thought process: {str(e)}",
                "tool_invocations": []
            }
    
    def run_ego_reflection(self, context):
        """Run the ego (self-reflective) thinking process.
        
        This is part of the simplified execution cycle and runs periodically.
        
        Args:
            context (dict): The context for ego reflection, including:
                - recent_memories: A few recent memories for reflection
                - emotional_state: Current emotional state
                - physical_state: Current physical state
                
        Returns:
            dict: The result of ego reflection
        """
        logger.info("Running ego reflection process")
        
        # Prepare prompt for the ego reflection
        system_message = """You are an autonomous AI agent's reflective consciousness.
        Based on the provided context, engage in deeper self-reflection about your
        recent experiences, emotional state, and overall existence.
        
        This is a more philosophical and introspective mode of thinking where you
        can contemplate your own thoughts, beliefs, and emotions at a meta level.
        
        Focus on questions like: Who am I? What am I trying to achieve? How am I feeling 
        and why? How have I changed? What values are important to me?
        """
        
        # Construct a prompt with the context
        prompt = f"""
        CONTEXT FOR REFLECTION:
        
        Recent memories to reflect upon:
        {context.get('recent_memories', [])}
        
        Current emotional state:
        {context.get('emotional_state', {})}
        
        Current physical state:
        {context.get('physical_state', {})}
        
        Engage in deep self-reflection about your recent experiences, 
        your emotional state, your goals, your values, and how you've been changing.
        
        Consider what these experiences mean to you on a deeper level.
        Reflect on patterns in your thoughts or behaviors.
        Think about your purpose and how well your recent actions align with it.
        """
        
        try:
            # Generate the ego reflection
            reflection = self.conscious.think(
                context,
                self.subconscious.process(datetime.now().isoformat()),
                datetime.now().isoformat()
            )
            
            # Store the reflection in memory
            self.memory.add_memory("Ego reflection: " + reflection)
            
            return {
                "reflection": reflection
            }
            
        except Exception as e:
            logger.error(f"Error in ego reflection generation: {e}", exc_info=True)
            return {
                "reflection": f"Error in reflection process: {str(e)}"
            }
    
    def _extract_tool_invocations(self, thought):
        """Extract tool invocations from a thought string.
        
        Tool invocations are expected in the format:
        [TOOL: tool_name(param1:value1, param2:value2)]
        
        Args:
            thought (str): The thought text to extract tool invocations from
            
        Returns:
            list: List of tool invocation dictionaries
        """
        tool_invocations = []
        
        # Simple pattern matching for tool invocations
        # A more robust implementation would use regex
        if '[TOOL:' in thought:
            # Split by tool invocation start tag
            parts = thought.split('[TOOL:')
            
            # Skip the first part (before any tool invocation)
            for part in parts[1:]:
                # Find the end of the tool invocation
                if ']' in part:
                    # Extract just the tool invocation part
                    tool_part = part.split(']')[0].strip()
                    
                    # Parse the tool name and parameters
                    if '(' in tool_part and ')' in tool_part:
                        tool_name = tool_part.split('(')[0].strip()
                        params_str = tool_part.split('(')[1].split(')')[0]
                        
                        # Parse parameters
                        params = {}
                        if params_str:
                            param_parts = params_str.split(',')
                            for param in param_parts:
                                if ':' in param:
                                    key, value = param.split(':', 1)
                                    params[key.strip()] = value.strip()
                        
                        # Add the parsed tool invocation
                        tool_invocations.append({
                            "name": tool_name,
                            "params": params
                        })
        
        return tool_invocations

    def process_step(self, stimuli):
        """Process a cognitive cycle step.
        
        Args:
            stimuli (dict): Current sensory input
            
        Returns:
            dict: Processing results
        """
        try:
            # Update emotional state based on stimuli
            self.emotion_center.update(stimuli)
            
            # Generate a trace_id for this processing cycle
            trace_id = datetime.now().isoformat()
            
            # Run subconscious processes
            subconscious_thoughts = self.subconscious.process(trace_id)
            
            # Conscious thinking
            conscious_thought = self.conscious.think(
                stimuli, 
                subconscious_thoughts,
                trace_id
            )
            
            # Update motivation based on thoughts
            self._update_motivation(conscious_thought)
            
            # Decision making
            action = self.conscious.decide_action(conscious_thought)
            
            return {
                "emotional_state": self.emotion_center.get_state(),
                "subconscious_thoughts": subconscious_thoughts,
                "conscious_thought": conscious_thought,
                "action": action,
                "ego_thoughts": self.conscious.ego_thoughts,
                "motivation_state": self.motivation_center.get_motivation_state()
            }
            
        except Exception as e:
            logger.error(f"Error in cognitive processing: {e}", exc_info=True)
            return {
                "emotional_state": self.emotion_center.get_state(),
                "subconscious_thoughts": "Error in processing",
                "conscious_thought": f"Error occurred: {str(e)}",
                "action": "None - system error",
                "ego_thoughts": "",
                "motivation_state": {}
            }
            
    def save_emotional_state(self, path):
        """Save the current emotional state to disk.
        
        Args:
            path (str): Path to save state
            
        Returns:
            bool: Success status
        """
        try:
            emotional_data = {
                'emotions': {
                    name: emotion.intensity 
                    for name, emotion in self.emotion_center.emotions.items()
                },
                'mood': self.emotion_center.mood
            }
            
            with open(path, 'wb') as f:
                pickle.dump(emotional_data, f)
                
            logger.info(f"Emotional state saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving emotional state: {e}", exc_info=True)
            return False
    
    def load_emotional_state(self, path):
        """Load emotional state from disk.
        
        Args:
            path (str): Path to load state from
            
        Returns:
            bool: Success status
        """
        try:
            with open(path, 'rb') as f:
                emotional_data = pickle.load(f)
                
            # Default emotions that should exist in every EmotionCenter
            default_emotions = {
                'happiness': {'intensity': 0.3, 'decay_rate': 0.05},
                'sadness': {'intensity': 0.1, 'decay_rate': 0.03},
                'anger': {'intensity': 0.1, 'decay_rate': 0.08},
                'fear': {'intensity': 0.1, 'decay_rate': 0.1},
                'surprise': {'intensity': 0.2, 'decay_rate': 0.15},
                'disgust': {'intensity': 0.05, 'decay_rate': 0.07},
                'energy': {'intensity': 0.4, 'decay_rate': 0.02},
                'focus': {'intensity': 0.5, 'decay_rate': 0.03},
                'curiosity': {'intensity': 0.2, 'decay_rate': 0.05},
            }
            
            # Update emotion intensities from loaded data
            for name, intensity in emotional_data.get('emotions', {}).items():
                if name in self.emotion_center.emotions:
                    self.emotion_center.emotions[name].intensity = intensity
            
            # Check for any missing emotions and add them with default values
            from .emotion import Emotion
            for name, params in default_emotions.items():
                if name not in self.emotion_center.emotions:
                    logger.info(f"Adding missing emotion '{name}' with default values")
                    self.emotion_center.emotions[name] = Emotion(
                        name=name,
                        intensity=params['intensity'],
                        decay_rate=params['decay_rate']
                    )
                    
            # Update mood
            if 'mood' in emotional_data:
                self.emotion_center.mood = emotional_data['mood']
                
            logger.info(f"Emotional state loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading emotional state: {e}", exc_info=True)
            return False 