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
    - Memory systems (short-term and long-term)
    - Emotional processing
    - Motivational drives
    - Conscious and subconscious processes
    
    TODO: Enhance cognitive architecture:
    - Implement metacognition capabilities
    - Add cognitive biases and heuristics
    - Improve integration between components
    - Add attention filtering
    - Implement better memory consolidation
    """
    
    def __init__(self, llm, memory_path=None, emotion_path=None):
        """Initialize the mind's cognitive components.
        
        Args:
            llm: Language model interface
            memory_path (str, optional): Path to persist memory state
            emotion_path (str, optional): Path to persist emotional state
        """
        # Initialize core components
        self.memory = Memory(persist_path=memory_path)
        self.emotion_center = EmotionCenter()
        self.emotion_center.llm_client = llm
        self.motivation_center = MotivationCenter()
        
        # Initialize processing components
        self.subconscious = Subconscious(self.memory, self.emotion_center)
        self.conscious = Conscious(
            self.memory, 
            self.emotion_center, 
            self.subconscious,
            llm
        )
        
        # Load emotional state if path provided
        if emotion_path and os.path.exists(emotion_path):
            self.load_emotional_state(emotion_path)
            
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
            
    def _update_motivation(self, thought):
        """Update motivational state based on conscious thought.
        
        Args:
            thought (str): Current conscious thought
        """
        # Extract achievements from thought
        achievements = []
        if "complete" in thought.lower() or "finish" in thought.lower():
            achievements.append("Task completion")
        if "learn" in thought.lower() or "understand" in thought.lower():
            achievements.append("Learning")
            
        # Update drives based on emotional state and achievements
        self.motivation_center.update_drives(
            self.emotion_center.get_state(),
            achievements
        )
        
        # Update task priorities based on emotional state
        self.motivation_center.update_tasks(self.emotion_center.get_state())
            
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