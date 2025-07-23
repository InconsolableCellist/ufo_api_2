"""Main emotion module implementation."""

import logging
import json
from datetime import datetime

logger = logging.getLogger('agent_simulation.emotion')

class Emotion:
    """A single emotion with intensity and decay characteristics.
    
    Attributes:
        name (str): The name of the emotion
        intensity (float): Current intensity of the emotion (0-1)
        decay_rate (float): Rate at which the emotion naturally decays
        influence_factors (dict): Mapping of stimuli to their effects on this emotion
    """
    
    def __init__(self, name, intensity=0.0, decay_rate=0.1, influence_factors=None):
        self.name = name
        self.intensity = intensity
        self.decay_rate = decay_rate
        self.influence_factors = influence_factors or {}
        
    def update(self, stimuli):
        """Update emotion intensity based on stimuli and natural decay.
        
        Args:
            stimuli (dict): Dictionary of stimuli and their values
        """
        # Apply stimuli effects
        for stimulus, effect in self.influence_factors.items():
            if stimulus in stimuli:
                self.intensity += effect * stimuli[stimulus]
        
        # Apply natural decay
        self.intensity *= (1 - self.decay_rate)
        self.intensity = max(0, min(1, self.intensity))  # Clamp between 0-1

class EmotionCenter:
    """Manages a collection of emotions and overall emotional state.
    
    The EmotionCenter maintains multiple emotions, handles their updates,
    calculates overall mood, and provides natural language descriptions
    of emotional states using the LLM.
    
    TODO: Enhance emotion system:
    - Add more nuanced emotions (curiosity, awe, contentment, etc.)
    - Implement emotional combinations and complex states
    - Add emotional memory/history to track trends over time
    - Implement more sophisticated emotional dynamics (e.g., mood contagion)
    - Add emotion regulation mechanisms
    """
    
    def __init__(self):
        # Initialize with default emotions
        self.emotions = {
            'happiness': Emotion('happiness', intensity=0.3, decay_rate=0.05),
            'sadness': Emotion('sadness', intensity=0.1, decay_rate=0.03),
            'anger': Emotion('anger', intensity=0.1, decay_rate=0.08),
            'fear': Emotion('fear', intensity=0.1, decay_rate=0.1),
            'surprise': Emotion('surprise', intensity=0.2, decay_rate=0.15),
            'disgust': Emotion('disgust', intensity=0.05, decay_rate=0.07),
            'energy': Emotion('energy', intensity=0.4, decay_rate=0.02),
            'focus': Emotion('focus', intensity=0.5, decay_rate=0.03),
            'curiosity': Emotion('curiosity', intensity=0.2, decay_rate=0.05),
        }
        self.mood = 0.5  # Overall mood from -1 (negative) to 1 (positive)
        self.llm_client = None  # Will be set by the agent
        
        # Cache for emotional state descriptions
        self._emotional_state_cache = {
            "description": None,
            "last_emotions": None,
            "last_mood": None,
            "timestamp": None
        }
        
    def update(self, stimuli):
        """Update all emotions based on stimuli and recalculate mood.
        
        TODO: Implement more realistic emotion dynamics:
        - Consider emotion interaction (e.g., fear can amplify anger)
        - Add emotional inertia (resistance to sudden changes)
        - Implement habituation to stimuli
        - Add baseline personality traits that influence emotional responses
        
        Args:
            stimuli (dict): Dictionary of stimuli and their values
        """
        # Update all emotions
        for emotion in self.emotions.values():
            emotion.update(stimuli)
            
        # Calculate overall mood (weighted average)
        # TODO: Improve mood calculation with more factors:
        # - Consider personality baseline
        # - Add time-weighted averaging (recent emotions matter more)
        # - Implement emotional "momentum"
        happiness = self.emotions.get('happiness', Emotion('happiness')).intensity
        surprise = self.emotions.get('surprise', Emotion('surprise')).intensity
        focus = self.emotions.get('focus', Emotion('focus')).intensity
        curiosity = self.emotions.get('curiosity', Emotion('curiosity')).intensity
        sadness = self.emotions.get('sadness', Emotion('sadness')).intensity
        anger = self.emotions.get('anger', Emotion('anger')).intensity
        fear = self.emotions.get('fear', Emotion('fear')).intensity
        
        positive = happiness + surprise * 0.5 + focus * 0.3 + curiosity * 0.2
        negative = sadness + anger + fear
        self.mood = (positive - negative) / (positive + negative + 1e-6)  # Avoid division by zero
    
    def get_state(self):
        """Get a descriptive summary of the current emotional state.
        
        TODO: Enhance state reporting:
        - Add emotional trend analysis
        - Include dominant emotion identification
        - Add emotional complexity metrics
        - Report emotional stability indicators
        
        Returns:
            dict: Emotional state information including raw values and description
        """
        # Get the raw emotion values
        emotion_values = {name: emotion.intensity for name, emotion in self.emotions.items()}
        
        # If no LLM client is available, return just the raw values
        if not hasattr(self, 'llm_client') or self.llm_client is None:
            return emotion_values
            
        # Check if emotional state has changed significantly
        current_state = {
            "emotions": emotion_values,
            "mood": self.mood
        }
        
        # If we have a cached state, check if it's still valid
        if self._emotional_state_cache["last_emotions"] is not None:
            # Calculate change in emotions
            emotion_changes = {
                name: abs(current_state["emotions"][name] - self._emotional_state_cache["last_emotions"][name])
                for name in current_state["emotions"]
            }
            
            # Check if any emotion has changed significantly (more than 0.1)
            significant_change = any(change > 0.1 for change in emotion_changes.values())
            
            # Check if mood has changed significantly
            mood_change = abs(current_state["mood"] - self._emotional_state_cache["last_mood"])
            significant_change = significant_change or mood_change > 0.1
            
            if not significant_change:
                # Return cached description if no significant changes
                return {
                    "raw_emotions": emotion_values,
                    "mood": self.mood,
                    "description": self._emotional_state_cache["description"]
                }
        
        # Prepare context for the LLM
        prompt = f"""
        Current emotional intensities:
        {json.dumps(emotion_values, indent=2)}
        
        Overall mood: {self.mood:.2f} (-1 to 1 scale)
        
        Based on these emotional intensities and overall mood, provide a brief, natural description
        of the emotional state from the agent's perspective. Focus on the dominant emotions
        and their interplay. Keep the description to 2-3 sentences.
        
        Format your response as a direct first-person statement of emotional awareness.
        """
        
        system_message = "You are an AI agent's emotional awareness. Describe the emotional state naturally and introspectively."
        
        try:
            description = self.llm_client._generate_completion(prompt, system_message)
            
            # Update cache
            self._emotional_state_cache = {
                "description": description,
                "last_emotions": emotion_values.copy(),
                "last_mood": self.mood,
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "raw_emotions": emotion_values,
                "mood": self.mood,
                "description": description
            }
        except Exception as e:
            logger.error(f"Error generating emotional state description: {e}", exc_info=True)
            return emotion_values 