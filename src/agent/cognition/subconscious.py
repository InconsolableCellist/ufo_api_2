"""Subconscious processing for the agent's cognitive system."""

import logging
import random

logger = logging.getLogger('agent_simulation.cognition.subconscious')

class Subconscious:
    """Manages subconscious processes and background thoughts.
    
    The Subconscious class handles:
    - Background memory processing
    - Random thought generation
    - Emotional processing
    - Pattern recognition
    
    TODO: Add more sophisticated background processes:
    - Dream-like processing during idle periods
    - Pattern recognition for underlying connections between memories
    - Emotional association building
    - Subconscious learning and adaptation
    """
    
    def __init__(self, memory, emotion_center):
        self.memory = memory
        self.emotion_center = emotion_center
        self.background_processes = [
            self._surface_memories
        ] # TODO: Add random thoughts and emotional processing back in
        self.last_thought = None  # Track the most recent thought
        self.current_emotional_state = None  # Store current emotional state
        
    def process(self, trace_id, recent_memories=None, emotional_state=None):
        """Run all background processes.
        
        Args:
            trace_id: ID for tracing/logging purposes
            recent_memories (list, optional): Recent memories to consider
            emotional_state (dict, optional): Current emotional state
            
        Returns:
            list: Generated thoughts and surfaced memories
        """
        # Use provided emotional state or get from emotion center
        if emotional_state is None:
            emotional_state = self.emotion_center.get_state()
            
        # Store current emotional state for use in background processes
        self.current_emotional_state = emotional_state
            
        # Store recent memories if provided
        if recent_memories is not None:
            self.last_thought = recent_memories[-1] if recent_memories else None
            
        thoughts = []
        for process in self.background_processes:
            thoughts.extend(process(trace_id))
        return thoughts
        
    def set_focus_thought(self, thought):
        """Explicitly set a thought for the subconscious to focus on.
        
        Args:
            thought (str): The thought to focus on
            
        Returns:
            bool: True if focus was set successfully
        """
        if thought:
            logger.info(f"Explicitly setting subconscious focus thought: '{thought[:50]}...'")
            # Debug the input type and length
            if not isinstance(thought, str):
                logger.warning(f"Non-string thought received: type={type(thought)}")
                thought = str(thought)
            logger.info(f"Setting last_thought with length: {len(thought)} characters")
            self.last_thought = thought
            return True
        return False
        
    def find_related_memories(self, thought_query, n=3):
        """Find memories related to a specific thought query.
        
        Args:
            thought_query (str): The thought to find related memories for
            n (int, optional): Number of memories to return
            
        Returns:
            list: Related memories
        """
        logger.info(f"Finding memories related to specific thought: '{thought_query[:50]}...'")
        
        # Debug the input type and length
        if not isinstance(thought_query, str):
            logger.warning(f"Non-string thought query received: type={type(thought_query)}")
            thought_query = str(thought_query)
        logger.info(f"Memory query with length: {len(thought_query)} characters")
        
        # Extract a manageable query from potentially long thought
        query = self._extract_query_from_thought(thought_query)
        
        # Use current emotional state if available, otherwise get from emotion center
        emotional_state = self.current_emotional_state or self.emotion_center.get_state()
        return self.memory.recall(emotional_state, query=query, n=n)
        
    def compute_emotional_response(self, recent_memories=None, related_long_term_memories=None):
        """Compute emotional changes based on recent memories and related long-term memories.
        
        This centralizes emotional updates in one place rather than having them scattered
        throughout the codebase.
        
        Args:
            recent_memories (list): Recent short-term memories to analyze
            related_long_term_memories (list): Related long-term memories
            
        Returns:
            dict: Emotion name to intensity change mapping
        """
        logger.info("Computing emotional response to memories")
        
        # Initialize emotional changes dictionary
        emotional_changes = {
            'happiness': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'energy': 0.0,
            'focus': 0.0,
            'curiosity': 0.0
        }
        
        # Process recent memories if available
        if recent_memories:
            # Calculate changes based on keywords in recent memories
            for memory in recent_memories:
                memory_str = str(memory).lower()
                
                # Simple keyword-based processing
                if any(word in memory_str for word in ['happy', 'joy', 'success', 'achieve']):
                    emotional_changes['happiness'] += 0.05
                    emotional_changes['energy'] += 0.03
                
                if any(word in memory_str for word in ['sad', 'disappoint', 'fail', 'lost']):
                    emotional_changes['sadness'] += 0.05
                    emotional_changes['energy'] -= 0.03
                
                if any(word in memory_str for word in ['anger', 'frustrat', 'annoy']):
                    emotional_changes['anger'] += 0.05
                    emotional_changes['energy'] += 0.02  # Anger can increase energy
                
                if any(word in memory_str for word in ['fear', 'afraid', 'scary', 'worry']):
                    emotional_changes['fear'] += 0.05
                    emotional_changes['focus'] += 0.03  # Fear can increase focus
                
                if any(word in memory_str for word in ['surprise', 'shock', 'unexpected']):
                    emotional_changes['surprise'] += 0.08  # Surprise fades quickly
                    emotional_changes['curiosity'] += 0.04
                
                if any(word in memory_str for word in ['disgust', 'repuls', 'gross']):
                    emotional_changes['disgust'] += 0.04
                
                if any(word in memory_str for word in ['curious', 'interest', 'learn']):
                    emotional_changes['curiosity'] += 0.06
                    emotional_changes['focus'] += 0.03
                
                if any(word in memory_str for word in ['focus', 'concentrat', 'attention']):
                    emotional_changes['focus'] += 0.05
                
                if any(word in memory_str for word in ['energy', 'active', 'vigor']):
                    emotional_changes['energy'] += 0.05
        
        # Process long-term memories if available
        if related_long_term_memories:
            # Long-term memories have less immediate emotional impact
            for memory in related_long_term_memories:
                memory_str = str(memory).lower()
                
                # Similar processing but with smaller changes
                if any(word in memory_str for word in ['happy', 'joy', 'success', 'achieve']):
                    emotional_changes['happiness'] += 0.02
                
                if any(word in memory_str for word in ['sad', 'disappoint', 'fail', 'lost']):
                    emotional_changes['sadness'] += 0.02
                
                # Additional emotional reactions from long-term memories
                if 'profound' in memory_str or 'insight' in memory_str:
                    emotional_changes['focus'] += 0.03
                    emotional_changes['curiosity'] += 0.02
        
        # Adjust emotion changes to avoid extreme swings 
        # and normalize the emotional response
        for emotion in emotional_changes:
            # Cap the change to avoid overly dramatic shifts
            emotional_changes[emotion] = max(-0.1, min(0.1, emotional_changes[emotion]))
            
        logger.info(f"Computed emotional changes: {emotional_changes}")
        
        return emotional_changes
        
    def _surface_memories(self, trace_id):
        """Surface relevant memories based on current context.
        
        TODO: Improve memory surfacing algorithm to consider more factors:
        - Current context relevance
        - Emotional resonance with different weightings
        - Recency vs importance balance
        - Associative connections between memories
        """

        logger.info(f"Surfacing memories")

        # If we have a recent thought, use it as a query
        if self.last_thought:
            # Debug the last_thought type and length
            if not isinstance(self.last_thought, str):
                logger.warning(f"Non-string last_thought found: type={type(self.last_thought)}")
                self.last_thought = str(self.last_thought)
            logger.info(f"Using last_thought with length: {len(self.last_thought)} characters")
            
            logger.info(f"Recalling memories related to recent thought: '{self.last_thought[:50]}...'")
            
            # Extract key terms from the thought for more focused memory retrieval
            query = self._extract_query_from_thought(self.last_thought)
            
            # Get memories related to the extracted query (semantic search)
            thought_related_memories = self.memory.recall(query=query, n=2)
            
            # Combine both (avoiding duplicates)
            all_memories = thought_related_memories

            logger.info(f"Surfaced memories: {all_memories}")
            return all_memories
        else:
            logger.info("No recent thought to surface memories from")
            return []
    
    def _extract_query_from_thought(self, thought):
        """Extract a focused query from a potentially long thought."""
        # If thought is short enough, use it directly
        if len(thought) < 200:
            return thought[:200]
            
        # For longer thoughts, try to extract key concepts
        # Placeholder for a more sophisticated algorithm
        # This could be enhanced with NLP techniques like:
        # - Keyword extraction
        # - Named entity recognition
        # - Topic modeling
        words = thought.split()
        if len(words) > 20:
            # Use the first 10 and last 10 words as a simple approach
            query = " ".join(words[:10] + words[-10:])
            return query
        
        # Fallback to a truncated version
        return thought[:200]
        
    def _generate_random_thoughts(self, trace_id):
        """Generate random thoughts to maintain cognitive activity.
        
        TODO: Implement a more sophisticated thought generation system:
        - Base thoughts on underlying emotional state
        - Connect to recent experiences or memories
        - Create metaphorical connections between concepts
        - Implement probabilistic selection based on salience
        """
        logger.info(f"Generating random thoughts")
        topics = ["philosophy", "daily life", "fantasy", "science", "relationships"]
        return [f"Random thought about {random.choice(topics)}"]
        
    def _process_emotions(self, trace_id):
        """Process emotional reactions to current state.
        
        TODO: Improve with more nuanced emotional processing:
        - Consider emotional trends over time
        - Implement emotional associations between concepts
        - Add unconscious emotional biases
        - Consider conflicting emotions and their interactions
        """
        logger.info(f"Processing emotions")
        emotions = self.current_emotional_state or self.emotion_center.get_state()
        
        anger = emotions.get('anger', 0.0)
        happiness = emotions.get('happiness', 0.0)
        focus = emotions.get('focus', 0.0)
        
        if anger > 0.7:
            return ["I'm feeling really angry about this"]
        elif happiness > 0.7:
            return ["I'm feeling really happy right now"]
        elif focus > 0.7:
            return ["I'm in a state of deep concentration right now, with heightened mental clarity"]
        return [] 