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
            self._surface_memories,
            self._generate_random_thoughts,
            self._process_emotions
        ]
        self.last_thought = None  # Track the most recent thought
        
    def process(self, trace_id):
        """Run all background processes.
        
        Args:
            trace_id: ID for tracing/logging purposes
            
        Returns:
            list: Generated thoughts and surfaced memories
        """
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
        
        emotional_state = self.emotion_center.get_state()
        return self.memory.recall(emotional_state, query=query, n=n)
        
    def _surface_memories(self, trace_id):
        """Surface relevant memories based on current context.
        
        TODO: Improve memory surfacing algorithm to consider more factors:
        - Current context relevance
        - Emotional resonance with different weightings
        - Recency vs importance balance
        - Associative connections between memories
        """
        emotional_state = self.emotion_center.get_state()
        
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
            thought_related_memories = self.memory.recall(emotional_state, query=query, n=2)
            
            # Also get emotionally relevant memories
            emotional_memories = self.memory.recall(emotional_state, n=1)
            
            # Combine both (avoiding duplicates)
            all_memories = thought_related_memories
            for mem in emotional_memories:
                if mem not in all_memories:
                    all_memories.append(mem)
                    
            return all_memories
        else:
            # Fallback to emotional state only if no recent thought
            return self.memory.recall(emotional_state)
    
    def _extract_query_from_thought(self, thought):
        """Extract key terms or a summary from a longer thought for memory queries.
        
        Args:
            thought (str): The thought to extract a query from
            
        Returns:
            str: Extracted query suitable for memory search
        """
        # Debug input
        logger.info(f"Extracting query from thought of length: {len(thought)} characters")
        
        # Start with a simple approach - take the first 100 characters
        if len(thought) <= 150:
            return thought
            
        # For longer thoughts, extract key sentences
        sentences = thought.split('.')
        
        # Log the number of sentences for debugging
        logger.info(f"Thought contains {len(sentences)} sentences")
        
        # Use the first 1-2 sentences as they often contain the main point
        if len(sentences) >= 2:
            extracted = sentences[0].strip() + '. ' + sentences[1].strip()
            if len(extracted) > 200:
                extracted = extracted[:200]
            logger.info(f"Extracted query from thought: '{extracted}' (length: {len(extracted)})")
            return extracted
            
        # Fallback to just the first part of the thought
        extracted = thought[:150]
        logger.info(f"Extracted query from thought: '{extracted}' (length: {len(extracted)})")
        return extracted
        
    def _generate_random_thoughts(self, trace_id):
        """Generate random thoughts to maintain cognitive activity.
        
        TODO: Implement a more sophisticated thought generation system:
        - Base thoughts on underlying emotional state
        - Connect to recent experiences or memories
        - Create metaphorical connections between concepts
        - Implement probabilistic selection based on salience
        """
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
        emotions = self.emotion_center.get_state()
        
        # Safely access emotions with defaults if they don't exist
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