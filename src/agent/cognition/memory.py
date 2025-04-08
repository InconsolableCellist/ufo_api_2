"""Memory management for the agent's cognitive system."""

import logging
import json
import os
import pickle
import time
from collections import deque
from datetime import datetime
import numpy as np
import faiss
from openai import OpenAI
import requests
from config import langfuse

logger = logging.getLogger('agent_simulation.cognition.memory')

# Get API configuration from environment or use defaults
API_HOST = os.getenv("API_HOST", "bestiary")
API_PORT = os.getenv("API_PORT", "5000")
API_BASE_URL = f"http://{API_HOST}:{API_PORT}/v1"
EMBEDDING_MODEL = "text-embedding-ada-002"

class Memory:
    """Manages the agent's memory system including short-term and long-term memory.
    
    The Memory class provides a sophisticated memory system with:
    - Short-term memory (recent thoughts/events)
    - Long-term memory with semantic search
    - Memory associations with emotional states
    - Memory metadata and type tracking
    - Persistence capabilities
    
    TODO: Enhance memory architecture:
    - Implement hierarchical memory structure (episodic, semantic, procedural)
    - Add forgetting mechanisms based on relevance and time
    - Implement memory consolidation during "rest" periods
    - Add metadata for memories (importance, vividness, etc.)
    - Implement associative memory networks
    """
    
    def __init__(self, embedding_dim=None, persist_path=None):
        # Short-term memory as a fixed-size deque
        self.short_term = deque(maxlen=10)  # Last 10 thoughts/events
        self.long_term = []  # Will store content strings
        self.associations = {}  # Memory-emotion associations
        
        # Memory metadata tracking
        self.memory_metadata = {}  # Store metadata for each memory
        self.thinking_cycle_count = 0  # Track thinking cycles
        self.memory_types = {
            "normal_thought": [],
            "ego_thought": [],
            "external_info": [],
            "stimuli_interpretation": [],
            "research": []  # Add specific type for research results
        }
        
        # FAISS index for semantic search
        self.embedding_dim = embedding_dim  # Will be set dynamically if None
        self.index = None  # Will be initialized after first embedding
        self.embeddings = []  # Store embeddings corresponding to long_term memories
        
        # Initialize OpenAI client for embeddings
        logger.info(f"Initializing OpenAI client with base URL: {API_BASE_URL}")
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key="not-needed"
        )
        
        # Path for persistence
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            logger.info(f"Loading memory from {persist_path}")
            self.load()
        else:
            logger.info("Starting with fresh memory (no persistence file found)")
            
    def increment_cycle(self):
        """Increment the thinking cycle counter."""
        self.thinking_cycle_count += 1
        return self.thinking_cycle_count
        
    def get_embedding(self, text):
        """Get embedding vector for text using OpenAI API.
        
        Args:
            text (str): Text to get embedding for
            
        Returns:
            numpy.ndarray: Embedding vector
            
        Raises:
            Exception: If embedding API call fails
        """
        try:
            # Truncate text to avoid "input too large" errors
            max_chars = 8000  # Conservative limit
            if len(text) > max_chars:
                logger.warning(f"Text too long for embedding ({len(text)} chars), truncating to {max_chars} chars")
                text = text[:max_chars]
                
            logger.info(f"Requesting embedding for text: '{text[:50]}...' (truncated) of length {len(text)}")
            
            # Create Langfuse span for tracking
            generation = langfuse.generation(
                name="embedding-request",
                model=EMBEDDING_MODEL,
                input=text,
                metadata={
                    "base_url": self.client.base_url,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Make API request
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text,
                "encoding_format": "float"
            }
            
            base_url = API_BASE_URL.rstrip('/')
            api_url = f"{base_url}/embeddings"
            
            start_time = time.time()
            response_raw = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30 
            )
            elapsed = time.time() - start_time
            
            if response_raw.status_code != 200:
                logger.warning(f"Embedding API returned status code {response_raw.status_code}: {response_raw.text}")
                raise Exception(f"API returned status code {response_raw.status_code}")
            
            response_data = response_raw.json()
            
            # Update Langfuse tracking
            generation.end(
                output=f"Embedding vector of dimension {len(response_data['data'][0]['embedding'])}",
                metadata={
                    "elapsed_time": elapsed,
                    "response_status": response_raw.status_code
                }
            )
            
            embedding = np.array(response_data['data'][0]['embedding'], dtype=np.float32)
            logger.info(f"Embedding vector shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
            
            # Initialize FAISS index if needed
            if self.embedding_dim is None or self.index is None:
                self.embedding_dim = embedding.shape[0]
                logger.info(f"Setting embedding dimension to {self.embedding_dim}")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}", exc_info=True)
            if 'generation' in locals():
                generation.end(
                    error=str(e),
                    metadata={"error_type": "embedding_api_error"}
                )
            raise
            
    def add(self, content, emotional_context=None, thought_type=None, source_stimuli=None, add_prefix=False):
        """Add a memory with its emotional context and optional metadata.
        
        Args:
            content (str): The content to store
            emotional_context (dict, optional): Emotional state associated with memory
            thought_type (str, optional): Type of thought ("normal_thought", "ego_thought", etc.)
            source_stimuli (dict, optional): Source stimuli that led to this thought
            add_prefix (bool, optional): Whether to add prefixes like "I remember thinking"
            
        Returns:
            int: Memory ID
        """
        # Create timestamp
        timestamp = datetime.now().isoformat()
        
        # Add appropriate prefix based on thought type if requested
        if add_prefix and thought_type:
            if thought_type == "normal_thought":
                prefixed_content = f"I remember thinking: {content}"
            elif thought_type == "ego_thought":
                prefixed_content = f"I remember reflecting: {content}"
            elif thought_type == "stimuli_interpretation":
                prefixed_content = f"I remember interpreting: {content}"
            else:
                prefixed_content = content
        else:
            prefixed_content = content
            
        logger.info(f"Adding memory: '{prefixed_content[:50]}...' (truncated)")
        
        self.short_term.append(prefixed_content)
        self.long_term.append(prefixed_content)
        
        # Store in memory types if applicable
        memory_id = len(self.long_term) - 1
        if thought_type in self.memory_types:
            self.memory_types[thought_type].append(memory_id)
            
        # Store metadata
        self.memory_metadata[prefixed_content] = {
            "id": memory_id,
            "original_content": content if prefixed_content != content else None,
            "type": thought_type,
            "timestamp": timestamp,
            "cycle": self.thinking_cycle_count,
            "emotional_context": emotional_context,
            "source_stimuli": source_stimuli,
            "last_recalled": None,
            "recall_count": 0
        }
        
        # Get embedding and add to FAISS index
        try:
            embedding = self.get_embedding(prefixed_content)
            
            # Verify dimensions match
            if self.embedding_dim != embedding.shape[0]:
                logger.warning(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.embedding_dim}")
                if len(self.embeddings) > 0:
                    logger.error("Cannot add embedding with different dimension to existing index")
                    return memory_id
                else:
                    self.embedding_dim = embedding.shape[0]
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                    logger.info(f"Updated embedding dimension to {self.embedding_dim}")
            
            self.embeddings.append(embedding)
            
            # Add to FAISS index
            faiss.normalize_L2(embedding.reshape(1, -1))  # Normalize for cosine similarity
            self.index.add(embedding.reshape(1, -1))
            
            if emotional_context:
                self.associations[prefixed_content] = emotional_context
                
            # Persist if path is set
            if self.persist_path:
                self.save()
                
        except Exception as e:
            logger.error(f"Failed to add memory due to embedding error: {e}")
            # Remove the memory since we couldn't embed it
            if prefixed_content in self.long_term:
                self.long_term.remove(prefixed_content)
            
        return memory_id
        
    def add_thought(self, content, thought_type, emotional_context=None, source_stimuli=None):
        """Add a thought with metadata and appropriate prefix (convenience method)."""
        return self.add(
            content=content, 
            emotional_context=emotional_context,
            thought_type=thought_type,
            source_stimuli=source_stimuli,
            add_prefix=True
        )
        
    def recall(self, emotional_state, query=None, n=3, memory_type=None):
        """Recall memories based on emotional state and/or query.
        
        Args:
            emotional_state (dict): Current emotional state
            query (str, optional): Query for semantic search
            n (int, optional): Number of memories to return
            memory_type (str, optional): Type of memories to recall
            
        Returns:
            list: Recalled memories
        """
        results = []
        
        if query and self.index is not None and self.embeddings:
            # Search by semantic similarity
            logger.info(f"Recalling memories with query: '{query}'")
            
            try:
                query_embedding = self.get_embedding(query)
                faiss.normalize_L2(query_embedding.reshape(1, -1))
                D, I = self.index.search(query_embedding.reshape(1, -1), min(n, len(self.embeddings)))
                
                for i in range(D.shape[1]):
                    if I[0, i] < len(self.long_term):
                        memory = self.long_term[I[0, i]]
                        results.append(memory)
                        self.update_recall_stats(memory)
            except Exception as e:
                logger.error(f"Error during semantic search: {e}", exc_info=True)
                
        # Filter by memory type if specified
        if memory_type and not results:
            type_results = self.recall_by_type(memory_type, n)
            for memory in type_results:
                self.update_recall_stats(memory)
            return type_results
            
        # If no results from semantic search, try emotional matching
        if not results and emotional_state:
            emotional_matches = []
            for memory in self.long_term:
                if memory in self.associations:
                    if self._emotional_match(self.associations[memory], emotional_state):
                        emotional_matches.append(memory)
                        self.update_recall_stats(memory)
                        
            results = emotional_matches[:n]
            
        return results
        
    def recall_by_type(self, thought_type, n=3):
        """Recall memories of a specific type."""
        if thought_type not in self.memory_types or not self.memory_types[thought_type]:
            return []
            
        memory_ids = self.memory_types[thought_type][-n:]
        memories = [self.long_term[i] for i in memory_ids]
        
        for memory in memories:
            self.update_recall_stats(memory)
            
        return memories
        
    def recall_research(self, query=None, n=3):
        """Recall research-specific memories, optionally filtered by query."""
        if "research" not in self.memory_types or not self.memory_types["research"]:
            return []
            
        research_ids = self.memory_types["research"]
        research_memories = [self.long_term[i] for i in research_ids]
        
        if not query:
            recent_memories = research_memories[-n:]
            for memory in recent_memories:
                self.update_recall_stats(memory)
            return recent_memories
            
        # If query provided, try to find matching research memories
        matching_memories = []
        for memory in research_memories:
            if query.lower() in memory.lower():
                matching_memories.append(memory)
                
        result_memories = matching_memories[-n:]
        for memory in result_memories:
            self.update_recall_stats(memory)
            
        return result_memories
        
    def recall_by_time_range(self, start_time, end_time, n=10):
        """Recall memories within a specific time range."""
        matches = []
        
        for content, metadata in self.memory_metadata.items():
            if 'timestamp' in metadata:
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                if start_time <= timestamp <= end_time:
                    matches.append(content)
                    self.update_recall_stats(content)
                    
        matches.sort(key=lambda m: self.memory_metadata.get(m, {}).get('timestamp', ''), reverse=True)
        return matches[:n]
        
    def update_recall_stats(self, content):
        """Update memory recall statistics."""
        if content in self.memory_metadata:
            self.memory_metadata[content]['last_recalled'] = datetime.now().isoformat()
            self.memory_metadata[content]['recall_count'] = self.memory_metadata[content].get('recall_count', 0) + 1
            
    def calculate_memory_importance(self, content):
        """Calculate importance score based on recency, recall frequency, and emotion."""
        if content not in self.memory_metadata:
            return 0
            
        metadata = self.memory_metadata[content]
        
        # Recency factor (more recent = more important)
        if 'timestamp' in metadata:
            time_created = datetime.fromisoformat(metadata['timestamp'])
            recency = 1.0 - min(1.0, (datetime.now() - time_created).total_seconds() / (7 * 24 * 3600))  # 1 week decay
        else:
            recency = 0.5  # Default if no timestamp
        
        # Recall frequency factor
        recall_factor = min(1.0, metadata.get('recall_count', 0) / 10)  # Cap at 10 recalls
        
        # Emotional intensity factor
        emotional_intensity = 0
        if metadata.get('emotional_context'):
            intensities = [v for k, v in metadata['emotional_context'].items() 
                          if isinstance(v, (int, float))]
            if intensities:
                emotional_intensity = sum(intensities) / len(intensities)
        
        # Calculate importance (weighted sum)
        importance = (0.4 * recency) + (0.3 * recall_factor) + (0.3 * emotional_intensity)
        return importance
        
    def get_memory_stats(self):
        """Get statistics about stored memories."""
        return {
            "total_memories": len(self.long_term),
            "by_type": {type_name: len(ids) for type_name, ids in self.memory_types.items()},
            "thinking_cycles": self.thinking_cycle_count,
            "most_recalled": sorted(
                [(content, m.get('recall_count', 0)) for content, m in self.memory_metadata.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
    def _emotional_match(self, memory_emotion, current_emotion):
        """Check if memory emotion matches current emotion."""
        # Extract primary emotion keys to check
        emotion_keys = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'energy', 'focus', 'curiosity']
        
        # If both have mood, use it as a primary match factor
        if 'mood' in memory_emotion and 'mood' in current_emotion:
            mood_similarity = 1.0 - abs(memory_emotion['mood'] - current_emotion['mood'])
            if mood_similarity > 0.7:
                logger.info(f"Mood match: {mood_similarity:.2f}")
                return True
                
        # Check for strong emotions in both memory and current state
        matched_emotions = []
        for emotion in emotion_keys:
            memory_intensity = memory_emotion.get(emotion, 0)
            current_intensity = current_emotion.get(emotion, 0)
            
            if memory_intensity > 0.6 and current_intensity > 0.6:
                matched_emotions.append(emotion)
                
        if matched_emotions:
            logger.info(f"Emotional match found on: {matched_emotions}")
            return True
            
        # Also match when we have emotional contrast
        opposing_pairs = [
            ('happiness', 'sadness'),
            ('anger', 'fear')
        ]
        
        for emotion1, emotion2 in opposing_pairs:
            if (memory_emotion.get(emotion1, 0) > 0.7 and current_emotion.get(emotion2, 0) > 0.7) or \
               (memory_emotion.get(emotion2, 0) > 0.7 and current_emotion.get(emotion1, 0) > 0.7):
                logger.info(f"Emotional contrast match between {emotion1} and {emotion2}")
                return True
        
        return any(memory_emotion.get(e, 0) > 0.5 and current_emotion.get(e, 0) > 0.5 
                  for e in ['happiness', 'sadness', 'anger', 'fear'])
        
    def save(self):
        """Save memory state to disk."""
        if not self.persist_path:
            return
            
        data = {
            'long_term': self.long_term,
            'associations': self.associations,
            'embeddings': np.vstack(self.embeddings) if self.embeddings else np.array([]),
            'memory_metadata': self.memory_metadata,
            'memory_types': self.memory_types,
            'thinking_cycle_count': self.thinking_cycle_count
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self):
        """Load memory state from disk."""
        if not os.path.exists(self.persist_path):
            return
            
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
            
        self.long_term = data.get('long_term', [])
        self.associations = data.get('associations', {})
        
        # Load memory metadata if available
        if 'memory_metadata' in data:
            self.memory_metadata = data['memory_metadata']
        else:
            # Create basic metadata for existing memories
            self.memory_metadata = {}
            for i, memory in enumerate(self.long_term):
                self.memory_metadata[memory] = {
                    "id": i,
                    "timestamp": datetime.now().isoformat(),  # Default to now
                    "cycle": 0,
                    "type": "external_info",  # Default type
                    "recall_count": 0
                }
            
        if 'memory_types' in data:
            self.memory_types = data['memory_types']
        else:
            # Create default memory types
            self.memory_types = {
                "normal_thought": [],
                "ego_thought": [],
                "external_info": list(range(len(self.long_term))),  # Assume all are external info
                "stimuli_interpretation": []
            }
            
        if 'thinking_cycle_count' in data:
            self.thinking_cycle_count = data['thinking_cycle_count']
        
        # Rebuild FAISS index if we have embeddings
        if 'embeddings' in data and len(data['embeddings']) > 0:
            embeddings = data['embeddings']
            self.embedding_dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Normalize and add to index
            normalized_embeddings = embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)
            self.index.add(normalized_embeddings)
            
            # Store embeddings
            self.embeddings = [embeddings[i] for i in range(embeddings.shape[0])] 