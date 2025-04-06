import openai
from openai import OpenAI
import random
from collections import deque
import numpy as np
import faiss
import pickle
import os
import logging
import json
import time
import sys
import requests
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from datetime import datetime
from colorama import Fore, Back, Style, init
import importlib
import re
import telegram  
import atexit
import signal
import tools
from tools import ToolRegistry, Tool, Journal, TelegramBot 
from templates import THOUGHT_PROMPT_TEMPLATE, AGENT_SYSTEM_INSTRUCTIONS, EGO_SYSTEM_2_INSTRUCTIONS, TOOL_DOCUMENTATION_TEMPLATE
import queue
import datetime
import importlib
import re
import os
import sys
import time
import json
import queue
import random
import logging
import uuid
import traceback
import requests
import threading
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Initialize colorama
init(autoreset=True)

# Configure logging with custom formatter for colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }

    def format(self, record):
        log_message = super().format(record)
        
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        
        if "LLM response:" in log_message:
            # Highlight LLM responses in magenta
            parts = log_message.split("LLM response: '")
            if len(parts) > 1:
                response_part = parts[1].rsplit("'", 1)
                if len(response_part) > 1:
                    log_message = parts[0] + "LLM response: '" + Fore.MAGENTA + response_part[0] + Fore.RESET + "'" + response_part[1]
        
        # Highlight emotional states
        if "emotional state:" in log_message.lower():
            # Find the emotional state part and color it
            parts = log_message.split("emotional state:")
            if len(parts) > 1:
                state_part = parts[1].split("\n", 1)
                if len(state_part) > 1:
                    log_message = parts[0] + "emotional state:" + Fore.BLUE + state_part[0] + Fore.RESET + "\n" + state_part[1]
                else:
                    log_message = parts[0] + "emotional state:" + Fore.BLUE + state_part[0] + Fore.RESET
        
        return color + log_message + Style.RESET_ALL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_simulation.log')  # File handler without colors
    ]
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('agent_simulation')
logger.addHandler(console_handler)

API_HOST = "bestiary"
API_PORT = "5000"
API_BASE_URL = f"http://{API_HOST}:{API_PORT}/v1"

SUMMARY_HOST = "mlboy"
SUMMARY_PORT = "5000"
SUMMARY_BASE_URL = f"http://{SUMMARY_HOST}:{SUMMARY_PORT}/v1"

os.environ["LANGFUSE_HOST"] = "http://zen:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-c039333b-d33f-44a5-a33c-5827e783f4b2"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-3c59e794-7426-49ea-b9a1-2eae0999fadf"
langfuse = Langfuse()

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETION_MODEL = "local-model"

LLM_CONFIG = {
    "model": COMPLETION_MODEL,
    "max_tokens": 750,
    "temperature": 1.75,
}

class Emotion:
    def __init__(self, name, intensity=0.0, decay_rate=0.1, influence_factors=None):
        self.name = name
        self.intensity = intensity
        self.decay_rate = decay_rate
        self.influence_factors = influence_factors or {}
        
    def update(self, stimuli):
        # Apply stimuli effects
        for stimulus, effect in self.influence_factors.items():
            if stimulus in stimuli:
                self.intensity += effect * stimuli[stimulus]
        
        # Apply natural decay
        self.intensity *= (1 - self.decay_rate)
        self.intensity = max(0, min(1, self.intensity))  # Clamp between 0-1

class EmotionCenter:
    def __init__(self):
        # TODO: Enhance emotion system:
        # - Add more nuanced emotions (curiosity, awe, contentment, etc.)
        # - Implement emotional combinations and complex states
        # - Add emotional memory/history to track trends over time
        # - Implement more sophisticated emotional dynamics (e.g., mood contagion)
        # - Add emotion regulation mechanisms
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
        self.llm_client = None
        
    def update(self, stimuli):
        # Update all emotions
        # TODO: Implement more realistic emotion dynamics:
        # - Consider emotion interaction (e.g., fear can amplify anger)
        # - Add emotional inertia (resistance to sudden changes)
        # - Implement habituation to stimuli
        # - Add baseline personality traits that influence emotional responses
        for emotion in self.emotions.values():
            emotion.update(stimuli)
            
        # Calculate overall mood (weighted average)
        # TODO: Improve mood calculation with more factors:
        # - Consider personality baseline
        # - Add time-weighted averaging (recent emotions matter more)
        # - Implement emotional "momentum"
        positive = self.emotions['happiness'].intensity + self.emotions['surprise'].intensity * 0.5 + self.emotions['focus'].intensity * 0.3 + self.emotions['curiosity'].intensity * 0.2
        negative = self.emotions['sadness'].intensity + self.emotions['anger'].intensity + self.emotions['fear'].intensity
        self.mood = (positive - negative) / (positive + negative + 1e-6)  # Avoid division by zero
        
    def get_state(self):
        # TODO: Enhance state reporting:
        # - Add emotional trend analysis
        # - Include dominant emotion identification
        # - Add emotional complexity metrics
        # - Report emotional stability indicators
        return {name: emotion.intensity for name, emotion in self.emotions.items()}

class Memory:
    def __init__(self, embedding_dim=None, persist_path=None):
        # TODO: Enhance memory architecture:
        # - Implement hierarchical memory structure (episodic, semantic, procedural)
        # - Add forgetting mechanisms based on relevance and time
        # - Implement memory consolidation during "rest" periods
        # - Add metadata for memories (importance, vividness, etc.)
        # - Implement associative memory networks
        self.short_term = deque(maxlen=10)  # Last 10 thoughts/events
        self.long_term = []  # Will store content strings
        self.associations = {}  # Memory-emotion associations
        
        # Added memory metadata tracking
        self.memory_metadata = {}  # Store metadata for each memory
        self.thinking_cycle_count = 0  # Track thinking cycles
        self.memory_types = {
            "normal_thought": [],
            "ego_thought": [],
            "external_info": [],
            "stimuli_interpretation": [],
            "research": []  # Add specific type for research results
        }
        
        # We'll initialize the FAISS index after getting the first embedding
        self.embedding_dim = embedding_dim  # This will be set dynamically if None
        self.index = None  # Will be initialized after first embedding
        self.embeddings = []  # Store embeddings corresponding to long_term memories
        
        # OpenAI client
        # TODO: Improve embedding system:
        # - Add support for multiple embedding models
        # - Implement caching for embeddings to reduce API calls
        # - Add fallback mechanisms for embedding failures
        # - Consider implementing local embedding models as backup
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
        """Increment the thinking cycle counter"""
        self.thinking_cycle_count += 1
        return self.thinking_cycle_count
        
    def get_embedding(self, text):
        """Get embedding vector for text using OpenAI API"""
        try:
            # Truncate text to avoid "input too large" errors
            # Most embedding APIs have token limits (OpenAI's is typically 8191 tokens)
            max_chars = 8000  # Conservative limit that should work for most APIs
            if len(text) > max_chars:
                logger.warning(f"Text too long for embedding ({len(text)} chars), truncating to {max_chars} chars")
                text = text[:max_chars]
                
            logger.info(f"Requesting embedding for text: '{text[:50]}...' (truncated) of length {len(text)}")
            
            logger.info(f"Connecting to embedding API at {self.client.base_url}")
            
            start_time = time.time()
            
            generation = langfuse.generation(
                name="embedding-request",
                model=EMBEDDING_MODEL,
                input=text,
                metadata={
                    "base_url": self.client.base_url,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            headers = {
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text,
                "encoding_format": "float"
            }
            
            base_url = API_BASE_URL.rstrip('/')  # Remove trailing slash if present
            api_url = f"{base_url}/embeddings"
            logger.info(f"Sending request to: {api_url}")
            
            response_raw = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30 
            )
            
            if response_raw.status_code != 200:
                logger.warning(f"Embedding API returned status code {response_raw.status_code}: {response_raw.text}")
                raise Exception(f"API returned status code {response_raw.status_code}")
            
            response_data = response_raw.json()
            
            elapsed = time.time() - start_time
            
            generation.end(
                output=f"Embedding vector of dimension {len(response_data['data'][0]['embedding'])}",
                metadata={
                    "elapsed_time": elapsed,
                    "response_status": response_raw.status_code
                }
            )
            
            logger.info(f"Embedding API response received in {elapsed:.2f}s")
            logger.debug(f"Embedding response status: {response_raw.status_code}")
            
            embedding = np.array(response_data['data'][0]['embedding'], dtype=np.float32)
            logger.info(f"Embedding vector shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
            
            # If this is our first embedding, set the dimension and initialize FAISS
            if self.embedding_dim is None or self.index is None:
                self.embedding_dim = embedding.shape[0]
                logger.info(f"Setting embedding dimension to {self.embedding_dim} based on first API response")
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}", exc_info=True)
            if 'generation' in locals():
                generation.end(
                    error=str(e),
                    metadata={"error_type": "embedding_api_error"}
                )
            raise Exception(f"Failed to get embedding: {str(e)}")
            
    def add(self, content, emotional_context=None, thought_type=None, source_stimuli=None, add_prefix=False):
        """Add a memory with its emotional context and optional metadata
        
        Args:
            content (str): The content to store
            emotional_context (dict, optional): Emotional state associated with memory
            thought_type (str, optional): Type of thought ("normal_thought", "ego_thought", etc.)
            source_stimuli (dict, optional): Source stimuli that led to this thought
            add_prefix (bool, optional): Whether to add prefixes like "I remember thinking"
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
            
        logger.info(f"Adding memory: '{prefixed_content[:50]}...' (truncated) of length {len(prefixed_content)}")
        
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
        
        logger.info("Getting embedding for new memory")
        try:
            embedding = self.get_embedding(prefixed_content)
            
            # Verify dimensions match what we expect
            if self.embedding_dim != embedding.shape[0]:
                logger.warning(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.embedding_dim}")
                # If we already have an index but dimensions don't match, we need to rebuild
                if len(self.embeddings) > 0:
                    logger.error("Cannot add embedding with different dimension to existing index")
                    # Return without adding this memory to avoid crashing
                    return memory_id
                else:
                    # If this is our first embedding, just update the dimension
                    self.embedding_dim = embedding.shape[0]
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                    logger.info(f"Updated embedding dimension to {self.embedding_dim}")
            
            self.embeddings.append(embedding)
            
            # Add to FAISS index
            logger.info("Adding embedding to FAISS index")
            faiss.normalize_L2(embedding.reshape(1, -1))  # Normalize for cosine similarity
            self.index.add(embedding.reshape(1, -1))
            
            if emotional_context:
                logger.info(f"Associating memory with emotional context: {emotional_context}")
                self.associations[prefixed_content] = emotional_context
                
            # Optionally persist after updates
            if self.persist_path:
                logger.info(f"Persisting memory to {self.persist_path}")
                self.save()
                
        except Exception as e:
            logger.error(f"Failed to add memory due to embedding error: {e}")
            # Remove the memory from long_term since we couldn't embed it
            if prefixed_content in self.long_term:
                self.long_term.remove(prefixed_content)
            # Don't persist since we didn't successfully add the memory
            
        return memory_id
    
    def add_thought(self, content, thought_type, emotional_context=None, source_stimuli=None):
        """Add a thought with metadata and appropriate prefix (convenience method)"""
        return self.add(
            content=content, 
            emotional_context=emotional_context,
            thought_type=thought_type,
            source_stimuli=source_stimuli,
            add_prefix=True
        )
        
    def recall(self, emotional_state, query=None, n=3, memory_type=None):
        """Recall memories based on emotional state and/or query"""
        # Use the original recall functionality but update recall statistics
        results = []
        
        if query and self.index is not None and self.embeddings:
            # Search by semantic similarity
            logger.info(f"Recalling memories with query: '{query}'")
            
            # Get query embedding
            try:
                query_embedding = self.get_embedding(query)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(query_embedding.reshape(1, -1))
                
                # Search
                D, I = self.index.search(query_embedding.reshape(1, -1), min(n, len(self.embeddings)))
                
                # Get actual memories
                for i in range(D.shape[1]):
                    if I[0, i] < len(self.long_term):
                        memory = self.long_term[I[0, i]]
                        results.append(memory)
                        # Update recall statistics
                        self.update_recall_stats(memory)
            except Exception as e:
                logger.error(f"Error during semantic search: {e}", exc_info=True)
                # Fall back to emotional matching if semantic search fails
                
        # Filter by memory type if specified
        if memory_type and not results:
            type_results = self.recall_by_type(memory_type, n)
            # Update recall statistics
            for memory in type_results:
                self.update_recall_stats(memory)
            return type_results
            
        # If we didn't get results from semantic search, try emotional matching
        if not results and emotional_state:
            # Match based on emotional similarity
            logger.info("Recalling memories based on emotional state")
            
            emotional_matches = []
            for memory in self.long_term:
                if memory in self.associations:
                    if self._emotional_match(self.associations[memory], emotional_state):
                        emotional_matches.append(memory)
                        # Update recall statistics
                        self.update_recall_stats(memory)
                        
            # Sort by emotional similarity (could be enhanced)
            results = emotional_matches[:n]
            
        return results
        
    def recall_by_type(self, thought_type, n=3):
        """Recall memories of a specific type"""
        if thought_type not in self.memory_types or not self.memory_types[thought_type]:
            return []
            
        # Get memory IDs of this type
        memory_ids = self.memory_types[thought_type][-n:]
        memories = [self.long_term[i] for i in memory_ids]
        
        # Update recall statistics
        for memory in memories:
            self.update_recall_stats(memory)
            
        return memories
        
    def recall_research(self, query=None, n=3):
        """Recall research-specific memories, optionally filtered by query"""
        # Get research memory IDs
        if "research" not in self.memory_types or not self.memory_types["research"]:
            return []
            
        research_ids = self.memory_types["research"]
        research_memories = [self.long_term[i] for i in research_ids]
        
        # If no query, return most recent research memories
        if not query:
            recent_memories = research_memories[-n:]
            
            # Update recall statistics
            for memory in recent_memories:
                self.update_recall_stats(memory)
                
            return recent_memories
            
        # If query provided, try to find matching research memories
        matching_memories = []
        for memory in research_memories:
            # Check if memory contains query (case-insensitive)
            if query.lower() in memory.lower():
                matching_memories.append(memory)
                
        # Return top n matching memories
        result_memories = matching_memories[-n:]
        
        # Update recall statistics
        for memory in result_memories:
            self.update_recall_stats(memory)
            
        return result_memories
        
    def recall_by_time_range(self, start_time, end_time, n=10):
        """Recall memories within a specific time range"""
        matches = []
        
        for content, metadata in self.memory_metadata.items():
            if 'timestamp' in metadata:
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                if start_time <= timestamp <= end_time:
                    matches.append(content)
                    # Update recall statistics
                    self.update_recall_stats(content)
                    
        # Sort by time (most recent first) and limit to n results
        matches.sort(key=lambda m: self.memory_metadata.get(m, {}).get('timestamp', ''), reverse=True)
        return matches[:n]
        
    def update_recall_stats(self, content):
        """Update memory recall statistics"""
        if content in self.memory_metadata:
            self.memory_metadata[content]['last_recalled'] = datetime.now().isoformat()
            self.memory_metadata[content]['recall_count'] = self.memory_metadata[content].get('recall_count', 0) + 1
            
    def calculate_memory_importance(self, content):
        """Calculate importance score based on recency, recall frequency, and emotion"""
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
            # Calculate average emotional intensity
            intensities = [v for k, v in metadata['emotional_context'].items() 
                          if isinstance(v, (int, float))]
            if intensities:
                emotional_intensity = sum(intensities) / len(intensities)
        
        # Calculate importance (weighted sum)
        importance = (0.4 * recency) + (0.3 * recall_factor) + (0.3 * emotional_intensity)
        return importance
        
    def get_memory_stats(self):
        """Get statistics about stored memories"""
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
        """Check if memory emotion matches current emotion"""
        # TODO: Enhance emotional matching with:
        # - Emotion similarity metrics that consider all emotions, not just a few
        # - Weighting system based on intensity and relevance
        # - Consider emotional trajectories/trends
        # - Add "emotional opposites" matching for contrast recall
        # - Implement emotional context modeling
        
        # Extract primary emotion keys to check
        emotion_keys = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'energy', 'focus', 'curiosity']
        
        # If both have mood, use it as a primary match factor
        if 'mood' in memory_emotion and 'mood' in current_emotion:
            # Mood match is a continuous value between 0 (no match) and 1 (perfect match)
            mood_similarity = 1.0 - abs(memory_emotion['mood'] - current_emotion['mood'])
            
            # If moods are highly similar (within 30% range), it's a match
            if mood_similarity > 0.7:
                logger.info(f"Mood match: {mood_similarity:.2f} - Memory mood: {memory_emotion['mood']:.2f}, Current mood: {current_emotion['mood']:.2f}")
                return True
                
        # Check for strong emotions in both memory and current state
        # A match exists if the same emotion is strong (>0.6) in both states
        matched_emotions = []
        for emotion in emotion_keys:
            memory_intensity = memory_emotion.get(emotion, 0)
            current_intensity = current_emotion.get(emotion, 0)
            
            # Match if both have the emotion with significant intensity
            if memory_intensity > 0.6 and current_intensity > 0.6:
                matched_emotions.append(emotion)
                
        if matched_emotions:
            logger.info(f"Emotional match found on: {matched_emotions}")
            return True
            
        # Also match when we have emotional contrast (strong opposite emotions)
        # This is useful for finding memories with emotional contrast
        opposing_pairs = [
            ('happiness', 'sadness'),
            ('anger', 'fear')
        ]
        
        for emotion1, emotion2 in opposing_pairs:
            # Check if one emotion is strong in memory and its opposite is strong in current
            if (memory_emotion.get(emotion1, 0) > 0.7 and current_emotion.get(emotion2, 0) > 0.7) or \
               (memory_emotion.get(emotion2, 0) > 0.7 and current_emotion.get(emotion1, 0) > 0.7):
                logger.info(f"Emotional contrast match between {emotion1} and {emotion2}")
                return True
        
        # Fallback to simple matching (legacy code)
        return any(memory_emotion.get(e, 0) > 0.5 and current_emotion.get(e, 0) > 0.5 
                  for e in ['happiness', 'sadness', 'anger', 'fear'])
        
    def save(self):
        """Save memory state to disk"""
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
        """Load memory state from disk"""
        if not os.path.exists(self.persist_path):
            return
            
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
            
        self.long_term = data.get('long_term', [])
        self.associations = data.get('associations', {})
        
        # Load memory metadata if available (for backward compatibility)
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
            # Create default memory types for backward compatibility
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

class Subconscious:
    def __init__(self, memory, emotion_center):
        self.memory = memory
        self.emotion_center = emotion_center
        # TODO: Add more sophisticated background processes for better subconscious processing
        # TODO: Consider adding dream-like processing during idle periods
        # TODO: Implement pattern recognition for underlying connections between memories
        self.background_processes = [
            self._surface_memories,
            self._generate_random_thoughts,
            self._process_emotions
        ]
        self.last_thought = None  # Track the most recent thought
        
    def process(self, trace_id):
        thoughts = []
        for process in self.background_processes:
            thoughts.extend(process(trace_id))
        return thoughts
        
    def set_focus_thought(self, thought):
        """Explicitly set a thought for the subconscious to focus on"""
        if thought:
            logger.info(f"Explicitly setting subconscious focus thought: '{thought[:50]}...'")
            # Debug the input type and length
            if not isinstance(thought, str):
                logger.warning(f"Non-string thought received: type={type(thought)}")
                # Convert to string if not already
                thought = str(thought)
            logger.info(f"Setting last_thought with length: {len(thought)} characters")
            self.last_thought = thought
            return True
        return False
        
    def find_related_memories(self, thought_query, n=3):
        """Find memories related to a specific thought query"""
        logger.info(f"Finding memories related to specific thought: '{thought_query[:50]}...'")
        
        # Debug the input type and length
        if not isinstance(thought_query, str):
            logger.warning(f"Non-string thought query received: type={type(thought_query)}")
            # Convert to string if not already
            thought_query = str(thought_query)
        logger.info(f"Memory query with length: {len(thought_query)} characters")
        
        # Extract a manageable query from potentially long thought
        query = self._extract_query_from_thought(thought_query)
        
        emotional_state = self.emotion_center.get_state()
        return self.memory.recall(emotional_state, query=query, n=n)
        
    def _surface_memories(self, trace_id):
        # TODO: Improve memory surfacing algorithm to consider more factors:
        # - Current context relevance
        # - Emotional resonance with different weightings
        # - Recency vs importance balance
        # - Associative connections between memories
        emotional_state = self.emotion_center.get_state()
        
        # If we have a recent thought, use it as a query to find related memories
        if self.last_thought:
            # Debug the last_thought type and length
            if not isinstance(self.last_thought, str):
                logger.warning(f"Non-string last_thought found: type={type(self.last_thought)}")
                # Convert to string if not already
                self.last_thought = str(self.last_thought)
            logger.info(f"Using last_thought with length: {len(self.last_thought)} characters")
            
            logger.info(f"Recalling memories related to recent thought: '{self.last_thought[:50]}...'")
            
            # Extract key terms from the thought for more focused memory retrieval
            # This prevents sending overly large texts to the embedding API
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
        """Extract key terms or a summary from a longer thought for memory queries"""
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
        # Simple random thought generation
        # TODO: Implement a more sophisticated thought generation system:
        # - Base thoughts on underlying emotional state
        # - Connect to recent experiences or memories
        # - Create metaphorical connections between concepts
        # - Implement probabilistic selection based on salience
        topics = ["philosophy", "daily life", "fantasy", "science", "relationships"]
        return [f"Random thought about {random.choice(topics)}"]
        
    def _process_emotions(self, trace_id):
        # Emotional reactions to current state
        # TODO: Improve with more nuanced emotional processing:
        # - Consider emotional trends over time
        # - Implement emotional associations between concepts
        # - Add unconscious emotional biases
        # - Consider conflicting emotions and their interactions
        emotions = self.emotion_center.get_state()
        if emotions['anger'] > 0.7:
            return ["I'm feeling really angry about this"]
        elif emotions['happiness'] > 0.7:
            return ["I'm feeling really happy right now"]
        elif emotions['focus'] > 0.7:
            return ["I'm in a state of deep concentration right now, with heightened mental clarity"]
        return []
        
    # TODO: Implement new subconscious processes:
    # - _form_associations(): Create associations between related concepts/memories
    # - _process_unconscious_biases(): Handle biases that affect perception
    # - _dream_processing(): Process memories during idle/rest periods
    # - _identify_patterns(): Recognize patterns across experiences

class Conscious:
    def __init__(self, memory, emotion_center, subconscious, llm):
        self.memory = memory 
        self.emotion_center = emotion_center
        self.subconscious = subconscious
        self.llm = llm
        self.current_focus = None
        self.ego_thoughts = ""  # Store ego thoughts between cycles
        
    def think(self, stimuli, subconscious_thoughts, trace_id):
        # Prepare context for LLM
        # TODO: Improve context preparation:
        # - Implement prioritization of memories based on relevance
        # - Add internal feedback loops for self-reflection
        # - Include previous thinking steps to maintain coherence
        # - Add structured reasoning patterns for different types of problems
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
            # This will be used for retrieving related memories in the next cycle
            # Debug - check what we're setting
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
        """Extract emotional implications from a thought using simple keyword matching"""
        # TODO: Improve emotional implication extraction:
        # - Use ML-based sentiment analysis instead of simple keyword matching
        # - Consider sentence structure and modifiers (e.g., negations)
        # - Implement emotional context understanding
        # - Add support for complex, mixed emotions
        # - Consider emotional intensity indicators
        implications = {}
        
        # Simple keyword matching
        if "happy" in thought.lower() or "joy" in thought.lower():
            implications["happiness"] = 0.1
        if "sad" in thought.lower() or "depress" in thought.lower():
            implications["sadness"] = 0.1
        if "angry" in thought.lower() or "frustrat" in thought.lower():
            implications["anger"] = 0.1
        # Add more emotion keywords
        
        return implications
        
    def decide_action(self, thoughts):
        # Simple decision making - in reality would be more complex
        # TODO: Implement more sophisticated decision making:
        # - Add goal-oriented reasoning
        # - Implement risk assessment
        # - Consider emotional impact of potential actions
        # - Add action prioritization based on urgency and importance
        # - Implement planning capabilities for multi-step actions
        if "angry" in thoughts.lower():
            return "Take deep breaths to calm down"
        return "Continue thinking"

class Task:
    def __init__(self, description, priority=0.5, status="pending"):
        self.description = description
        self.priority = priority
        self.status = status
        self.subtasks = []
        
    def add_subtask(self, subtask):
        self.subtasks.append(subtask)

class MotivationCenter:
    def __init__(self):
        self.tasks = []
        self.core_motivations = [
            "Understand myself",
            "Grow emotionally",
            "Explore ideas"
        ]
        
    def update_tasks(self, emotional_state):
        # Adjust task priorities based on emotional state
        for task in self.tasks:
            if "stress" in emotional_state:
                task.priority *= 0.8  # Lower priority when stressed
                
    def get_current_task(self):
        if not self.tasks:
            return None
        return max(self.tasks, key=lambda t: t.priority)

class Mind:
    def __init__(self, llm, memory_path=None, emotion_path=None):
        self.memory = Memory(persist_path=memory_path)
        self.emotion_center = EmotionCenter()
        self.emotion_center.llm_client = llm
        self.subconscious = Subconscious(self.memory, self.emotion_center)
        self.conscious = Conscious(self.memory, self.emotion_center, self.subconscious, llm)
        self.motivation_center = MotivationCenter()
        
        # Load emotional state if path is provided
        if emotion_path and os.path.exists(emotion_path):
            self.load_emotional_state(emotion_path)
    
    def save_emotional_state(self, path):
        """Save the current emotional state to disk"""
        try:
            emotional_data = {
                'emotions': {name: emotion.intensity for name, emotion in self.emotion_center.emotions.items()},
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
        """Load emotional state from disk"""
        try:
            with open(path, 'rb') as f:
                emotional_data = pickle.load(f)
                
            # Update emotion intensities
            for name, intensity in emotional_data.get('emotions', {}).items():
                if name in self.emotion_center.emotions:
                    self.emotion_center.emotions[name].intensity = intensity
                    
            # Update mood
            if 'mood' in emotional_data:
                self.emotion_center.mood = emotional_data['mood']
                
            logger.info(f"Emotional state loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading emotional state: {e}", exc_info=True)
            return False

    def process_step(self, stimuli):
        # Create a trace for the entire thinking cycle
        # TODO: Enhance cognitive cycle:
        # - Implement a more sophisticated cognitive architecture (e.g., LIDA, ACT-R inspired)
        # - Add attention filtering for stimuli processing
        # - Implement better integration between conscious and subconscious processes
        # - Add cognitive biases and heuristics
        # - Implement metacognition capabilities
        trace = langfuse.trace(
            name="cognitive-cycle",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "stimuli": json.dumps(stimuli) if stimuli else "{}",
                "has_ego": bool(self.conscious.ego_thoughts)
            }
        )
        
        try:
            # Update emotional state based on stimuli
            self.emotion_center.update(stimuli)
            
            # Run subconscious processes
            subconscious_span = trace.span(name="subconscious-processing")
            subconscious_thoughts = self.subconscious.process(trace_id=trace.id)
            subconscious_span.end()
            
            # Conscious thinking
            # TODO: Improve conscious processing:
            # - Add structured reasoning patterns for different types of problems
            # - Implement multiple thinking strategies (creative, analytical, etc.)
            # - Add self-monitoring of thought quality
            # - Implement cognitive resource management (attention, focus, etc.)
            conscious_span = trace.span(name="conscious-thinking")
            conscious_thought = self.conscious.think(stimuli, subconscious_thoughts, trace_id=trace.id)
            conscious_span.end()
            
            # Decision making
            # TODO: Enhance decision making:
            # - Implement more sophisticated decision frameworks
            # - Add risk assessment capabilities
            # - Consider emotional impact on decisions
            # - Add planning and projection capabilities
            # - Implement goal prioritization mechanisms
            action_span = trace.span(name="action-decision")
            action = self.conscious.decide_action(conscious_thought)
            action_span.end()
            
            result = {
                "emotional_state": self.emotion_center.get_state(),
                "subconscious_thoughts": subconscious_thoughts,
                "conscious_thought": conscious_thought,
                "action": action,
                "ego_thoughts": self.conscious.ego_thoughts
            }
            
            # Update the trace with the result
            trace.update(
                output=json.dumps(result),
                metadata={
                    "success": True,
                    "has_ego_thoughts": bool(self.conscious.ego_thoughts)
                }
            )
            
            return result
            
        except Exception as e:
            # Update the trace with the error
            trace.update(
                error=str(e),
                metadata={"success": False}
            )
            logger.error(f"Error in cognitive processing: {e}", exc_info=True)
            return {
                "emotional_state": self.emotion_center.get_state(),
                "subconscious_thoughts": "Error in processing",
                "conscious_thought": f"Error occurred: {str(e)}",
                "action": "None - system error"
            }

class Agent:
    def __init__(self, llm, memory_path=None, emotion_path=None, tool_registry_path="tool_registry_state.pkl", telegram_token=None, telegram_chat_id=None, journal_path="agent_journal.txt"):
        self.llm = llm
        self.llm.attach_to_agent(self)  # Connect the LLM to the agent
        self.mind = Mind(self.llm, memory_path, emotion_path)
        self.physical_state = {
            "energy": 0.8,
            "health": 1.0
        }
        
        # Save paths for shutdown
        self.memory_path = memory_path
        self.emotion_path = emotion_path
        self.tool_registry_path = tool_registry_path
        
        # Initialize thought summary manager
        self.thought_summary_manager = ThoughtSummaryManager()
        
        # Initialize journal and telegram bot
        self.journal = Journal(journal_path)
        self.telegram_bot = TelegramBot(telegram_token, telegram_chat_id)
        
        # Register agent-specific tools
        self._register_agent_tools()
        
        # Load tool registry state if available
        if tool_registry_path:
            self.llm.tool_registry.load_state(tool_registry_path)
        
    def update_physical_state(self):
        # Physical state affects emotions and vice versa
        emotions = self.mind.emotion_center.get_state()
        self.physical_state["energy"] -= 0.01  # Base energy drain
        self.physical_state["energy"] += emotions["energy"] * 0.05
        self.physical_state["energy"] = max(0, min(1, self.physical_state["energy"]))
        
        # If very low energy, increase desire to rest
        if self.physical_state["energy"] < 0.2:
            self.mind.emotion_center.emotions["energy"].intensity += 0.1

    def invoke_tool(self, tool_name, **params):
        """Allow the agent to invoke tools that can affect its state"""
        if tool_name == "adjust_emotion":
            emotion_name = params.get("emotion")
            intensity_change = params.get("change", 0)
            
            if emotion_name in self.mind.emotion_center.emotions:
                emotion = self.mind.emotion_center.emotions[emotion_name]
                emotion.intensity += intensity_change
                emotion.intensity = max(0, min(1, emotion.intensity))
                return f"Adjusted {emotion_name} by {intensity_change}"
            return f"Unknown emotion: {emotion_name}"
        
        elif tool_name == "rest":
            self.physical_state["energy"] += params.get("amount", 0.1)
            self.physical_state["energy"] = min(1.0, self.physical_state["energy"])
            return "Rested and recovered some energy"
        
        # Add more tools as needed
        
        return f"Unknown tool: {tool_name}"

    def _register_agent_tools(self):
        """Register tools specific to this agent"""
        # Tool to adjust emotions
        self.llm.tool_registry.register_tool(Tool(
            name="adjust_emotion",
            description="Adjust the intensity of an emotion",
            function=lambda emotion, change: self._adjust_emotion(emotion, change),
            usage_example="[TOOL: adjust_emotion(emotion:happiness, change:0.1)]"
        ))
        
        # Tool to rest and recover energy
        self.llm.tool_registry.register_tool(Tool(
            name="rest",
            description="Rest to recover energy",
            function=lambda amount=0.1: self._rest(amount),
            usage_example="[TOOL: rest(amount:0.1)]"
        ))
        
        # Tool to recall memories
        self.llm.tool_registry.register_tool(Tool(
            name="recall_memories",
            description="Recall memories related to a query or emotional state",
            function=lambda query=None, count=3: self._recall_memories(query, count),
            usage_example="[TOOL: recall_memories(query:happiness, count:3)]"
        ))
        
        # Tool to find memories related to a specific thought
        self.llm.tool_registry.register_tool(Tool(
            name="find_related_memories",
            description="Find memories related to a specific thought or concept",
            function=lambda thought=None, count=3: self._find_related_memories(thought, count),
            usage_example="[TOOL: find_related_memories(thought:artificial intelligence, count:3)]"
        ))
        
        # Tool to set subconscious focus
        self.llm.tool_registry.register_tool(Tool(
            name="set_subconscious_focus",
            description="Set a thought for the subconscious to focus on when finding related memories",
            function=lambda thought: self._set_subconscious_focus(thought),
            usage_example="[TOOL: set_subconscious_focus(thought:I want to understand more about machine learning)]"
        ))
        
        # Tool to get current emotional state
        self.llm.tool_registry.register_tool(Tool(
            name="get_emotional_state",
            description="Get a natural description of the current emotional state",
            function=lambda: self._get_emotional_state(),
            usage_example="[TOOL: get_emotional_state()]"
        ))
        
        self.llm.tool_registry.register_tool(Tool(
            name="send_telegram",
            description="Send a message",
            function=lambda message: self._send_telegram_message(message),
            usage_example="[TOOL: send_telegram(message:This is an urgent message)]"
        ))
        
        # Tool to read pending Telegram messages
        self.llm.tool_registry.register_tool(Tool(
            name="receive_telegram",
            description="Read pending messages",
            function=lambda: self._receive_telegram_messages(),
            usage_example="[TOOL: receive_telegram()]"
        ))
        
        self.llm.tool_registry.register_tool(Tool(
            name="write_journal",
            description="Write an entry to the agent's journal for future reference",
            function=lambda entry=None, text=None, message=None, value=None: self._write_journal_entry(entry or text or message or value),
            usage_example="[TOOL: write_journal(entry:Today I learned something interesting)] or [TOOL: write_journal(entry=\"My journal entry with quotes\")] or [TOOL: write_journal(text:Another format example)] (for the love of God why can't you consistently call this properly?)"
        ))
        
        self.llm.tool_registry.register_tool(Tool(
            name="report_bug",
            description="Report a bug to the agent's bug tracking system",
            function=lambda report: self._report_bug(report),
            usage_example="[TOOL: report_bug(report:The agent is experiencing a critical issue with memory recall)]"
        ))

        # Tool to find memories related to a specific thought
        self.llm.tool_registry.register_tool(Tool(
            name="find_related_memories",
            description="Find memories related to a specific thought or concept",
            function=lambda thought=None, count=3: self._find_related_memories(thought, count),
            usage_example="[TOOL: find_related_memories(thought:artificial intelligence, count:3)]"
        ))
        
        # Tool to recall research memories
        self.llm.tool_registry.register_tool(Tool(
            name="recall_research_memories",
            description="Recall memories specifically from research, optionally filtered by query",
            function=lambda query=None, count=3: self._recall_research_memories(query, count),
            usage_example="[TOOL: recall_research_memories(query:quantum computing, count:3)]"
        ))
        
    def _receive_telegram_messages(self):
        """Read unread messages from Telegram and mark them as read"""
        if not self.telegram_bot.token:
            return {
                "success": False,
                "output": "Telegram bot not configured. Cannot receive messages."
            }
        
        try:
            # Get unread messages
            unread_messages = self.telegram_bot.get_unread_messages()
            
            if not unread_messages:
                return {
                    "success": True,
                    "output": "No new messages."
                }
            
            # Format messages for display
            formatted_messages = []
            for msg in unread_messages:
                timestamp = msg.get("timestamp", "Unknown time")
                sender = msg.get("from", "Unknown sender")
                text = msg.get("text", "(No text)")
                
                formatted_messages.append(f"[{timestamp}] {sender}: {text}")
            
            # Mark messages as read
            self.telegram_bot.mark_messages_as_read()
            
            return {
                "success": True,
                "output": "New messages:\n\n" + "\n\n".join(formatted_messages)
            }
        except Exception as e:
            logger.error(f"Error receiving Telegram messages: {e}", exc_info=True)
            return {
                "success": False,
                "output": f"Error receiving messages: {str(e)}"
            }

    def _adjust_emotion(self, emotion, change):
        """Adjust an emotion's intensity"""
        if emotion in self.mind.emotion_center.emotions:
            emotion_obj = self.mind.emotion_center.emotions[emotion]
            emotion_obj.intensity += float(change)
            emotion_obj.intensity = max(0, min(1, emotion_obj.intensity))
            return f"Adjusted {emotion} by {change}, new intensity: {emotion_obj.intensity:.2f}"
        return f"Unknown emotion: {emotion}"
        
    def _rest(self, amount):
        """Rest to recover energy"""
        amount = float(amount)
        self.physical_state["energy"] += amount
        self.physical_state["energy"] = min(1.0, self.physical_state["energy"])
        return f"Rested and recovered {amount:.2f} energy. Current energy: {self.physical_state['energy']:.2f}"
        
    def _recall_memories(self, query, count):
        """Recall memories based on query or emotional state"""
        count = int(count)
        
        # If a query is provided, use semantic search
        if query:
            logger.info(f"Recalling memories with direct query: {query}")
            memories = self.mind.memory.recall(
                emotional_state=self.mind.emotion_center.get_state(),
                query=query,
                n=count
            )
        else:
            # If no query, fall back to emotional state only
            logger.info("Recalling memories based on emotional state only")
            memories = self.mind.memory.recall(
                emotional_state=self.mind.emotion_center.get_state(),
                n=count
            )
            
        return memories
        
    def _find_related_memories(self, thought, count):
        """Find memories related to a specific thought using the subconscious"""
        if not thought:
            return "A thought query must be provided"
            
        # Handle potentially very long inputs
        if len(thought) > 500:
            logger.info(f"Truncating very long thought query from {len(thought)} chars to 500 chars")
            thought = thought[:500] + "..."
            
        count = int(count)
        logger.info(f"Finding memories related to thought: '{thought[:50]}...'")
        
        # Use the subconscious to find related memories
        memories = self.mind.subconscious.find_related_memories(thought, count)
        
        if not memories:
            return "No related memories found"
            
        return memories
        
    def _set_subconscious_focus(self, thought):
        """Set a thought for the subconscious to focus on"""
        if not thought:
            return "A thought must be provided to focus on"
            
        success = self.mind.subconscious.set_focus_thought(thought)
        
        if success:
            return f"Subconscious now focusing on: '{thought[:50]}...'"
        else:
            return "Failed to set subconscious focus"

    def _send_telegram_message(self, message):
        """Send a message via Telegram"""
        if not self.telegram_bot.token:
            return "Telegram bot not configured. Message not sent."
        
        success = self.telegram_bot.send_message(message)
        if success:
            return f"Message sent successfully: '{message}'"
        else:
            return "Failed to send message. Check logs for details."

    def _write_journal_entry(self, entry=None, value=None):
        """Write an entry to the journal"""
        try:
            # Use entry or value parameter, whichever is provided
            content = entry if entry is not None else value
            
            # Handle various input formats that might be passed in by the LLM
            if content is None:
                content = "Empty journal entry"
                
            # Handle cases where entry is passed with quotes
            if isinstance(content, str):
                # Remove surrounding quotes if they exist
                if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                    content = content[1:-1]
            
            # Convert non-string entries to string
            if not isinstance(content, str):
                content = str(content)
                
            # Log the entry type and content for debugging
            logger.info(f"Writing journal entry of type {type(content)}: '{content[:50]}...'")
            
            # Ensure we have an entry
            if not content.strip():
                content = "Blank journal entry recorded at this timestamp"
                
            # Write the entry
            success = self.journal.write_entry(content)
            
            if success:
                return f"Journal entry recorded: '{content[:50]}...'" if len(content) > 50 else f"Journal entry recorded: '{content}'"
            else:
                # If writing failed, try once more with a simplified entry
                fallback_entry = f"Journal entry (simplified): {content[:100]}"
                fallback_success = self.journal.write_entry(fallback_entry)
                
                if fallback_success:
                    return f"Journal entry recorded (simplified format): '{fallback_entry[:50]}...'"
                else:
                    return "Failed to write journal entry. Check logs for details."
                    
        except Exception as e:
            # Catch any exceptions to prevent failure
            logger.error(f"Error in _write_journal_entry: {e}", exc_info=True)
            
            # Try one last time with an error note
            try:
                error_entry = f"Journal entry attempted but encountered error: {str(e)[:100]}"
                self.journal.write_entry(error_entry)
                return "Journal entry recorded with error note."
            except:
                return "Journal entry attempted but failed. Will keep trying."

    def _get_emotional_state(self):
        """Get a natural description of the current emotional state using the LLM"""
        if not hasattr(self, 'mind'):
            return "Unable to get emotional state: agent mind not properly initialized"
            
        # Get the raw emotion values
        emotion_values = self.mind.emotion_center.get_state()
        
        # Prepare context for the LLM
        prompt = f"""
        Current emotional intensities:
        {json.dumps(emotion_values, indent=2)}
        
        Overall mood: {self.mind.emotion_center.mood:.2f} (-1 to 1 scale)
        
        Based on these emotional intensities and overall mood, provide a brief, natural description
        of the emotional state from the agent's perspective. Focus on the dominant emotions
        and their interplay. Keep the description to 2-3 sentences.
        
        Format your response as a direct first-person statement of emotional awareness.
        """
        
        system_message = "You are an AI agent's emotional awareness. Describe the emotional state naturally and introspectively."
        
        try:
            # Use the agent's LLM interface to generate the description
            description = self.llm._generate_completion(prompt, system_message)
            return description
        except Exception as e:
            logger.error(f"Error generating emotional state description: {e}", exc_info=True)
            return f"Raw emotional state: {json.dumps(emotion_values, indent=2)}"

    def _report_bug(self, report):
        """Report a bug to the agent's bug tracking system"""
        try:
            # Ensure the report is a string
            if not isinstance(report, str):
                report = str(report)
            
            # Write the report to the bug tracking file
            with open("bug_reports.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()}: {report}\n")
            
            return f"Bug reported successfully: '{report[:50]}...'" if len(report) > 50 else f"Bug reported successfully: '{report}'"
        except Exception as e:
            logger.error(f"Error reporting bug: {e}", exc_info=True)
            return f"Failed to report bug: {str(e)}"

    def shutdown(self):
        """Properly shutdown the agent, saving all states"""
        logger.info("Agent shutdown initiated")
        
        # Save memory
        if self.memory_path:
            self.mind.memory.save()
            logger.info(f"Memory saved to {self.memory_path}")
            
        # Save emotional state
        if self.emotion_path:
            self.mind.save_emotional_state(self.emotion_path)
            
        # Save tool registry state
        if self.tool_registry_path:
            self.llm.tool_registry.save_state(self.tool_registry_path)
            logger.info(f"Tool registry state saved to {self.tool_registry_path}")
            
        # Stop thought summarization process
        if hasattr(self, 'thought_summary_manager'):
            self.thought_summary_manager.stop_summarization()
            logger.info("Thought summarization process stopped")
            
        # Write final journal entry
        self.journal.write_entry("Agent shutdown completed. Goodbye for now.")
        
        logger.info("Agent shutdown completed")
        return True

class SimulationController:
    def __init__(self, agent, time_step=1.0):
        self.agent = agent
        self.time_step = time_step  # In simulation time units
        self.current_time = 0
        self.stimuli_sources = []
        
    def add_stimuli_source(self, source):
        self.stimuli_sources.append(source)
        
    def run_step(self):
        # Collect stimuli from all sources
        stimuli = {}
        for source in self.stimuli_sources:
            stimuli.update(source.get_stimuli())
            
        # Process agent step
        result = self.agent.mind.process_step(stimuli)
        
        # Update physical state
        self.agent.update_physical_state()
        
        # Advance time
        self.current_time += self.time_step
        
        return {
            "time": self.current_time,
            "agent_state": {
                "physical": self.agent.physical_state,
                "mental": result
            }
        }

    def shutdown(self):
        """Properly shutdown the simulation"""
        logger.info("Simulation shutdown initiated")
        self.agent.shutdown()
        logger.info("Simulation shutdown completed")
        return True

class LLMInterface:
    def __init__(self, base_url=None):
        logger.info(f"Initializing LLM interface with base URL: {API_BASE_URL}")
        self.client = None
        self.initialize_client()
        self.agent = None  # Will be set when attached to an agent
        self.tool_registry = ToolRegistry()
        self.tool_registry.llm_client = self  # Give the registry access to the LLM
        
        # Add generation counter and thinking time tracking
        self.generation_counter = 0
        self.total_thinking_time = 0.0
        
        # Register default tools from the tool registry first
        self.tool_registry.register_default_tools()
        
        # Then register any LLM-specific tools
        self._register_agent_specific_tools()
        
    def initialize_client(self):
        try:
            self.client = OpenAI(
                base_url=API_BASE_URL,
                api_key="not-needed"  # Since it's your local endpoint
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            self.client = None
    
    def _generate_completion(self, prompt, system_message):
        """Generate a completion using the OpenAI API"""
        try:
            if not self.client:
                logger.warning("OpenAI client not initialized, attempting to initialize")
                self.initialize_client()
                if not self.client:
                    logger.error("Failed to initialize OpenAI client")
                    return "Error: Could not connect to LLM API"
            
            # Increment the generation counter
            self.generation_counter += 1
            
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Log the request payload
            request_payload = {
                "model": LLM_CONFIG["model"],
                "messages": messages,
                "max_tokens": LLM_CONFIG["max_tokens"],
                "temperature": LLM_CONFIG["temperature"]
            }
            logger.info(f"Request payload: {json.dumps(request_payload, indent=2)}")
            
            # Start tracking with Langfuse
            generation = langfuse.generation(
                name="llm-completion",
                model=LLM_CONFIG["model"],
                input={
                    "messages": messages,
                    "max_tokens": LLM_CONFIG["max_tokens"], 
                    "temperature": LLM_CONFIG["temperature"]
                },
                metadata={
                    "system_message": system_message,
                    "prompt_length": len(prompt),
                    "timestamp": datetime.now().isoformat(),
                    "generation_number": self.generation_counter
                }
            )
            
            # Send the request
            logger.info("Sending request to LLM API...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
                messages=messages,
                max_tokens=LLM_CONFIG["max_tokens"],
                temperature=LLM_CONFIG["temperature"]
            )
            
            elapsed = time.time() - start_time
            # Update total thinking time
            self.total_thinking_time += elapsed
            logger.info(f"LLM API response received in {elapsed:.2f}s. Total thinking time: {self.total_thinking_time:.2f}s")
            
            # Extract the response text
            result = response.choices[0].message.content
            
            # Log the full response (without truncating)
            logger.info(f"LLM response: '{result}'")
            
            # Update the Langfuse generation with the result
            generation.end(
                output=result,
                metadata={
                    "elapsed_time": elapsed,
                    "response_time": elapsed,
                    "output_length": len(result),
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None,
                    "total_thinking_time": self.total_thinking_time,
                    "total_tokens": len(result) // 4,  # Very rough approximation
                    "generation_counter": self.generation_counter
                }
            )
            
            # Save the state to persist thinking time and generation count
            if self.tool_registry:
                self.tool_registry.save_state()
            
            # If we have an agent with a thought_summary_manager, and this is not an ego thought, save to summary db
            if hasattr(self, 'agent') and hasattr(self.agent, 'thought_summary_manager'):
                # Clean system message for comparison (trim whitespace)
                clean_system_msg = system_message.strip()
                
                # For debugging, log the entire system message occasionally
                if random.random() < 0.05:  # Log 5% of system messages for debugging
                    logger.debug(f"FULL SYSTEM MESSAGE: '{clean_system_msg}'")
                
                # Identify different types of messages
                is_ego_thought = any([
                    "agent's ego" in clean_system_msg.lower(),
                    "you are his ego" in clean_system_msg.lower()
                ])
                
                # Identify pure tool responses (not agent thoughts that include tool invocations)
                is_tool_response = any([
                    "search engine" in clean_system_msg.lower(),
                    "research analyst" in clean_system_msg.lower(),
                    "web_scrape" in prompt and "response" in prompt.lower(),
                    "search_web" in prompt and "response" in prompt.lower()
                ])
                
                # Identify agent thought process messages
                is_agent_thought = any([
                    "your name is simon" in clean_system_msg.lower(),
                    "name is simon" in clean_system_msg.lower(),
                    "you are simon" in clean_system_msg.lower(),
                    "thought process" in clean_system_msg.lower(),
                    "agent system instructions" in clean_system_msg.lower()
                ])
                
                # As a fallback, check if it looks like AGENT_SYSTEM_INSTRUCTIONS
                if not is_agent_thought and len(clean_system_msg) > 100:
                    # Check for key phrases in the default agent system instructions
                    agent_instruction_markers = [
                        "high-agency",
                        "introspective",
                        "emotional state",
                        "your responses should be natural",
                        "your personality"
                    ]
                    
                    # Count how many markers are present
                    marker_count = sum(1 for marker in agent_instruction_markers 
                                      if marker in clean_system_msg.lower())
                    
                    # If at least 2 markers are present, it's likely the agent instructions
                    is_agent_thought = marker_count >= 2
                
                if is_agent_thought and len(result) > 100:
                    # This is the agent's main thought process
                    self.agent.thought_summary_manager.add_thought(result, thought_type="normal_thought")
                    logger.info(f"Added agent thought to summary database, length: {len(result)}")
                elif is_tool_response:
                    # This is a tool response, not an agent thought
                    logger.info("Skipping tool response from summary database")
                else:
                    # Log detailed message for debugging
                    logger.info(f"Skipping message - agent:{is_agent_thought}, ego:{is_ego_thought}, tool:{is_tool_response}")
                    logger.debug(f"System message (first 100 chars): '{clean_system_msg[:100]}'")
                    
                    # To debug issues, log the first part of the result too
                    logger.debug(f"Result preview: '{result[:100]}...'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}", exc_info=True)
            
            # Log error in Langfuse if it was created
            if 'generation' in locals():
                generation.end(
                    error=str(e),
                    metadata={"error_type": "llm_api_error"}
                )
                
            return f"Error occurred while generating thought: {str(e)}"

    @observe()
    def generate_thought(self, context):
        # Langfuse context is automatically created by the @observe decorator
        # Create a structured trace for the thought generation process
        trace = langfuse.trace(
            name="thought-generation",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "has_ego": bool(context.get("ego_thoughts", "")),
                "has_stimuli": bool(context.get("stimuli")),
                "generation_counter": self.generation_counter,
                "total_thinking_time": self.total_thinking_time
            }
        )
        
        try:
            # Debug context sizes to identify potential issues
            logger.info("Analyzing context sizes for thought generation:")
            for key, value in context.items():
                if isinstance(value, str):
                    logger.info(f"  - Context[{key}]: {len(value)} characters")
                elif isinstance(value, list):
                    logger.info(f"  - Context[{key}]: {len(value)} items")
                    # Check if any list items are extremely large
                    for i, item in enumerate(value[:5]):  # First 5 items only
                        if isinstance(item, str) and len(item) > 1000:
                            logger.warning(f"    - Large item at index {i}: {len(item)} characters")
                elif isinstance(value, dict):
                    logger.info(f"  - Context[{key}]: {len(value)} keys")
                else:
                    logger.info(f"  - Context[{key}]: {type(value)}")

            # Track generation time
            start_time = time.time()
            
            # Get ego thoughts from previous cycle if any
            ego_thoughts = context.get("ego_thoughts", "")
            
            # Get original short-term memory
            orig_short_term_memory = context.get("short_term_memory", [])
            
            # Keep a working copy of short-term memory that will be updated during the conversation
            # This ensures each response can see previous responses in the same thinking cycle
            working_short_term_memory = list(orig_short_term_memory)
            
            # Check if memories are strings or objects with content attribute
            if working_short_term_memory and isinstance(working_short_term_memory[0], str):
                recent_memories = working_short_term_memory
            else:
                recent_memories = [m.content for m in working_short_term_memory if hasattr(m, 'content')]
            
            # Get recent journal entries if available
            recent_journal_entries = []
            if hasattr(self.agent, 'journal'):
                recent_journal_entries = self.agent.journal.read_recent_entries(num_entries=10)
            
            # Generate the available tools documentation
            available_tools_docs = []
            for i, tool_doc in enumerate(self.tool_registry.list_tools(), 1):
                tool_text = TOOL_DOCUMENTATION_TEMPLATE.format(
                    index=i,
                    name=tool_doc["name"],
                    description=tool_doc["description"],
                    usage=tool_doc["usage"]
                )
                available_tools_docs.append(tool_text)
            
            available_tools_text = "\n".join(available_tools_docs)
            
            # Add journal entries to the context if available
            journal_context = ""
            if recent_journal_entries:
                journal_context = "\nRecent journal entries:\n" + "\n".join(recent_journal_entries)
            
            # Format recent tool usage
            recent_tools = self.tool_registry.get_recent_tools(10)
            if recent_tools:
                tool_entries = []
                for tool in recent_tools:
                    # Format parameters as key:value pairs
                    params_str = ", ".join([f"{k}:{v}" for k, v in tool['params'].items()])
                    # Include timestamp in a readable format
                    timestamp = datetime.fromisoformat(tool['timestamp']).strftime("%H:%M:%S")
                    tool_entries.append(f"- {timestamp} | {tool['name']}({params_str})")
                recent_tools_text = "\n".join(tool_entries)
            else:
                recent_tools_text = "No recent tool usage"
            
            # Format recent tool results
            recent_results = self.tool_registry.get_recent_results(3)
            if recent_results:
                results_entries = []
                for result in recent_results:
                    # Format the result with success/failure indicator
                    res_obj = result['result']
                    if res_obj.get('success', False):
                        output = res_obj.get('output', 'No output')
                        # Truncate long outputs
                        if len(output) > 200:
                            output = output[:200] + "..."
                        results_entries.append(f"- {result['name']}: SUCCESS - {output}")
                    else:
                        error = res_obj.get('error', 'Unknown error')
                        results_entries.append(f"- {result['name']}: FAILED - {error}")
                recent_results_text = "\n".join(results_entries)
            else:
                recent_results_text = "No recent results"
            
            # Get current goals with duration information
            goals = self.tool_registry.get_goals()
            logger.info(f"Retrieved goals in generate_thought: {goals}")
            logger.info(f"Number of short-term goals: {len(goals.get('short_term_details', []))}")
            logger.info(f"Raw short-term goals data: {goals.get('short_term_details')}")
            
            # Format short-term goals with bullet points, numbering and duration
            if goals["short_term"]:
                numbered_goals = []
                for i, goal_detail in enumerate(goals.get("short_term_details", [])):
                    goal_text = goal_detail["text"]
                    duration = goal_detail["duration"]
                    cycles = goal_detail["cycles"]
                    numbered_goals.append(f"[{i}] {goal_text} (Active for: {duration}, {cycles} cycles)")
                    logger.info(f"Formatted goal {i}: {goal_text}")
                short_term_goals = "\n".join(numbered_goals)
                logger.info(f"Formatted short-term goals: {short_term_goals}")
                logger.info(f"Total formatted goals: {len(numbered_goals)}")
            else:
                short_term_goals = "No short-term goals"
                logger.info("No short-term goals found")
                
            # Format long-term goal with emphasis and duration
            if goals["long_term"] and goals.get("long_term_details"):
                long_term_detail = goals["long_term_details"]
                duration = long_term_detail["duration"]
                cycles = long_term_detail["cycles"]
                long_term_goal = f">>> {goals['long_term']} <<< (Active for: {duration}, {cycles} cycles)"
                logger.info(f"Formatted long-term goal: {long_term_goal}")
            else:
                long_term_goal = "No long-term goal"
                logger.info("No long-term goal found")
            
            # Add generation statistics
            generation_stats = f"You've thought {self.generation_counter} times for {self.total_thinking_time:.2f}s"
            
            # Check for pending Telegram messages
            pending_messages = "No pending messages."
            if hasattr(self, 'agent') and self.agent and self.agent.telegram_bot:
                # Just check if there are unread messages, don't mark as read yet
                unread_messages = self.agent.telegram_bot.get_unread_messages()
                if unread_messages:
                    pending_messages = f"You have {len(unread_messages)} unread message(s). Use the receive_telegram tool to read them."
            
            # Format ego thoughts if there are any
            if ego_thoughts:
                # Check if these ego thoughts have already been processed
                processed_ego_marker = "[PROCESSED]"
                if processed_ego_marker in ego_thoughts:
                    logger.warning("Ego thoughts already processed - skipping formatting")
                    formatted_ego_thoughts = ""
                else:
                    # Format ego thoughts with a dramatic presentation
                    formatted_ego_thoughts = f"!!!\n***Suddenly, the following thought(s) occur to you. You try to ignore them but cannot, they echo in your mind for a full minute, completely diverting your attention before they fade, and you can think again:\n{ego_thoughts}\n***\n!!!"
                    
                    # Mark as processed to prevent re-formatting in future cycles
                    if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
                        self.agent.mind.conscious.ego_thoughts = f"{processed_ego_marker}{ego_thoughts}"
                        logger.info("Marked ego thoughts as processed")
            else:
                formatted_ego_thoughts = ""
                
            # Initialize the response and tool results
            current_response = ""
            last_tool_results = context.get("last_tool_results", [])
            all_tool_calls = []
            iteration_count = 0
            ego_thoughts_refresh_interval = 10  # Increased from 3 to 10 - Generate ego thoughts less frequently
            intermediate_ego_thoughts = ""
            max_iterations = 10  # Maximum number of iterations to prevent infinite loops
            
            # Create a local variable to track whether to show ego thoughts in this iteration
            # We only want to show them in the first iteration
            show_ego_thoughts_this_iteration = formatted_ego_thoughts != ""
            
            while iteration_count < max_iterations:
                iteration_count += 1
                # Create span for each thought iteration
                thought_span = trace.span(name=f"thought-iteration-{iteration_count}")
                
                # Logger to trace memory usage
                logger.info(f"Iteration {iteration_count} - Working with {len(recent_memories)} memories")
                
                # Check if it's time to generate intermediate ego thoughts
                if iteration_count > 1 and (iteration_count - 1) % ego_thoughts_refresh_interval == 0:
                    # Create Langfuse span for ego thoughts generation
                    ego_thoughts_span = thought_span.span(name="intermediate-ego-thoughts")
                    
                    logger.info(f"Generating intermediate ego thoughts at iteration {iteration_count}")
                    # Create updated context with current state
                    interim_context = dict(context)
                    interim_context.update({
                        "recent_memories": recent_memories,
                        "recent_response": current_response
                    })
                    
                    # Generate intermediate ego thoughts
                    intermediate_ego_thoughts = self._generate_ego_thoughts(interim_context)
                    logger.info(f"Generated intermediate ego thoughts: {intermediate_ego_thoughts[:100]}...")
                    
                    # Update Langfuse span with the generated ego thoughts
                    ego_thoughts_span.update(
                        output=intermediate_ego_thoughts[:200],
                        metadata={
                            "ego_thoughts_length": len(intermediate_ego_thoughts),
                            "iteration": iteration_count
                        }
                    )
                    ego_thoughts_span.end()
                    
                    # Format ego thoughts for the next prompt
                    if intermediate_ego_thoughts:
                        formatted_ego_thoughts = f"!!!\n***Suddenly, the following thought(s) occur to you. You try to ignore them but cannot, they echo in your mind for a full minute, completely diverting your attention before they fade, and you can think again:\n{intermediate_ego_thoughts}\n***\n!!!"
                    
                    # Store the ego thoughts for the next cycle
                    if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
                        # Mark as processed to prevent re-formatting in future cycles
                        processed_ego_marker = "[PROCESSED]"
                        self.agent.mind.conscious.ego_thoughts = f"{processed_ego_marker}{intermediate_ego_thoughts}"
                        logger.info("Marked intermediate ego thoughts as processed")
                        
                        # Also add ego thoughts to short-term memory
                        if hasattr(self.agent.mind, 'memory'):
                            memory_entry = f"[EGO THOUGHTS]: {intermediate_ego_thoughts}"
                            self.agent.mind.memory.short_term.append(memory_entry)
                            logger.info(f"Added intermediate ego thoughts to short-term memory")
                            
                            # Update recent_memories to include the ego thoughts
                            if isinstance(working_short_term_memory[0], str):
                                working_short_term_memory.append(memory_entry)
                                recent_memories = working_short_term_memory[-10:]  # Keep only last 10
                            else:
                                class MemoryEntry:
                                    def __init__(self, content):
                                        self.content = content
                                working_short_term_memory.append(MemoryEntry(memory_entry))
                                recent_memories = [m.content for m in working_short_term_memory[-10:] if hasattr(m, 'content')]
                
                # Prepare the prompt
                if last_tool_results:
                    # Format all tool results for the prompt
                    results_text = "\n".join([
                        f"Tool '{name}' result: {result.get('output', '')}"
                        for name, result in last_tool_results
                    ])
                    
                    # Get recent bug reports if available
                    recent_bug_reports = ""
                    if hasattr(self, 'tool_registry') and hasattr(self.tool_registry, 'get_recent_bug_reports'):
                        try:
                            bug_reports = self.tool_registry.get_recent_bug_reports(3)  # Get last 3 reports
                            if bug_reports and bug_reports[0] != "No bug reports yet":
                                recent_bug_reports = "\nRecent bug reports:\n" + "\n".join(bug_reports)
                        except Exception as e:
                            logger.warning(f"Error getting bug reports: {e}")
                    
                    # Only show ego thoughts in the first iteration
                    current_ego_thoughts = formatted_ego_thoughts if show_ego_thoughts_this_iteration else ""
                    
                    prompt = THOUGHT_PROMPT_TEMPLATE.format(
                        emotional_state=context.get("emotional_state", {}),
                        recent_memories=recent_memories if recent_memories else "None",
                        subconscious_thoughts=context.get("subconscious_thoughts", []),
                        stimuli=context.get("stimuli", {}),
                        current_focus=context.get("current_focus"),
                        available_tools=available_tools_text + journal_context + recent_bug_reports,  # Add bug reports to available tools
                        recent_tools=recent_tools_text,
                        recent_results=recent_results_text,
                        short_term_goals=short_term_goals,
                        long_term_goal=long_term_goal,
                        generation_stats=generation_stats,
                        ego_thoughts=current_ego_thoughts,
                        pending_messages=pending_messages
                    )
                else:
                    # Get recent bug reports if available
                    recent_bug_reports = ""
                    if hasattr(self, 'tool_registry') and hasattr(self.tool_registry, 'get_recent_bug_reports'):
                        try:
                            bug_reports = self.tool_registry.get_recent_bug_reports(3)  # Get last 3 reports
                            if bug_reports and bug_reports[0] != "No bug reports yet":
                                recent_bug_reports = "\nRecent bug reports:\n" + "\n".join(bug_reports)
                        except Exception as e:
                            logger.warning(f"Error getting bug reports: {e}")
                    
                    # Only show ego thoughts in the first iteration
                    current_ego_thoughts = formatted_ego_thoughts if show_ego_thoughts_this_iteration else ""
                    
                    prompt = THOUGHT_PROMPT_TEMPLATE.format(
                        emotional_state=context.get("emotional_state", {}),
                        recent_memories=recent_memories if recent_memories else "None",
                        subconscious_thoughts=context.get("subconscious_thoughts", []),
                        stimuli=context.get("stimuli", {}),
                        current_focus=context.get("current_focus"),
                        available_tools=available_tools_text + journal_context + recent_bug_reports,  # Add bug reports
                        recent_tools=recent_tools_text,
                        recent_results=recent_results_text,
                        short_term_goals=short_term_goals,
                        long_term_goal=long_term_goal,
                        generation_stats=generation_stats,
                        ego_thoughts=current_ego_thoughts,
                        pending_messages=pending_messages
                    )
                
                # After the first iteration, don't show ego thoughts anymore
                show_ego_thoughts_this_iteration = False
                
                # Generate the response
                response = self._generate_completion(prompt, AGENT_SYSTEM_INSTRUCTIONS)

                # Parse and handle any tool invocations
                parsed_response, tool_results = self._handle_tool_invocations(response, context)
                
                # Track tool usage in Langfuse
                for tool_name, tool_result in tool_results:
                    # Create span for each tool call
                    tool_span = thought_span.span(name=f"tool-{tool_name}")
                    
                    # Add tool details to Langfuse
                    tool_span.update(
                        input=json.dumps(self.tool_registry.get_recent_tools(1)[0]['params'] if self.tool_registry.get_recent_tools(1) else {}),
                        output=json.dumps(tool_result),
                        metadata={
                            "tool_name": tool_name,
                            "success": tool_result.get("success", False)
                        }
                    )
                    tool_span.end()
                    
                    # Keep track of all tools used
                    all_tool_calls.append({
                        "name": tool_name,
                        "result": tool_result
                    })
                
                # Add current iteration's response to the ongoing conversation context
                # Important: Update recent_memories for the next iteration to include this response
                if parsed_response:
                    # Add the parsed response to our current response
                    if current_response:
                        current_response += "\n\n"
                    current_response += parsed_response
                    
                    # Add this response to the working memory for the next iteration
                    working_short_term_memory.append(parsed_response)
                    
                    # Update recent_memories list for next iteration's context
                    if isinstance(working_short_term_memory[0], str):
                        recent_memories = working_short_term_memory[-10:]  # Keep only last 10
                    else:
                        # If we had objects with content attribute, maintain that structure
                        class MemoryEntry:
                            def __init__(self, content):
                                self.content = content
                        
                        working_short_term_memory.append(MemoryEntry(parsed_response))
                        recent_memories = [m.content for m in working_short_term_memory[-10:] if hasattr(m, 'content')]
                
                # End this thought iteration span
                thought_span.end()
                
                # If no tools were used, we're done
                if not tool_results:
                    break
                    
                # Update the last tool results for the next iteration
                last_tool_results = tool_results
                
                # Refresh recent tool usage and results for the next iteration's prompt
                recent_tools = self.tool_registry.get_recent_tools(10)
                if recent_tools:
                    tool_entries = []
                    for tool in recent_tools:
                        # Format parameters as key:value pairs
                        params_str = ", ".join([f"{k}:{v}" for k, v in tool['params'].items()])
                        # Include timestamp in a readable format
                        timestamp = datetime.fromisoformat(tool['timestamp']).strftime("%H:%M:%S")
                        tool_entries.append(f"- {timestamp} | {tool['name']}({params_str})")
                    recent_tools_text = "\n".join(tool_entries)
                else:
                    recent_tools_text = "No recent tool usage"
                
                # Update recent tool results too
                recent_results = self.tool_registry.get_recent_results(3)
                if recent_results:
                    results_entries = []
                    for result in recent_results:
                        res_obj = result['result']
                        if res_obj.get('success', False):
                            output = res_obj.get('output', 'No output')
                            if len(output) > 200:
                                output = output[:200] + "..."
                            results_entries.append(f"- {result['name']}: SUCCESS - {output}")
                        else:
                            error = res_obj.get('error', 'Unknown error')
                            results_entries.append(f"- {result['name']}: FAILED - {error}")
                    recent_results_text = "\n".join(results_entries)
                else:
                    recent_results_text = "No recent results"
                    
                # Check if we're about to reach the maximum iterations
                if iteration_count == max_iterations - 1:
                    # Add a message to inform that we're stopping due to too many iterations
                    loop_warning = "\n\n[SYSTEM: Maximum tool invocation loop reached. Forcing stop of tool chain. Please complete your thought without additional tools.]"
                    current_response += loop_warning
                    logger.warning(f"Maximum iteration limit reached ({max_iterations}). Forcing end of tool chain.")
            
            # Generate ego thoughts about this thinking cycle
            updated_context = dict(context)
            updated_context.update({
                "recent_memories": recent_memories,
                "recent_response": current_response,
                "previous_ego_thoughts": intermediate_ego_thoughts
            })
            
            # Create Langfuse span for final ego thoughts generation
            final_ego_span = trace.span(name="final-ego-thoughts")
            
            new_ego_thoughts = self._generate_ego_thoughts(updated_context)
            logger.info(f"Generated new ego thoughts: {new_ego_thoughts[:100]}...")
            
            # Update Langfuse span with the generated ego thoughts
            final_ego_span.update(
                output=new_ego_thoughts[:200],
                metadata={
                    "ego_thoughts_length": len(new_ego_thoughts),
                    "previous_ego_thoughts_length": len(intermediate_ego_thoughts) if intermediate_ego_thoughts else 0,
                    "generation_counter": self.generation_counter
                }
            )
            final_ego_span.end()
            
            # End the trace with full results
            trace.update(
                output=current_response,
                metadata={
                    "tool_calls_count": len(all_tool_calls),
                    "tools_used": [t["name"] for t in all_tool_calls],
                    "thought_length": len(current_response),
                    "iterations": iteration_count,
                    "generation_counter": self.generation_counter,
                    "total_thinking_time": self.total_thinking_time,
                    "has_ego_thoughts": bool(new_ego_thoughts)
                }
            )
            
            # Store the ego thoughts for the next cycle
            if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
                # Mark as processed to prevent re-formatting in future cycles
                processed_ego_marker = "[PROCESSED]"
                self.agent.mind.conscious.ego_thoughts = f"{processed_ego_marker}{new_ego_thoughts}"
                logger.info("Marked final ego thoughts as processed")
                
                # Also add ego thoughts to short-term memory
                if hasattr(self.agent.mind, 'memory'):
                    memory_entry = f"[FINAL EGO THOUGHTS]: {new_ego_thoughts}"
                    self.agent.mind.memory.short_term.append(memory_entry)
                    logger.info(f"Added final ego thoughts to short-term memory")
            
            # Track total thinking time
            end_time = time.time()
            thinking_time = end_time - start_time
            self.total_thinking_time += thinking_time
            
            # Add the final thought response to short-term memory
            if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'memory'):
                memory_entry = f"[FINAL THOUGHT]: {current_response}"
                self.agent.mind.memory.short_term.append(memory_entry)
                logger.info(f"Added final thought response to short-term memory")
            
            # Add thought to the summary database for summarization
            if hasattr(self.agent, 'thought_summary_manager'):
                # Check if the thought contains content (not just tool invocations)
                # We want to include thoughts that contain tool invocations as part of reasoning
                if len(current_response) > 100:
                    self.agent.thought_summary_manager.add_thought(current_response, thought_type="normal_thought")
                    logger.info(f"Added thought to summary database, length: {len(current_response)}")
                else:
                    logger.info("Skipping short thought from summary database")
            
            return current_response, new_ego_thoughts
            
        except Exception as e:
            logger.error(f"Error in generate_thought: {e}", exc_info=True)
            
            # Update trace with error information
            if 'trace' in locals():
                trace.update(
                    error=str(e),
                    metadata={"error_type": "thought_generation_error"}
                )
            
            return f"Error generating thought: {str(e)}", ""

    def _register_agent_specific_tools(self):
        """Register tools specific to this agent"""
        # Tool to set focus - LLM specific since it affects LLM context
        self.tool_registry.register_tool(Tool(
            name="set_focus",
            description="Set the current focus of attention",
            function=lambda value=None, focus=None: self._set_focus(value or focus),
            usage_example="[TOOL: set_focus(data analysis)]"
        ))
        
        # Tool for system instructions - LLM specific since it affects LLM context
        self.tool_registry.register_tool(Tool(
            name="system_instruction",
            description="Add a system instruction to be followed during thinking",
            function=lambda instruction=None, value=None: self._add_system_instruction(instruction or value),
            usage_example="[TOOL: system_instruction(Think more creatively)]"
        ))
        
        # Tool to get generation stats
        self.tool_registry.register_tool(Tool(
            name="get_generation_stats",
            description="Get statistics about LLM generations, including count and total thinking time",
            function=self._get_generation_stats,
            usage_example="[TOOL: get_generation_stats()]"
        ))
        
        # Tool to get goal statistics
        self.tool_registry.register_tool(Tool(
            name="get_goal_stats",
            description="Get statistics about current goals, including duration and cycles",
            function=lambda: self.tool_registry.get_goal_stats(),
            usage_example="[TOOL: get_goal_stats()]"
        ))
        
    def _set_focus(self, value):
        """Set the agent's focus"""
        # Handle None value
        if value is None:
            return {
                "success": False,
                "error": "No focus value provided. Please specify a focus."
            }
            
        if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
            self.agent.mind.conscious.current_focus = value
            return f"Focus set to: {value}"
        return "Unable to set focus: agent mind not properly initialized"
        
    def _add_system_instruction(self, instruction):
        """Add a system instruction to be followed"""
        # Handle None value
        if instruction is None:
            return {
                "success": False,
                "error": "No instruction provided. Please specify an instruction."
            }
            
        # Store the instruction for later use
        if not hasattr(self, 'system_instructions'):
            self.system_instructions = []
        self.system_instructions.append(instruction)
        return f"System instruction added: {instruction}"
    
    def _get_generation_stats(self):
        """Get statistics about generations"""
        stats = {
            "generation_counter": self.generation_counter,
            "total_thinking_time": self.total_thinking_time,
            "avg_thinking_time": self.total_thinking_time / max(1, self.generation_counter)
        }
        
        return {
            "success": True,
            "output": f"Generation statistics:\n- Total generations: {stats['generation_counter']}\n- Total thinking time: {stats['total_thinking_time']:.2f}s\n- Average thinking time: {stats['avg_thinking_time']:.2f}s per generation"
        }
    
    def _format_result(self, result):
        """Format the result to prevent truncation of complex objects"""
        if isinstance(result, (dict, list, tuple, set)):
            try:
                import json
                return json.dumps(result, indent=2, default=str)
            except:
                pass
        return str(result)

    def _handle_tool_invocations(self, response, context):
        """Parse and handle tool invocations in the response"""
        # Updated regex to handle quoted parameters and nested tool invocations
        tool_pattern = r'\[TOOL:\s*(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)\]'
        matches = re.findall(tool_pattern, response)
        
        if not matches:
            return response, []
        
        tool_results = []
        parsed_response = response
        
        # Create a Langfuse trace for tool handling
        tools_trace = langfuse.trace(
            name="tool-invocations-handling",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_tools_found": len(matches),
                "tool_names": [match[0] for match in matches]
            }
        )
        
        # Process all tool invocations in order
        for tool_match in matches:
            tool_name = tool_match[0]
            params_str = tool_match[1]
            
            # Create a span for this specific tool
            tool_span = tools_trace.span(name=f"tool-{tool_name}")
            
            # Parse parameters - improved parameter parsing
            params = {}
            
            logger.info(f"Parsing parameters for tool {tool_name}: '{params_str}'")
            
            if params_str:
                # Handle parameters with quoted values that may contain nested tool invocations
                if '=' in params_str:
                    # Match key=value pairs where value can be quoted and contain nested content
                    param_pairs = []
                    # First try to extract parameters with quoted values
                    quoted_params = re.findall(r'(\w+)=(["\'])((?:(?!\2).|\\\2)*)\2', params_str)
                    for key, quote, value in quoted_params:
                        params[key] = value
                        # Mark this part as processed by replacing it in the params_str
                        params_str = params_str.replace(f"{key}={quote}{value}{quote}", "", 1)
                    
                    # Then process any remaining key=value pairs without quotes
                    remaining_params = re.findall(r'(\w+)=([^,"\'][^,]*)', params_str)
                    for key, value in remaining_params:
                        key = key.strip()
                        value = value.strip()
                        params[key] = value
                
                # If we still don't have parameters and there are colons, try key:value format
                if not params and ':' in params_str:
                    # Handle key:value format with possible quoted values
                    param_pairs = re.findall(r'([^,:]+?):([^,]+?)(?:,|$)', params_str)
                    
                    for key, value in param_pairs:
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        # Try to convert value to appropriate type
                        try:
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            elif '.' in value and value.replace('.', '', 1).isdigit():
                                value = float(value)
                            elif value.isdigit():
                                value = int(value)
                        except (ValueError, AttributeError):
                            # Keep as string if conversion fails
                            pass
                        
                        params[key] = value
                
                # If we still don't have parameters, treat the whole string as a single value
                if not params:
                    params["value"] = params_str.strip()
            
            logger.info(f"Final parsed parameters for {tool_name}: {params}")
            
            # Update Langfuse with parsed parameters
            tool_span.update(
                input=json.dumps(params),
                metadata={
                    "tool_name": tool_name,
                    "raw_params_str": params_str
                }
            )
            
            try:
                # Execute the tool using the registry
                result = self.tool_registry.execute_tool(tool_name, **params)
                
                # Ensure result is in the correct format
                if isinstance(result, dict):
                    if "success" not in result:
                        result = {
                            "success": True,
                            "output": self._format_result(result)
                        }
                else:
                    result = {
                        "success": True,
                        "output": self._format_result(result)
                    }
                
                # Update Langfuse with successful result
                tool_span.update(
                    output=json.dumps(result),
                    metadata={
                        "success": True,
                        "output_length": len(str(result.get("output", "")))
                    }
                )
                
            except Exception as e:
                # Handle tool execution errors
                result = {
                    "success": False,
                    "error": str(e),
                    "tool": tool_name
                }
                
                # Update Langfuse with error
                tool_span.update(
                    error=str(e),
                    metadata={
                        "success": False,
                        "error_type": "tool_execution_error"
                    }
                )
                
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            
            # End the tool span
            tool_span.end()
            
            tool_results.append((tool_name, result))
            
            # Replace the tool invocation with its result in the response
            tool_invocation = f"[TOOL: {tool_name}({params_str})]"
            result_text = f"Tool '{tool_name}' result: {result.get('output', '')}"
            parsed_response = parsed_response.replace(tool_invocation, result_text)
        
        # If we have multiple tool results, format them together
        if len(tool_results) > 1:
            # Create a summary of all tool results
            results_summary = "\n".join([
                f"Tool '{name}' result: {result.get('output', '')}"
                for name, result in tool_results
            ])
            
            # Add the summary to the parsed response
            parsed_response += f"\n\nAll tool results:\n{results_summary}"
        
        # End the tools trace with final results
        tools_trace.update(
            output=json.dumps([{"tool": name, "success": result.get("success", False)} for name, result in tool_results]),
            metadata={
                "successful_tools": sum(1 for _, result in tool_results if result.get("success", False)),
                "failed_tools": sum(1 for _, result in tool_results if not result.get("success", False))
            }
        )
        
        return parsed_response, tool_results
    
    def _execute_tool(self, tool_name, params, context):
        """Execute a tool and return the result"""
        result = self.tool_registry.execute_tool(tool_name, **params)
        
        # Format the result for display
        if isinstance(result, dict):
            if result.get("success", False):
                return self._format_result(result.get("output", ""))
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
        
        return self._format_result(result)

    def attach_to_agent(self, agent):
        """Attach this LLM interface to an agent"""
        self.agent = agent

    def _generate_ego_thoughts(self, context):
        """Generate ego thoughts as a higher-level perspective on the agent's state and actions"""
        try:
            logger.info("Generating ego thoughts...")
            
            # Get the templates
            from templates import EGO_SYSTEM_2_INSTRUCTIONS, THOUGHT_PROMPT_TEMPLATE
            
            # Check for previous ego thoughts
            previous_ego_thoughts = context.get("previous_ego_thoughts", "")
            # Check if ego_thoughts is also in context (could cause duplication)
            existing_ego_thoughts = context.get("ego_thoughts", "")
            
            # Debug logging for ego thoughts sources
            logger.info(f"Previous ego thoughts length: {len(previous_ego_thoughts)} chars")
            logger.info(f"Existing ego thoughts length: {len(existing_ego_thoughts)} chars")
            
            # If both are present, log a warning as this could indicate duplication
            if previous_ego_thoughts and existing_ego_thoughts:
                logger.warning("Both previous_ego_thoughts and ego_thoughts found in context - potential duplication risk")
                
                # Check if they're identical
                if previous_ego_thoughts == existing_ego_thoughts:
                    logger.warning("Duplicate detected: previous_ego_thoughts and ego_thoughts are identical")
                
            # Always prefer previous_ego_thoughts if available
            if previous_ego_thoughts:
                logger.info(f"Found previous ego thoughts: {previous_ego_thoughts[:100]}...")
            elif existing_ego_thoughts:
                # If no previous_ego_thoughts but ego_thoughts exists, use that instead
                previous_ego_thoughts = existing_ego_thoughts
                logger.info(f"Using existing ego_thoughts as previous: {previous_ego_thoughts[:100]}...")
            
            emotional_state = context.get("emotional_state", {})
            
            # Get more recent memories (10 instead of the default 5)
            if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'memory'):
                # Retrieve 10 memories for ego's broader perspective
                recent_memories = self.agent.mind.memory.recall(emotional_state, n=10)
                if isinstance(recent_memories, list) and recent_memories and hasattr(recent_memories[0], 'content'):
                    recent_memories = [m.content for m in recent_memories if hasattr(m, 'content')]
                logger.info(f"Retrieved {len(recent_memories)} memories for ego perspective")
            else:
                # Fall back to context memories if direct memory access not available
                recent_memories = context.get("recent_memories", [])
                if isinstance(recent_memories, list) and recent_memories and hasattr(recent_memories[0], 'content'):
                    recent_memories = [m.content for m in recent_memories if hasattr(m, 'content')]
            
            # Limit subconscious thoughts to just a few key ones
            subconscious_thoughts = context.get("subconscious_thoughts", [])
            if isinstance(subconscious_thoughts, list) and len(subconscious_thoughts) > 3:
                # Just keep the 3 most important subconscious thoughts
                subconscious_thoughts = subconscious_thoughts[:3]
                logger.info("Limited subconscious thoughts to 3 for ego perspective")
            
            stimuli = context.get("stimuli", {})
            current_focus = context.get("current_focus", "Nothing in particular")
            
            # Get goals with duration information
            goals = self.tool_registry.get_goals()
            logger.info(f"Retrieved goals in _generate_ego_thoughts: {goals}")
            logger.info(f"Number of short-term goals in ego thoughts: {len(goals.get('short_term_details', []))}")
            
            # Format short-term goals with duration
            if goals.get("short_term_details"):
                short_term_goals = []
                for i, goal_detail in enumerate(goals["short_term_details"]):
                    text = goal_detail["text"]
                    duration = goal_detail["duration"]
                    cycles = goal_detail.get("cycles", 0)
                    short_term_goals.append(f"[{i}] {text} (Active for: {duration}, {cycles} cycles)")
                    logger.info(f"Ego thoughts - formatted goal {i}: {text}")
                short_term_goals = "\n".join(short_term_goals)
                logger.info(f"Ego thoughts - total formatted goals: {len(short_term_goals.split('\n'))}")
            else:
                short_term_goals = "No short-term goals"
                logger.info("Ego thoughts - no short-term goals found")
            
            # Format long-term goal with duration
            if goals.get("long_term_details"):
                long_term_detail = goals["long_term_details"]
                text = long_term_detail["text"]
                duration = long_term_detail["duration"]
                cycles = long_term_detail.get("cycles", 0)
                long_term_goal = f"{text} (Active for: {duration}, {cycles} cycles)"
            else:
                long_term_goal = "No long-term goal"
            
            # Get generation stats
            generation_stats = f"Thought {self.generation_counter} times for {self.total_thinking_time:.2f}s"
            
            # Get available tools
            available_tools_docs = []
            for i, tool_doc in enumerate(self.tool_registry.list_tools(), 1):
                tool_text = TOOL_DOCUMENTATION_TEMPLATE.format(
                    index=i,
                    name=tool_doc["name"],
                    description=tool_doc["description"],
                    usage=tool_doc["usage"]
                )
                available_tools_docs.append(tool_text)
            available_tools_text = "\n".join(available_tools_docs)
            
            # Get recent tool usage and results
            recent_tools = self.tool_registry.get_recent_tools(10)
            if recent_tools:
                tool_entries = []
                for tool in recent_tools:
                    params_str = ", ".join([f"{k}:{v}" for k, v in tool['params'].items()])
                    timestamp = datetime.fromisoformat(tool['timestamp']).strftime("%H:%M:%S")
                    tool_entries.append(f"- {timestamp} | {tool['name']}({params_str})")
                recent_tools_text = "\n".join(tool_entries)
            else:
                recent_tools_text = "No recent tool usage"
            
            recent_results = self.tool_registry.get_recent_results(3)
            if recent_results:
                results_entries = []
                for result in recent_results:
                    res_obj = result['result']
                    if res_obj.get('success', False):
                        output = res_obj.get('output', 'No output')
                        if len(output) > 200:
                            output = output[:200] + "..."
                        results_entries.append(f"- {result['name']}: SUCCESS - {output}")
                    else:
                        error = res_obj.get('error', 'Unknown error')
                        results_entries.append(f"- {result['name']}: FAILED - {error}")
                recent_results_text = "\n".join(results_entries)
            else:
                recent_results_text = "No recent results"
            
            # Also include the current thought response if available
            recent_response = context.get("recent_response", "")
            if recent_response:
                recent_memories.append(f"[MOST RECENT THOUGHT]: {recent_response}")
            
            # Format the prompt with all the information
            # Get recent bug reports if available
            recent_bug_reports = ""
            if hasattr(self, 'tool_registry') and hasattr(self.tool_registry, 'get_recent_bug_reports'):
                try:
                    bug_reports = self.tool_registry.get_recent_bug_reports(3)  # Get last 3 reports
                    if bug_reports and bug_reports[0] != "No bug reports yet":
                        recent_bug_reports = "\nRecent bug reports:\n" + "\n".join(bug_reports)
                except Exception as e:
                    logger.warning(f"Error getting bug reports for ego thoughts: {e}")
            
            ego_prompt = THOUGHT_PROMPT_TEMPLATE.format(
                emotional_state=emotional_state,
                recent_memories=recent_memories if recent_memories else "None",
                subconscious_thoughts=subconscious_thoughts,
                stimuli=stimuli,
                current_focus=current_focus,
                short_term_goals=short_term_goals,
                long_term_goal=long_term_goal,
                generation_stats=generation_stats,
                available_tools=available_tools_text + recent_bug_reports,
                recent_tools=recent_tools_text,
                recent_results=recent_results_text,
                ego_thoughts="",
                pending_messages="No pending messages."
            )
            
            # If we have previous ego thoughts, add them to the prompt
            if previous_ego_thoughts:
                ego_prompt += f"\n\nYour previous ego thoughts were:\n{previous_ego_thoughts}\n\nConsider these thoughts as you develop new insights, but don't repeat them exactly."
            
            # Generate the ego thoughts
            ego_thoughts = self._generate_completion(ego_prompt, EGO_SYSTEM_2_INSTRUCTIONS)
            
            # Log and return the ego thoughts
            logger.info(f"Ego thoughts generated: {ego_thoughts[:100]}...")
            return ego_thoughts
            
        except Exception as e:
            logger.error(f"Error generating ego thoughts: {e}", exc_info=True)
            return ""

    def _recall_research_memories(self, query=None, count=3):
        """Recall memories specifically from research"""
        try:
            # Convert count to integer if it's a string
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    count = 3
                    
            # Get research memories
            memories = self.mind.memory.recall_research(query, count)
            
            if not memories:
                if query:
                    return {
                        "success": True,
                        "output": f"No research memories found related to '{query}'. Try using [TOOL: deepdive_research(query:{query})] to perform research on this topic."
                    }
                else:
                    return {
                        "success": True,
                        "output": "No research memories found. Try using [TOOL: deepdive_research(query:your topic)] to perform research."
                    }
                    
            # Format the result
            result = "Research memories:\n\n"
            for i, memory in enumerate(memories, 1):
                result += f"{i}. {memory}\n\n"
                
            result += "To perform new research, use: [TOOL: deepdive_research(query:your topic)]"
            
            return {
                "success": True,
                "output": result
            }
            
        except Exception as e:
            logger.error(f"Error recalling research memories: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to recall research memories: {str(e)}"
            }

def test_connection(url=API_BASE_URL):
    """Test the connection to the API endpoint"""
    try:
        logger.info(f"Testing connection to {url}")
        response = requests.get(url, timeout=5)  # Add timeout
        logger.info(f"Connection test result: {response.status_code}")
        return response.status_code
    except requests.exceptions.Timeout:
        logger.error("Connection test timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection refused - API server may not be running")
        return None
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return None

# Add a more robust startup sequence
def initialize_system():
    """Initialize the system with proper error handling"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        status = test_connection()
        if status and 200 <= status < 300:
            logger.info("API connection successful")
            return True
        
        logger.warning(f"API connection failed (attempt {attempt+1}/{max_retries}), retrying in {retry_delay}s...")
        time.sleep(retry_delay)
    
    logger.error(f"Failed to connect to API after {max_retries} attempts")
    return False

def handle_shutdown(signum=None, frame=None):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    try:
        if 'controller' in globals():
            controller.shutdown()
            
        # Force cleanup of any potentially hanging resources
        # Close Langfuse client if it's open
        if 'langfuse' in globals():
            try:
                langfuse.flush()
            except:
                pass
        
        # Clean up any OpenAI clients
        for key, value in list(globals().items()):
            if isinstance(value, OpenAI):
                try:
                    del globals()[key]
                except:
                    pass
        
        # Force garbage collection to clean up resources
        import gc
        gc.collect()
        
        logger.info("Shutdown cleanup completed, exiting now")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    # Use os._exit for a more forceful exit that doesn't wait for threads
    import os
    os._exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, handle_shutdown)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_shutdown)  # Termination signal
atexit.register(lambda: handle_shutdown() if 'controller' in globals() else None)

if __name__ == "__main__":
    if initialize_system():
        logger.info("System initialized successfully")
        
        # Initialize components
        llm = LLMInterface()
        agent = Agent(
            llm=llm, 
            memory_path="agent_memory.pkl",
            emotion_path="agent_emotions.pkl",
            telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID")
        )
        controller = SimulationController(agent)
        
        # Run simulation or other operations
        # ...
    else:
        logger.error("System initialization failed")

def reload_templates():
    """Reload templates from the templates.py file"""
    import templates
    importlib.reload(templates)
    logger.info("Templates reloaded")

def update_llm_config(new_config):
    """Update LLM configuration parameters"""
    global LLM_CONFIG
    LLM_CONFIG.update(new_config)
    logger.info(f"LLM configuration updated: {LLM_CONFIG}")

class ProcessingManager:
    def __init__(self, agent):
        self.agent = agent
        self.stimuli_queue = queue.Queue()
        self.processed_stimuli_queue = queue.Queue()  # Queue for processed stimuli
        self.current_cycle_type = None
        # Modified cycle sequence to make ego cycles less frequent
        # Pattern: stimuli, normal, normal, normal, ego, emotion
        self.cycle_types = ['stimuli', 'normal', 'normal', 'normal', 'ego', 'emotion'] 
        self.cycle_index = 0
        self.last_cycle_results = {}
        
    def add_stimuli(self, stimuli):
        """Add raw stimuli to the queue for processing"""
        self.stimuli_queue.put(stimuli)
        
    def next_cycle(self):
        """Run the next processing cycle"""
        # Determine cycle type
        cycle_type = self.cycle_types[self.cycle_index]
        self.cycle_index = (self.cycle_index + 1) % len(self.cycle_types)
        
        # Run the appropriate cycle
        if cycle_type == 'stimuli':
            return self._run_stimuli_cycle()
        elif cycle_type == 'normal':
            return self._run_normal_cycle()
        elif cycle_type == 'ego':
            return self._run_ego_cycle()
        elif cycle_type == 'emotion':
            return self._run_emotion_cycle()
            
    def _run_stimuli_cycle(self):
        """Process raw stimuli and generate interpreted stimuli"""
        # Collect raw stimuli
        raw_stimuli = {}
        while not self.stimuli_queue.empty():
            try:
                raw_stimuli.update(self.stimuli_queue.get_nowait())
            except queue.Empty:
                break
                
        # If no stimuli, skip processing
        if not raw_stimuli:
            return {
                "cycle_type": "stimuli",
                "result": "No stimuli to process"
            }
            
        # Prepare context for stimuli interpretation
        context = {
            "raw_stimuli": raw_stimuli,
            "emotional_state": self.agent.mind.emotion_center.get_state(),
            "current_focus": self.agent.mind.conscious.current_focus,
            "recent_memories": list(self.agent.mind.memory.short_term)[-3:]
        }
        
        # Generate interpretation of stimuli using LLM
        system_message = "You are an AI agent's perception system. Analyze and interpret incoming stimuli."
        
        prompt = f"""
        You have received the following raw stimuli:
        {json.dumps(raw_stimuli, indent=2)}
        
        Based on your current context, interpret these stimuli to extract:
        1. Meaningful information and observations
        2. Potential significance or implications
        3. Priority level (low, medium, high)
        
        Your current emotional state is:
        {json.dumps(context['emotional_state'], indent=2)}
        
        Your current focus is: {context['current_focus'] or 'None'}
        
        Format your response as a JSON object with the following structure:
        {{
            "interpreted_stimuli": [
                {{
                    "source": "source of the stimuli",
                    "observation": "what you observe",
                    "significance": "potential significance",
                    "priority": "low|medium|high"
                }}
            ],
            "attention_focus": "what aspect deserves focus",
            "thought_implications": "initial thoughts prompted by these stimuli"
        }}
        """
        
        # Generate interpretation
        try:
            interpreted_result = self.agent.llm._generate_completion(prompt, system_message)
            
            # Parse the result
            try:
                interpreted_data = json.loads(interpreted_result)
            except:
                # Fallback if JSON parsing fails
                interpreted_data = {
                    "interpreted_stimuli": [
                        {
                            "source": "unknown",
                            "observation": "Failed to parse stimuli",
                            "significance": "unknown",
                            "priority": "low"
                        }
                    ],
                    "attention_focus": "error recovery",
                    "thought_implications": "Need to improve stimuli processing"
                }
            
            # Increment thinking cycle counter for memory tracking
            self.agent.mind.memory.increment_cycle()
                
            # Store the interpretation as a memory with appropriate metadata
            memory_id = self.agent.mind.memory.add_thought(
                content=json.dumps(interpreted_data, indent=2),
                thought_type="stimuli_interpretation",
                emotional_context=self.agent.mind.emotion_center.get_state(),
                source_stimuli=raw_stimuli
            )
                
            # Add interpretation to processed stimuli queue
            self.processed_stimuli_queue.put({
                "raw_stimuli": raw_stimuli,
                "interpretation": interpreted_data
            })
            
            return {
                "cycle_type": "stimuli",
                "raw_stimuli": raw_stimuli,
                "interpreted_stimuli": interpreted_data,
                "memory_id": memory_id
            }
        except Exception as e:
            logger.error(f"Error processing stimuli: {e}")
            
            # Add minimal processed data even on error
            self.processed_stimuli_queue.put({
                "raw_stimuli": raw_stimuli,
                "interpretation": {
                    "error": str(e),
                    "interpreted_stimuli": [
                        {
                            "source": "unknown",
                            "observation": f"Error processing: {str(e)}",
                            "significance": "system error",
                            "priority": "medium"
                        }
                    ]
                }
            })
            
            return {
                "cycle_type": "stimuli",
                "error": str(e),
                "raw_stimuli": raw_stimuli
            }
        
    def _run_normal_cycle(self):
        """Run a normal thinking cycle with processed stimuli"""
        # Collect processed stimuli
        processed_stimuli = []
        while not self.processed_stimuli_queue.empty():
            try:
                processed_stimuli.append(self.processed_stimuli_queue.get_nowait())
            except queue.Empty:
                break
                
        # Run subconscious processes
        subconscious_thoughts = self.agent.mind.subconscious.process(trace_id=None)
        
        # Prepare stimuli for conscious thinking
        if processed_stimuli:
            # Format processed stimuli for conscious thinking
            formatted_stimuli = {
                "processed_stimuli": processed_stimuli,
                "high_priority_items": [
                    item for ps in processed_stimuli 
                    for item in ps["interpretation"].get("interpreted_stimuli", [])
                    if item.get("priority") == "high"
                ]
            }
        else:
            formatted_stimuli = {}
        
        # Conscious thinking with processed stimuli
        conscious_thought = self.agent.mind.conscious.think(
            formatted_stimuli, subconscious_thoughts, trace_id=None
        )
        
        # Extract emotional implications from the thought
        emotional_implications = self._extract_emotional_implications(conscious_thought)
        
        # Update emotions based on thoughts (not raw stimuli)
        if emotional_implications:
            self.agent.mind.emotion_center.update(emotional_implications)
        
        # Decision making
        action = self.agent.mind.conscious.decide_action(conscious_thought)
        
        # Increment thinking cycle counter for memory tracking
        self.agent.mind.memory.increment_cycle()
        
        # Store the thought as a memory with appropriate metadata
        memory_id = self.agent.mind.memory.add_thought(
            content=conscious_thought,
            thought_type="normal_thought",
            emotional_context=self.agent.mind.emotion_center.get_state(),
            source_stimuli=formatted_stimuli if processed_stimuli else None
        )
        
        # Add thought to the summary database for later summarization
        if hasattr(self.agent, 'thought_summary_manager'):
            # Check if the thought contains content (not just tool invocations)
            # We want to include thoughts that contain tool invocations as part of reasoning
            # But avoid adding pure tool responses
            if len(conscious_thought) > 100:
                self.agent.thought_summary_manager.add_thought(
                    conscious_thought, thought_type="normal_thought"
                )
                logger.info("Added normal thought cycle result to summary database for summarization")
            else:
                logger.info("Skipping short thought from summary database")
        
        # Clear ego thoughts after processing to prevent duplication in future cycles
        self.agent.mind.conscious.ego_thoughts = ""
        
        return {
            "cycle_type": "normal",
            "stimuli_processed": len(processed_stimuli) > 0,
            "emotional_state": self.agent.mind.emotion_center.get_state(),
            "subconscious_thoughts": subconscious_thoughts,
            "conscious_thought": conscious_thought,
            "emotional_implications": emotional_implications,
            "action": action,
            "memory_id": memory_id
        }
        
    def _run_ego_cycle(self):
        """Run an ego-focused cycle"""
        # Create a Langfuse trace for the ego cycle
        ego_cycle_trace = langfuse.trace(
            name="ego-cycle",
            metadata={
                "cycle_type": "ego",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Log current state of ego thoughts
        current_ego_thoughts = self.agent.mind.conscious.ego_thoughts
        logger.info(f"Before ego cycle - Current ego_thoughts length: {len(current_ego_thoughts)} chars")
        
        # Check if ego thoughts have already been processed
        processed_ego_marker = "[PROCESSED]"
        if current_ego_thoughts and processed_ego_marker in current_ego_thoughts:
            # Remove the processed marker to get the original thoughts
            current_ego_thoughts = current_ego_thoughts.replace(processed_ego_marker, "")
            logger.info("Removed [PROCESSED] marker from ego thoughts")
        
        # Prepare context for ego thoughts
        context = {
            "emotional_state": self.agent.mind.emotion_center.get_state(),
            "short_term_memory": self.agent.mind.memory.recall(
                self.agent.mind.emotion_center.get_state(), n=5
            ),
            "recent_processed_stimuli": []
        }
        
        # Include the current ego_thoughts as previous_ego_thoughts to avoid duplication
        # This helps the LLM know what's already been generated
        if current_ego_thoughts:
            context["previous_ego_thoughts"] = current_ego_thoughts
            # Do NOT include ego_thoughts again to avoid duplication
        
        # Include recent processed stimuli in ego context
        try:
            while not self.processed_stimuli_queue.empty():
                context["recent_processed_stimuli"].append(self.processed_stimuli_queue.get(block=False))
        except queue.Empty:
            pass
        
        # Update Langfuse with context information
        ego_cycle_trace.update(
            input=json.dumps({
                "previous_ego_thoughts_length": len(current_ego_thoughts) if current_ego_thoughts else 0,
                "memory_count": len(context.get("short_term_memory", [])),
                "stimuli_count": len(context.get("recent_processed_stimuli", []))
            })
        )
        
        # Generate ego thoughts directly
        ego_thoughts = self.agent.llm._generate_ego_thoughts(context)
        
        # Store ego thoughts for next normal cycle
        # Log when we update ego thoughts
        logger.info(f"Setting ego_thoughts - New length: {len(ego_thoughts)} chars")
        # Mark as processed to prevent re-formatting in future cycles
        processed_ego_marker = "[PROCESSED]"
        self.agent.mind.conscious.ego_thoughts = f"{processed_ego_marker}{ego_thoughts}"
        logger.info("Marked ego cycle thoughts as processed")
        
        # Increment thinking cycle counter for memory tracking
        self.agent.mind.memory.increment_cycle()
        
        # Store the ego thoughts as a memory with appropriate metadata
        memory_id = self.agent.mind.memory.add_thought(
            content=ego_thoughts,
            thought_type="ego_thought",
            emotional_context=self.agent.mind.emotion_center.get_state()
        )
        
        # Update Langfuse with results
        ego_cycle_trace.update(
            output=ego_thoughts[:200] if ego_thoughts else "",
            metadata={
                "ego_thoughts_length": len(ego_thoughts) if ego_thoughts else 0,
                "memory_id": memory_id
            }
        )
        ego_cycle_trace.end()
        
        return {
            "cycle_type": "ego",
            "emotional_state": self.agent.mind.emotion_center.get_state(),
            "ego_thoughts": ego_thoughts,
            "memory_id": memory_id
        }
        
    def _run_emotion_cycle(self):
        """Run an emotion-focused cycle - natural decay only"""
        # Apply natural decay of emotions over time
        for emotion in self.agent.mind.emotion_center.emotions.values():
            emotion.intensity *= (1 - emotion.decay_rate)
            emotion.intensity = max(0, min(1, emotion.intensity))
            
        # Recalculate mood
        emotional_state = self.agent.mind.emotion_center.get_state()
        
        # Calculate mood using the same formula as in the EmotionCenter class
        emotions = self.agent.mind.emotion_center.emotions
        positive = emotions['happiness'].intensity + emotions['surprise'].intensity * 0.5 + \
                   emotions.get('focus', Emotion('focus')).intensity * 0.3 + \
                   emotions.get('curiosity', Emotion('curiosity')).intensity * 0.2
        negative = emotions['sadness'].intensity + emotions['anger'].intensity + emotions['fear'].intensity
        mood = (positive - negative) / (positive + negative + 1e-6)
        
        # Update mood
        self.agent.mind.emotion_center.mood = mood
        
        return {
            "cycle_type": "emotion",
            "emotional_state": emotional_state,
            "mood": mood
        }
        
    def _extract_emotional_implications(self, thought):
        """Extract emotional implications from a thought using the existing logic"""
        implications = {}
        
        # Use the same logic as in Conscious._extract_emotional_implications
        if "happy" in thought.lower() or "joy" in thought.lower():
            implications["happiness"] = 0.1
        if "sad" in thought.lower() or "depress" in thought.lower():
            implications["sadness"] = 0.1
        if "angry" in thought.lower() or "frustrat" in thought.lower():
            implications["anger"] = 0.1
        if "afraid" in thought.lower() or "fear" in thought.lower():
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

class ThoughtSummaryManager:
    """Manages the storage and summarization of thoughts"""
    
    def __init__(self, db_path="thought_summaries.pkl"):
        self.db_path = db_path
        self.thoughts_db = self._load_db()
        self.summary_queue = queue.Queue()
        self.summarization_active = True
        self.summary_thread = None
        self.summary_available = True  # Track if summary API is available
        
        # Start the summarization thread
        self.start_summarization()
    
    def _load_db(self):
        """Load the database from disk or create a new one"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading thought database: {e}")
            return []
    
    def _save_db(self):
        """Save the database to disk"""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.thoughts_db, f)
        except Exception as e:
            logger.error(f"Error saving thought database: {e}")
    
    def add_thought(self, thought, thought_type="normal_thought"):
        """Add a thought to the database and queue it for summarization"""
        if thought_type != "normal_thought":
            # Skip ego and emotional thoughts
            logger.info(f"Skipping non-normal thought of type {thought_type}")
            return
        
        # Validate the thought
        if not thought or len(thought) < 50:
            logger.warning(f"Thought too short ({len(thought) if thought else 0} chars), not adding to summary database")
            return
        
        # Log the thought being added
        logger.info(f"Adding thought to summary database - Type: {thought_type}, Length: {len(thought)}")
        logger.info(f"Thought preview: {thought[:100]}...")
            
        timestamp = time.time()
        entry = {
            "timestamp": timestamp,
            "thought": thought,
            "summary": None,
            "timestamp_formatted": datetime.fromtimestamp(timestamp).isoformat()
        }
        
        # Add to database
        self.thoughts_db.append(entry)
        self._save_db()
        
        # Queue for summarization
        if self.summarization_active:
            self.summary_queue.put(entry)
            logger.info(f"Queued thought for summarization, queue size: {self.summary_queue.qsize()}")
        
        return entry
    
    def start_summarization(self):
        """Start the summarization thread"""
        if self.summary_thread is None or not self.summary_thread.is_alive():
            self.summarization_active = True
            self.summary_thread = threading.Thread(target=self._summarization_worker, daemon=True)
            self.summary_thread.start()
            logger.info("Thought summarization thread started")
            return True
        return False
    
    def stop_summarization(self):
        """Stop the summarization thread"""
        self.summarization_active = False
        if self.summary_thread and self.summary_thread.is_alive():
            logger.info("Stopping thought summarization thread")
            return True
        return False
    
    def get_summaries(self, limit=10, offset=0):
        """Get the most recent thought summaries"""
        sorted_entries = sorted(self.thoughts_db, key=lambda x: x["timestamp"], reverse=True)
        return sorted_entries[offset:offset+limit]
    
    def get_summary_status(self):
        """Get the status of the summarization process"""
        return {
            "active": self.summarization_active,
            "api_available": self.summary_available,
            "queue_size": self.summary_queue.qsize(),
            "total_entries": len(self.thoughts_db),
            "summarized_entries": sum(1 for entry in self.thoughts_db if entry.get("summary") is not None)
        }
    
    def _summarize_thought(self, entry):
        """Summarize a thought using the summary API"""
        try:
            # Get the template from templates.py
            from templates import SUMMARY_TEMPLATE
            
            # Replace {thoughts} with the actual thought content
            prompt = SUMMARY_TEMPLATE.replace("{thoughts}", entry["thought"])
            
            # Make API request to summary endpoint
            logger.info(f"Sending request to summary API for thought {entry['timestamp_formatted']}")
            response = requests.post(
                f"{SUMMARY_BASE_URL}/chat/completions",
                json={
                    "model": "summary-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                timeout=60  # Increased timeout to 60 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Update entry with summary
                entry["summary"] = summary
                
                # Mark API as available
                self.summary_available = True
                logger.info(f"Successfully summarized thought: {summary[:50]}...")
                
                # Save the updated database
                self._save_db()
                return True
            else:
                logger.error(f"Failed to summarize thought: {response.status_code}, {response.text}")
                if response.status_code in (500, 502, 503, 504):
                    # Mark API as unavailable on server errors
                    self.summary_available = False
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Summary API error: {e}")
            self.summary_available = False
            return False
        except Exception as e:
            logger.error(f"Error summarizing thought: {e}")
            return False
    
    def _summarization_worker(self):
        """Worker thread for processing the summary queue"""
        while self.summarization_active:
            try:
                # If API is not available, wait before retrying
                if not self.summary_available:
                    logger.warning("Summary API not available, pausing summarization")
                    time.sleep(30)  # Wait 30 seconds before checking again
                    continue
                
                # Get next entry from queue
                entry = self.summary_queue.get(block=True, timeout=1)
                
                # Skip if already summarized
                if entry.get("summary") is not None:
                    self.summary_queue.task_done()
                    continue
                
                # Summarize the thought synchronously
                logger.info(f"Processing thought from {entry['timestamp_formatted']} for summarization")
                success = self._summarize_thought(entry)
                
                # If unsuccessful and API is available, re-queue with backoff
                if not success and self.summary_available:
                    logger.warning("Failed to summarize thought, will retry later")
                    time.sleep(5)  # Short delay before re-queuing
                    self.summary_queue.put(entry)
                
                self.summary_queue.task_done()
                
            except queue.Empty:
                # No items in queue, just continue
                pass
            except Exception as e:
                logger.error(f"Error in summarization worker: {e}")
                time.sleep(5)  # Delay before next iteration on error
                
        logger.info("Summarization worker stopped")

    def force_summarize_all(self):
        """Force immediate summarization of all thoughts in the queue"""
        logger.info(f"Forcing summarization of {self.summary_queue.qsize()} pending thoughts")
        
        # Process all thoughts in the queue synchronously
        processed_count = 0
        failed_count = 0
        
        # Make a copy of the queue to avoid concurrent modification
        thoughts_to_process = []
        while not self.summary_queue.empty():
            try:
                thoughts_to_process.append(self.summary_queue.get(block=False))
                self.summary_queue.task_done()
            except queue.Empty:
                break
        
        # Process all thoughts
        for entry in thoughts_to_process:
            # Skip if already summarized
            if entry.get("summary") is not None:
                continue
                
            logger.info(f"Force summarizing thought from {entry['timestamp_formatted']}")
            success = self._summarize_thought(entry)
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
                # Put back in queue for later processing
                self.summary_queue.put(entry)
        
        return {
            "processed": processed_count,
            "failed": failed_count,
            "remaining": self.summary_queue.qsize()
        }
