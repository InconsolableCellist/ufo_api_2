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
import telegram  # Add this for Telegram bot functionality
import atexit
import signal
import tools
from tools import ToolRegistry, Tool, Journal, TelegramBot  # Import specific classes from tools

# Import templates
from templates import THOUGHT_PROMPT_TEMPLATE, ACTION_RESPONSE_TEMPLATE, SYSTEM_MESSAGE_TEMPLATE, TOOL_DOCUMENTATION_TEMPLATE

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
        
        # Color based on log level
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        
        # Special coloring for specific content
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_simulation.log')  # File handler without colors
    ]
)

# Add colored console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('agent_simulation')
logger.addHandler(console_handler)

# Define API endpoint configuration
API_HOST = "mlboy"
API_PORT = "5000"
API_BASE_URL = f"http://{API_HOST}:{API_PORT}/v1"

# Configure Langfuse
os.environ["LANGFUSE_HOST"] = "http://zen:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-c039333b-d33f-44a5-a33c-5827e783f4b2"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-3c59e794-7426-49ea-b9a1-2eae0999fadf"
langfuse = Langfuse()

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETION_MODEL = "local-model"

# LLM generation parameters
LLM_CONFIG = {
    "model": COMPLETION_MODEL,
    "max_tokens": 750,
    "temperature": 1.75,
    "system_message": "You are an AI agent's thought process. Respond with natural, introspective thoughts based on the current context, ideally less than 500 tokens."
}

class Emotion:
    def __init__(self, name, intensity=0.0, decay_rate=0.1, influence_factors=None):
        self.name = name
        self.intensity = intensity
        self.decay_rate = decay_rate  # How quickly it fades over time
        self.influence_factors = influence_factors or {}  # What stimuli affect this emotion
        
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
            'energy': Emotion('energy', intensity=0.4, decay_rate=0.02),  # Start with some energy
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
        positive = self.emotions['happiness'].intensity + self.emotions['surprise'].intensity * 0.5
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
            
    def get_embedding(self, text):
        """Get embedding vector for text using OpenAI API"""
        try:
            logger.info(f"Requesting embedding for text: '{text[:50]}...' (truncated)")
            
            # Log connection attempt
            logger.info(f"Connecting to embedding API at {self.client.base_url}")
            
            start_time = time.time()
            
            # Create generation in Langfuse
            generation = langfuse.generation(
                name="embedding-request",
                model=EMBEDDING_MODEL,
                input=text,
                metadata={
                    "base_url": self.client.base_url,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Try a direct HTTP request to have more control over parameters
            headers = {
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": EMBEDDING_MODEL,
                "input": text,
                "encoding_format": "float"
                # Remove dimensions and pooling parameters that might be causing issues
            }
            
            # Fix the URL to avoid double slashes - ensure proper URL construction
            base_url = API_BASE_URL.rstrip('/')  # Remove trailing slash if present
            api_url = f"{base_url}/embeddings"
            logger.info(f"Sending request to: {api_url}")
            
            response_raw = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30  # Add a timeout to prevent hanging
            )
            
            # Check for error status codes
            if response_raw.status_code != 200:
                logger.warning(f"Embedding API returned status code {response_raw.status_code}: {response_raw.text}")
                raise Exception(f"API returned status code {response_raw.status_code}")
            
            response_data = response_raw.json()
            
            elapsed = time.time() - start_time
            
            # Update Langfuse with response
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
            
            # Log error in Langfuse if it was created
            if 'generation' in locals():
                generation.end(
                    error=str(e),
                    metadata={"error_type": "embedding_api_error"}
                )
            
            # Raise the exception to be handled by the caller
            raise Exception(f"Failed to get embedding: {str(e)}")
            
    def add(self, content, emotional_context=None):
        """Add a memory with its emotional context"""
        logger.info(f"Adding memory: '{content[:50]}...' (truncated)")
        
        self.short_term.append(content)
        self.long_term.append(content)
        
        # Get embedding and add to FAISS index
        logger.info("Getting embedding for new memory")
        try:
            embedding = self.get_embedding(content)
            
            # Verify dimensions match what we expect
            if self.embedding_dim != embedding.shape[0]:
                logger.warning(f"Embedding dimension mismatch: got {embedding.shape[0]}, expected {self.embedding_dim}")
                # If we already have an index but dimensions don't match, we need to rebuild
                if len(self.embeddings) > 0:
                    logger.error("Cannot add embedding with different dimension to existing index")
                    # Return without adding this memory to avoid crashing
                    return
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
                self.associations[content] = emotional_context
                
            # Optionally persist after updates
            if self.persist_path:
                logger.info(f"Persisting memory to {self.persist_path}")
                self.save()
                
        except Exception as e:
            logger.error(f"Failed to add memory due to embedding error: {e}")
            # Remove the memory from long_term since we couldn't embed it
            if content in self.long_term:
                self.long_term.remove(content)
            # Don't persist since we didn't successfully add the memory
            
    def recall(self, emotional_state, query=None, n=3):
        """Recall memories based on emotional state and/or query text"""
        if len(self.long_term) == 0:
            logger.info("No memories to recall (empty long-term memory)")
            return []
            
        # If query is provided, use it for semantic search
        if query:
            logger.info(f"Recalling memories with query: '{query}'")
            query_embedding = self.get_embedding(query)
            faiss.normalize_L2(query_embedding.reshape(1, -1))
            
            logger.info(f"Searching FAISS index with query embedding")
            distances, indices = self.index.search(query_embedding.reshape(1, -1), min(n, len(self.long_term)))
            
            logger.info(f"FAISS search results - indices: {indices[0]}, distances: {distances[0]}")
            memories = [self.long_term[idx] for idx in indices[0]]
            logger.info(f"Retrieved {len(memories)} memories via semantic search")
        else:
            # Filter memories based on emotional state
            # TODO: Implement more sophisticated emotional memory retrieval:
            # - Consider context-dependent emotional salience
            # - Add weighting for memory importance/intensity
            # - Implement primacy/recency effects
            # - Consider retrieval based on emotional contrast, not just similarity
            logger.info(f"Recalling memories based on emotional state: {emotional_state}")
            relevant = [mem for mem, ctx in self.associations.items() 
                       if self._emotional_match(ctx, emotional_state)]
            logger.info(f"Found {len(relevant)} emotionally relevant memories")
            
            memories = random.sample(relevant, min(n, len(relevant))) if relevant else []
            
            # If not enough emotional matches, supplement with random memories
            if len(memories) < n:
                remaining = n - len(memories)
                available = [mem for mem in self.long_term if mem not in memories]
                logger.info(f"Adding {min(remaining, len(available))} random memories to supplement")
                if available:
                    memories.extend(random.sample(available, min(remaining, len(available))))
                    
        logger.info(f"Recalled {len(memories)} memories in total")
        return memories
        
    def _emotional_match(self, memory_emotion, current_emotion):
        """Check if memory emotion matches current emotion"""
        # TODO: Enhance emotional matching with:
        # - Emotion similarity metrics that consider all emotions, not just a few
        # - Weighting system based on intensity and relevance
        # - Consider emotional trajectories/trends
        # - Add "emotional opposites" matching for contrast recall
        # - Implement emotional context modeling
        
        # More sophisticated matching using mood and dominant emotions
        if 'mood' in memory_emotion and 'mood' in current_emotion:
            return abs(memory_emotion['mood'] - current_emotion['mood']) < 0.3
        
        # Fallback to simple matching
        return any(memory_emotion.get(e, 0) > 0.5 and current_emotion.get(e, 0) > 0.5 
                  for e in ['happiness', 'sadness', 'anger', 'fear'])
                  
    def save(self):
        """Save memory state to disk"""
        if not self.persist_path:
            return
            
        data = {
            'long_term': self.long_term,
            'associations': self.associations,
            'embeddings': np.vstack(self.embeddings) if self.embeddings else np.array([])
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self):
        """Load memory state from disk"""
        if not os.path.exists(self.persist_path):
            return
            
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
            
        self.long_term = data['long_term']
        self.associations = data['associations']
        
        # Rebuild FAISS index
        if len(data['embeddings']) > 0:
            # Set embedding_dim from the loaded embeddings
            self.embedding_dim = data['embeddings'].shape[1]
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.embeddings = list(data['embeddings'])
            self.index.add(data['embeddings'])
        else:
            # If no embeddings, set a default dimension but don't create index yet
            self.embedding_dim = 1536  # Default OpenAI embedding dimension
            self.index = None  # Will be created on first embedding

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
        
    def process(self, trace_id):
        thoughts = []
        for process in self.background_processes:
            thoughts.extend(process(trace_id))
        return thoughts
        
    def _surface_memories(self, trace_id):
        # TODO: Improve memory surfacing algorithm to consider more factors:
        # - Current context relevance
        # - Emotional resonance with different weightings
        # - Recency vs importance balance
        # - Associative connections between memories
        emotional_state = self.emotion_center.get_state()
        return self.memory.recall(emotional_state)
        
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
        # TODO: Add working memory as a separate component from long-term memory
        # TODO: Implement attention mechanisms to prioritize information processing
        # TODO: Add metacognition capabilities to reflect on own thinking
        
    def think(self, stimuli, subconscious_thoughts, trace_id):
        # Prepare context for LLM
        # TODO: Improve context preparation:
        # - Implement prioritization of memories based on relevance
        # - Add internal feedback loops for self-reflection
        # - Include previous thinking steps to maintain coherence
        # - Add structured reasoning patterns for different types of problems
        context = {
            "short_term_memory": list(self.memory.short_term),
            "emotional_state": self.emotion_center.get_state(),
            "subconscious_thoughts": subconscious_thoughts,
            "stimuli": stimuli,
            "current_focus": self.current_focus
        }
        
        # Generate thought
        thought = self.llm.generate_thought(context)
        self.memory.add(thought, self.emotion_center.get_state())
        
        # Extract emotional implications from thought
        emotional_implications = self._extract_emotional_implications(thought)
        
        # Apply emotional implications
        for emotion, change in emotional_implications.items():
            if emotion in self.emotion_center.emotions:
                self.emotion_center.emotions[emotion].intensity += change
                self.emotion_center.emotions[emotion].intensity = max(0, min(1, self.emotion_center.emotions[emotion].intensity))
        
        return thought
        
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
                "stimuli": json.dumps(stimuli) if stimuli else "{}"
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
                "action": action
            }
            
            # Update the trace with the result
            trace.update(
                output=json.dumps(result),
                metadata={"success": True}
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
        
        # Tool to get current emotional state
        self.llm.tool_registry.register_tool(Tool(
            name="get_emotional_state",
            description="Get a natural description of the current emotional state",
            function=lambda: self._get_emotional_state(),
            usage_example="[TOOL: get_emotional_state()]"
        ))
        
        self.llm.tool_registry.register_tool(Tool(
            name="send_telegram",
            description="Send an urgent message to the human via Telegram (use only for important communications)",
            function=lambda message: self._send_telegram_message(message),
            usage_example="[TOOL: send_telegram(message:This is an urgent message)]"
        ))
        
        self.llm.tool_registry.register_tool(Tool(
            name="write_journal",
            description="Write an entry to the agent's journal for future reference",
            function=lambda entry: self._write_journal_entry(entry),
            usage_example="[TOOL: write_journal(entry:Today I learned something interesting)]"
        ))
        
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
        """Recall memories"""
        count = int(count)
        memories = self.mind.memory.recall(
            emotional_state=self.mind.emotion_center.get_state(),
            query=query,
            n=count
        )
        return memories

    def _send_telegram_message(self, message):
        """Send a message via Telegram"""
        if not self.telegram_bot.token:
            return "Telegram bot not configured. Message not sent."
        
        success = self.telegram_bot.send_message(message)
        if success:
            return f"Message sent successfully: '{message}'"
        else:
            return "Failed to send message. Check logs for details."

    def _write_journal_entry(self, entry):
        """Write an entry to the journal"""
        success = self.journal.write_entry(entry)
        if success:
            return f"Journal entry recorded: '{entry[:50]}...'" if len(entry) > 50 else f"Journal entry recorded: '{entry}'"
        else:
            return "Failed to write journal entry. Check logs for details."

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
            response_text = response.choices[0].message.content
            logger.info(f"LLM response: '{response_text}'")
            
            # Update Langfuse with the response
            generation.end(
                output=response_text,
                metadata={
                    "elapsed_time": elapsed,
                    "output_length": len(response_text),
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None,
                    "total_thinking_time": self.total_thinking_time,
                    "generation_number": self.generation_counter
                }
            )
            
            # Save the state to persist thinking time and generation count
            if self.tool_registry:
                self.tool_registry.save_state()
                
            return response_text
            
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
                "emotional_state": json.dumps(context.get("emotional_state", {})),
                "has_stimuli": bool(context.get("stimuli")),
                "generation_counter": self.generation_counter,
                "total_thinking_time": self.total_thinking_time
            }
        )

        try:
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
            
            # Get current goals
            goals = self.tool_registry.get_goals()
            logger.info(f"Retrieved goals in generate_thought: {goals}")
            
            # Format short-term goals with bullet points and numbering
            if goals["short_term"]:
                numbered_goals = []
                for i, goal in enumerate(goals["short_term"]):
                    numbered_goals.append(f"[{i}] {goal}")
                short_term_goals = "\n".join(numbered_goals)
                logger.info(f"Formatted short-term goals: {short_term_goals}")
            else:
                short_term_goals = "No short-term goals"
                logger.info("No short-term goals found")
                
            # Format long-term goal with emphasis
            if goals["long_term"]:
                long_term_goal = f">>> {goals['long_term']} <<<"
                logger.info(f"Formatted long-term goal: {long_term_goal}")
            else:
                long_term_goal = "No long-term goal"
                logger.info("No long-term goal found")
            
            # Get recent user conversations and latest response
            recent_user_conversations = "\n\n".join(self.tool_registry.read_user_conversations(5))
            if not recent_user_conversations:
                recent_user_conversations = "No recent conversations with the user."
            
            user_response = self.tool_registry.get_latest_user_response()
            
            # Add generation statistics
            generation_stats = f"You've thought {self.generation_counter} times for {self.total_thinking_time:.2f}s"
            
            # Initialize the response and tool results
            current_response = ""
            last_tool_results = context.get("last_tool_results", [])
            all_tool_calls = []
            iteration_count = 0
            
            while True:
                iteration_count += 1
                # Create span for each thought iteration
                thought_span = trace.span(name=f"thought-iteration-{iteration_count}")
                
                # Logger to trace memory usage
                logger.info(f"Iteration {iteration_count} - Working with {len(recent_memories)} memories")
                
                # Prepare the prompt
                if last_tool_results:
                    # Format all tool results for the prompt
                    results_text = "\n".join([
                        f"Tool '{name}' result: {result.get('output', '')}"
                        for name, result in last_tool_results
                    ])
                    
                    prompt = ACTION_RESPONSE_TEMPLATE.format(
                        emotional_state=context.get("emotional_state", {}),
                        recent_memories=recent_memories if recent_memories else "None",
                        subconscious_thoughts=context.get("subconscious_thoughts", []),
                        stimuli=context.get("stimuli", {}),
                        current_focus=context.get("current_focus"),
                        result=results_text,  # Use all tool results
                        available_tools=available_tools_text + journal_context,
                        recent_tools=recent_tools_text,
                        recent_results=recent_results_text,
                        short_term_goals=short_term_goals,
                        long_term_goal=long_term_goal,
                        recent_user_conversations=recent_user_conversations,
                        user_response=user_response,
                        generation_stats=generation_stats
                    )
                else:
                    prompt = THOUGHT_PROMPT_TEMPLATE.format(
                        emotional_state=context.get("emotional_state", {}),
                        recent_memories=recent_memories if recent_memories else "None",
                        subconscious_thoughts=context.get("subconscious_thoughts", []),
                        stimuli=context.get("stimuli", {}),
                        current_focus=context.get("current_focus"),
                        available_tools=available_tools_text + journal_context,
                        recent_tools=recent_tools_text,
                        recent_results=recent_results_text,
                        short_term_goals=short_term_goals,
                        long_term_goal=long_term_goal,
                        recent_user_conversations=recent_user_conversations,
                        user_response=user_response,
                        generation_stats=generation_stats
                    )
                
                # Generate the response
                response = self._generate_completion(prompt, SYSTEM_MESSAGE_TEMPLATE)
                
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
            
            # End the trace with full results
            trace.update(
                output=current_response,
                metadata={
                    "tool_calls_count": len(all_tool_calls),
                    "tools_used": [t["name"] for t in all_tool_calls],
                    "thought_length": len(current_response),
                    "iterations": iteration_count,
                    "generation_counter": self.generation_counter,
                    "total_thinking_time": self.total_thinking_time
                }
            )
            
            return current_response
            
        except Exception as e:
            logger.error(f"Error in generate_thought: {e}", exc_info=True)
            
            # Update trace with error information
            if 'trace' in locals():
                trace.update(
                    error=str(e),
                    metadata={"error_type": "thought_generation_error"}
                )
            
            return f"Error generating thought: {str(e)}"

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
    
    def _handle_tool_invocations(self, response, context):
        """Parse and handle tool invocations in the response"""
        # Updated regex to handle quoted parameters and more flexible formats
        tool_pattern = r'\[TOOL:\s*(\w+)\s*\(([^)]*)\)\]'
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
                # Handle both key:value and plain value formats
                if ':' in params_str:
                    # Handle key:value format
                    # Split by commas, but not within quotes
                    param_pairs = re.findall(r'([^,]+?):([^,]+?)(?:,|$)', params_str)
                    logger.info(f"Parsed parameter pairs: {param_pairs}")
                    
                    for key, value in param_pairs:
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        # Try to convert value to appropriate type
                        try:
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            elif '.' in value:
                                value = float(value)
                            else:
                                try:
                                    value = int(value)
                                except ValueError:
                                    # Keep as string if conversion fails
                                    pass
                        except (ValueError, AttributeError):
                            # Keep as string if conversion fails
                            pass
                        
                        params[key] = value
                else:
                    # Handle plain value format (e.g., set_focus(data analysis))
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
                            "output": str(result)
                        }
                else:
                    result = {
                        "success": True,
                        "output": str(result)
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
                return str(result.get("output", ""))
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
        
        return str(result)

    def attach_to_agent(self, agent):
        """Attach this LLM interface to an agent"""
        self.agent = agent

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
    if 'controller' in globals():
        controller.shutdown()
    sys.exit(0)

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
