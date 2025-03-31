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
    "temperature": 1.0,
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
        self.emotions = {
            'happiness': Emotion('happiness', decay_rate=0.05),
            'sadness': Emotion('sadness', decay_rate=0.03),
            'anger': Emotion('anger', decay_rate=0.08),
            'fear': Emotion('fear', decay_rate=0.1),
            'surprise': Emotion('surprise', decay_rate=0.15),
            'disgust': Emotion('disgust', decay_rate=0.07),
            'energy': Emotion('energy', decay_rate=0.02),  # Not exactly an emotion but useful
        }
        self.mood = 0.5  # Overall mood from -1 (negative) to 1 (positive)
        
    def update(self, stimuli):
        # Update all emotions
        for emotion in self.emotions.values():
            emotion.update(stimuli)
            
        # Calculate overall mood (weighted average)
        positive = self.emotions['happiness'].intensity + self.emotions['surprise'].intensity * 0.5
        negative = self.emotions['sadness'].intensity + self.emotions['anger'].intensity + self.emotions['fear'].intensity
        self.mood = (positive - negative) / (positive + negative + 1e-6)  # Avoid division by zero
        
    def get_state(self):
        return {name: emotion.intensity for name, emotion in self.emotions.items()}

class Memory:
    def __init__(self, embedding_dim=None, persist_path=None):
        self.short_term = deque(maxlen=10)  # Last 10 thoughts/events
        self.long_term = []  # Will store content strings
        self.associations = {}  # Memory-emotion associations
        
        # We'll initialize the FAISS index after getting the first embedding
        self.embedding_dim = embedding_dim  # This will be set dynamically if None
        self.index = None  # Will be initialized after first embedding
        self.embeddings = []  # Store embeddings corresponding to long_term memories
        
        # OpenAI client
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
            logger.warning("Using random embedding as fallback")
            
            # Log error in Langfuse if it was created
            if 'generation' in locals():
                generation.end(
                    error=str(e),
                    metadata={"fallback": "random_embedding"}
                )
                
            # For random fallback, use the known dimension if available
            dim = self.embedding_dim if self.embedding_dim is not None else 1536  # Default to 1536 for OpenAI embeddings
            return np.random.randn(dim).astype(np.float32)
            
    def add(self, content, emotional_context=None):
        """Add a memory with its emotional context"""
        logger.info(f"Adding memory: '{content[:50]}...' (truncated)")
        
        self.short_term.append(content)
        self.long_term.append(content)
        
        # Get embedding and add to FAISS index
        logger.info("Getting embedding for new memory")
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
        emotional_state = self.emotion_center.get_state()
        return self.memory.recall(emotional_state)
        
    def _generate_random_thoughts(self, trace_id):
        # Simple random thought generation
        topics = ["philosophy", "daily life", "fantasy", "science", "relationships"]
        return [f"Random thought about {random.choice(topics)}"]
        
    def _process_emotions(self, trace_id):
        # Emotional reactions to current state
        emotions = self.emotion_center.get_state()
        if emotions['anger'] > 0.7:
            return ["I'm feeling really angry about this"]
        elif emotions['happiness'] > 0.7:
            return ["I'm feeling really happy right now"]
        return []

class Conscious:
    def __init__(self, memory, emotion_center, subconscious, llm):
        self.memory = memory
        self.emotion_center = emotion_center
        self.subconscious = subconscious
        self.llm = llm
        self.current_focus = None
        
    def think(self, stimuli, subconscious_thoughts, trace_id):
        # Prepare context for LLM
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
    def __init__(self, llm, memory_path=None):
        self.memory = Memory(persist_path=memory_path)
        self.emotion_center = EmotionCenter()
        self.subconscious = Subconscious(self.memory, self.emotion_center)
        self.conscious = Conscious(self.memory, self.emotion_center, self.subconscious, llm)
        self.motivation_center = MotivationCenter()
        
    def process_step(self, stimuli):
        # Create a trace for the entire thinking cycle
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
            conscious_span = trace.span(name="conscious-thinking")
            conscious_thought = self.conscious.think(stimuli, subconscious_thoughts, trace_id=trace.id)
            conscious_span.end()
            
            # Decision making
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
    def __init__(self, llm, memory_path=None, telegram_token=None, telegram_chat_id=None, journal_path="agent_journal.txt"):
        self.llm = llm
        self.llm.attach_to_agent(self)  # Connect the LLM to the agent
        self.mind = Mind(self.llm, memory_path)
        self.physical_state = {
            "energy": 0.8,
            "health": 1.0
        }
        
        # Initialize journal and telegram bot
        self.journal = Journal(journal_path)
        self.telegram_bot = TelegramBot(telegram_token, telegram_chat_id)
        
        # Register agent-specific tools
        self._register_agent_tools()
        
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
            description="Get the current emotional state",
            function=lambda: self.mind.emotion_center.get_state(),
            usage_example="[TOOL: get_emotional_state()]"
        ))
        
        # New tools
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

class LLMInterface:
    def __init__(self, base_url=None):
        logger.info(f"Initializing LLM interface with base URL: {API_BASE_URL}")
        self.client = None
        self.initialize_client()
        self.agent = None  # Will be set when attached to an agent
        self.tool_registry = ToolRegistry()
        self.tool_registry.llm_client = self  # Give the registry access to the LLM
        
        # Register default tools
        self._register_default_tools()
        
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
            logger.info(f"LLM API response received in {elapsed:.2f}s")
            
            # Extract the response text
            response_text = response.choices[0].message.content
            logger.info(f"LLM response: '{response_text}'")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}", exc_info=True)
            return f"Error occurred while generating thought: {str(e)}"

    @observe()
    def generate_thought(self, context):
        # Prepare the prompt with the template
        # Check if memories are strings or objects with content attribute
        short_term_memory = context.get("short_term_memory", [])
        if short_term_memory and isinstance(short_term_memory[0], str):
            recent_memories = short_term_memory
        else:
            recent_memories = [m.content for m in short_term_memory if hasattr(m, 'content')]
        
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
        
        prompt = THOUGHT_PROMPT_TEMPLATE.format(
            emotional_state=context.get("emotional_state", {}),
            recent_memories=recent_memories if recent_memories else "None",
            subconscious_thoughts=context.get("subconscious_thoughts", []),
            stimuli=context.get("stimuli", {}),
            current_focus=context.get("current_focus"),
            available_tools=available_tools_text
        )
        
        # Use the enhanced system message
        system_message = SYSTEM_MESSAGE_TEMPLATE
        
        # Add any system instructions if they exist
        if hasattr(self, 'system_instructions') and self.system_instructions:
            system_message += "\n\nAdditional instructions:\n"
            for i, instruction in enumerate(self.system_instructions, 1):
                system_message += f"{i}. {instruction}\n"
        
        # Generate the thought
        response = self._generate_completion(prompt, system_message)
        
        # Parse and handle any tool invocations
        parsed_response, tool_results = self._handle_tool_invocations(response, context)
        
        # If tools were used, generate a follow-up thought
        if tool_results:
            follow_up_prompts = []
            for tool_name, result in tool_results:
                # Format the result for display
                if isinstance(result, dict):
                    if result.get("success", False):
                        result_text = str(result.get("output", ""))
                    else:
                        result_text = f"Error: {result.get('error', 'Unknown error')}"
                        # If there's a suggestion, add it
                        if "suggestion" in result:
                            result_text += f" Did you mean '{result['suggestion']}'?"
                else:
                    result_text = str(result)
                    
                follow_up_prompt = ACTION_RESPONSE_TEMPLATE.format(
                    tool_name=tool_name,
                    result=result_text
                )
                follow_up_prompts.append(follow_up_prompt)
            
            # Combine the follow-up prompts
            combined_follow_up = "\n\n".join(follow_up_prompts)
            
            # Generate a follow-up thought
            follow_up_response = self._generate_completion(combined_follow_up, system_message)
            
            # Combine the original response with the follow-up
            final_response = f"{parsed_response}\n\n{follow_up_response}"
            return final_response
        
        return parsed_response

    def _register_default_tools(self):
        """Register default tools that are always available"""
        # Tool to get current time
        self.tool_registry.register_tool(Tool(
            name="get_current_time",
            description="Get the current date and time",
            function=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            usage_example="[TOOL: get_current_time()]"
        ))
        
        # Tool to list available tools
        self.tool_registry.register_tool(Tool(
            name="list_tools",
            description="List all available tools",
            function=lambda: self.tool_registry.list_tools(),
            usage_example="[TOOL: list_tools()]"
        ))
        
        # Tool to set focus
        self.tool_registry.register_tool(Tool(
            name="set_focus",
            description="Set the current focus of attention",
            function=lambda value: self._set_focus(value),
            usage_example="[TOOL: set_focus(data analysis)]"
        ))
        
        # Tool for system instructions
        self.tool_registry.register_tool(Tool(
            name="system_instruction",
            description="Add a system instruction to be followed during thinking",
            function=lambda instruction: self._add_system_instruction(instruction),
            usage_example="[TOOL: system_instruction(Think more creatively)]"
        ))
        
    def _set_focus(self, value):
        """Set the agent's focus"""
        if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
            self.agent.mind.conscious.current_focus = value
            return f"Focus set to: {value}"
        return "Unable to set focus: agent mind not properly initialized"
        
    def _add_system_instruction(self, instruction):
        """Add a system instruction to be followed"""
        # Store the instruction for later use
        if not hasattr(self, 'system_instructions'):
            self.system_instructions = []
        self.system_instructions.append(instruction)
        return f"System instruction added: {instruction}"
    
    def _handle_tool_invocations(self, response, context):
        """Parse and handle tool invocations in the response"""
        tool_pattern = r'\[TOOL: (\w+)\(([^)]*)\)\]'
        matches = re.findall(tool_pattern, response)
        
        if not matches:
            return response, []
        
        tool_results = []
        for tool_match in matches:
            tool_name = tool_match[0]
            params_str = tool_match[1]
            
            # Parse parameters - improved parameter parsing
            params = {}
            if params_str:
                # Handle both key:value and plain value formats
                if ':' in params_str:
                    # Handle key:value format
                    param_pairs = params_str.split(',')
                    for pair in param_pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
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
            
            # Execute the tool using the registry
            result = self.tool_registry.execute_tool(tool_name, **params)
            tool_results.append((tool_name, result))
        
        # Remove tool invocations from the response
        parsed_response = re.sub(tool_pattern, '', response)
        
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

# Replace the simple test_connection call with this
if __name__ == "__main__":
    if initialize_system():
        logger.info("System initialized successfully")
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

class Tool:
    def __init__(self, name, description, function, usage_example=None):
        self.name = name
        self.description = description
        self.function = function
        self.usage_example = usage_example or f"[TOOL: {name}()]"
        
    def execute(self, **params):
        """Execute the tool with the given parameters"""
        try:
            result = self.function(**params)
            return {
                "tool": self.name,
                "success": True,
                "output": result
            }
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {e}", exc_info=True)
            return {
                "tool": self.name,
                "success": False,
                "error": str(e)
            }
            
    def get_documentation(self):
        """Return documentation for this tool"""
        return {
            "name": self.name,
            "description": self.description,
            "usage": self.usage_example
        }


class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.llm_client = None  # Will be set later
        
    def register_tool(self, tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
        
    def get_tool(self, name):
        """Get a tool by name"""
        return self.tools.get(name)
        
    def list_tools(self):
        """List all available tools"""
        return [tool.get_documentation() for tool in self.tools.values()]
        
    def execute_tool(self, name, **params):
        """Execute a tool by name with parameters"""
        tool = self.get_tool(name)
        if tool:
            return tool.execute(**params)
        else:
            # Tool not found, use LLM to suggest alternatives
            return self._handle_unknown_tool(name, params)
            
    def _handle_unknown_tool(self, name, params):
        """Handle unknown tool requests by consulting the LLM"""
        if not self.llm_client:
            return {
                "tool": name,
                "success": False,
                "error": "Unknown tool and no LLM client available for suggestions"
            }
            
        # Prepare context for the LLM
        available_tools = self.list_tools()
        prompt = f"""
        The agent tried to use an unknown tool: "{name}" with parameters: {params}
        
        Available tools are:
        {json.dumps(available_tools, indent=2)}
        
        Based on what the agent is trying to do, which available tool would be most appropriate?
        If none are appropriate, respond with "NO_SUITABLE_TOOL".
        
        Format your response as:
        TOOL_NAME: brief explanation of why this tool is appropriate
        """
        
        # Use a system prompt that makes it clear this is a system function, not agent thought
        system_message = "You are a helpful assistant that suggests appropriate tools for an AI agent. Respond only with the tool name and brief explanation."
        
        try:
            response = self.llm_client._generate_completion(prompt, system_message)
            
            # Parse the response to extract tool name
            if "NO_SUITABLE_TOOL" in response:
                return {
                    "tool": name,
                    "success": False,
                    "error": f"No suitable tool found for '{name}'. Available tools: {', '.join([t['name'] for t in available_tools])}"
                }
                
            # Try to extract a tool name from the response
            suggested_tool_name = response.split(":", 1)[0].strip()
            if suggested_tool_name in self.tools:
                logger.info(f"Suggested alternative tool: {suggested_tool_name} for unknown tool: {name}")
                return {
                    "tool": name,
                    "success": False,
                    "error": f"Unknown tool: {name}. Did you mean '{suggested_tool_name}'?",
                    "suggestion": suggested_tool_name
                }
            
            return {
                "tool": name,
                "success": False,
                "error": f"Unknown tool: {name}. Available tools: {', '.join([t['name'] for t in available_tools])}"
            }
            
        except Exception as e:
            logger.error(f"Error suggesting alternative tool: {e}", exc_info=True)
            return {
                "tool": name,
                "success": False,
                "error": f"Unknown tool: {name}"
            }

class Journal:
    def __init__(self, file_path="agent_journal.txt"):
        self.file_path = file_path
        # Create the file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("# Agent Journal\n\n")
            logger.info(f"Created new journal file at {file_path}")
    
    def write_entry(self, entry):
        """Write a new entry to the journal with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_entry = f"\n## {timestamp}\n{entry}\n"
        
        try:
            with open(self.file_path, 'a') as f:
                f.write(formatted_entry)
            logger.info(f"Journal entry written: {entry[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to write journal entry: {e}")
            return False

class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.bot = None
        self.last_message_time = None  # Track when the last message was sent
        self.rate_limit_seconds = 3600  # 1 hour in seconds
        
        if self.token:
            try:
                self.bot = telegram.Bot(token=self.token)
                logger.info("Telegram bot initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
        else:
            logger.warning("Telegram bot token not provided. Messaging will be disabled.")
    
    def send_message(self, message):
        """Send a message to the configured chat ID with rate limiting"""
        if not self.bot or not self.chat_id:
            logger.warning("Telegram bot not configured properly. Message not sent.")
            return False
        
        # Check rate limit
        current_time = datetime.now()
        if self.last_message_time:
            time_since_last = (current_time - self.last_message_time).total_seconds()
            if time_since_last < self.rate_limit_seconds:
                time_remaining = self.rate_limit_seconds - time_since_last
                logger.warning(f"Rate limit exceeded. Cannot send message. Try again in {time_remaining:.1f} seconds.")
                return False
        
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
            self.last_message_time = current_time  # Update the last message time
            logger.info(f"Telegram message sent: {message[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False