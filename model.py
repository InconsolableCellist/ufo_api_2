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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_simulation.log')
    ]
)
logger = logging.getLogger('agent_simulation')

# Define API endpoint configuration
API_HOST = "mlboy"
API_PORT = "5000"
API_BASE_URL = f"http://{API_HOST}:{API_PORT}/v1"

# Configure Langfuse
os.environ["LANGFUSE_HOST"] = "http://zen:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-c039333b-d33f-44a5-a33c-5827e783f4b2"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-3c59e794-7426-49ea-b9a1-2eae0999fadf"
langfuse = Langfuse()

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
    def __init__(self, embedding_dim=1536, persist_path=None):
        self.short_term = deque(maxlen=10)  # Last 10 thoughts/events
        self.long_term = []  # Will store content strings
        self.associations = {}  # Memory-emotion associations
        
        # FAISS index for vector similarity search
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity
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
                model="text-embedding-ada-002",
                input=text,
                metadata={
                    "base_url": self.client.base_url,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",  # Adjust model as needed
                input=text
            )
            elapsed = time.time() - start_time
            
            # Update Langfuse with response
            generation.end(
                output=f"Embedding vector of dimension {len(response.data[0].embedding)}",
                metadata={
                    "elapsed_time": elapsed,
                    "vector_norm": np.linalg.norm(np.array(response.data[0].embedding)).item()
                }
            )
            
            logger.info(f"Embedding API response received in {elapsed:.2f}s")
            logger.debug(f"Embedding response: {response}")
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            logger.info(f"Embedding vector shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
            
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
                
            # Return a random embedding as fallback
            return np.random.randn(self.embedding_dim).astype(np.float32)
            
    def add(self, content, emotional_context=None):
        """Add a memory with its emotional context"""
        logger.info(f"Adding memory: '{content[:50]}...' (truncated)")
        
        self.short_term.append(content)
        self.long_term.append(content)
        
        # Get embedding and add to FAISS index
        logger.info("Getting embedding for new memory")
        embedding = self.get_embedding(content)
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
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if len(data['embeddings']) > 0:
            self.embeddings = list(data['embeddings'])
            self.index.add(data['embeddings'])

class Subconscious:
    def __init__(self, memory, emotion_center):
        self.memory = memory
        self.emotion_center = emotion_center
        self.background_processes = [
            self._surface_memories,
            self._generate_random_thoughts,
            self._process_emotions
        ]
        
    def process(self):
        thoughts = []
        for process in self.background_processes:
            thoughts.extend(process())
        return thoughts
        
    def _surface_memories(self):
        emotional_state = self.emotion_center.get_state()
        return self.memory.recall(emotional_state)
        
    def _generate_random_thoughts(self):
        # Simple random thought generation
        topics = ["philosophy", "daily life", "fantasy", "science", "relationships"]
        return [f"Random thought about {random.choice(topics)}"]
        
    def _process_emotions(self):
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
        
    def think(self, stimuli, subconscious_thoughts):
        # Prepare context for LLM
        context = {
            "short_term_memory": list(self.memory.short_term),
            "emotional_state": self.emotion_center.get_state(),
            "subconscious_thoughts": subconscious_thoughts,
            "stimuli": stimuli,
            "current_focus": self.current_focus
        }
        
        # Generate thought (in reality would use proper LLM prompt)
        thought = self.llm.generate_thought(context)
        self.memory.add(thought, self.emotion_center.get_state())
        return thought
        
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
        # Update emotional state based on stimuli
        self.emotion_center.update(stimuli)
        
        # Run subconscious processes
        subconscious_thoughts = self.subconscious.process()
        
        # Conscious thinking
        conscious_thought = self.conscious.think(stimuli, subconscious_thoughts)
        
        # Decision making
        action = self.conscious.decide_action(conscious_thought)
        
        return {
            "emotional_state": self.emotion_center.get_state(),
            "subconscious_thoughts": subconscious_thoughts,
            "conscious_thought": conscious_thought,
            "action": action
        }

class Agent:
    def __init__(self, llm, memory_path=None):
        self.mind = Mind(llm, memory_path)
        self.physical_state = {
            "energy": 0.8,
            "health": 1.0
        }
        
    def update_physical_state(self):
        # Physical state affects emotions and vice versa
        emotions = self.mind.emotion_center.get_state()
        self.physical_state["energy"] -= 0.01  # Base energy drain
        self.physical_state["energy"] += emotions["energy"] * 0.05
        self.physical_state["energy"] = max(0, min(1, self.physical_state["energy"]))
        
        # If very low energy, increase desire to rest
        if self.physical_state["energy"] < 0.2:
            self.mind.emotion_center.emotions["energy"].intensity += 0.1

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
    def __init__(self):
        logger.info(f"Initializing LLM interface with base URL: {API_BASE_URL}")
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key="not-needed"  # Since it's your local endpoint
        )
        
    def generate_thought(self, context):
        try:
            # Create a structured prompt from the context
            prompt = self._create_prompt(context)
            logger.info(f"Generated prompt for thought: '{prompt[:100]}...' (truncated)")
            
            # Log connection attempt
            logger.info(f"Connecting to LLM API at {self.client.base_url}")
            
            # Log the request payload
            request_payload = {
                "model": "local-model",
                "messages": [
                    {"role": "system", "content": "You are an AI agent's thought process. Respond with natural, introspective thoughts based on the current context."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            logger.info(f"Request payload: {json.dumps(request_payload, indent=2)}")
            
            # Create generation in Langfuse
            generation = langfuse.generation(
                name="agent-thought",
                model="local-model",
                model_parameters={
                    "max_tokens": 150,
                    "temperature": 0.7
                },
                input=[
                    {"role": "system", "content": "You are an AI agent's thought process. Respond with natural, introspective thoughts based on the current context."},
                    {"role": "user", "content": prompt}
                ],
                metadata={
                    "base_url": self.client.base_url,
                    "timestamp": datetime.now().isoformat(),
                    "context_type": str(type(context))
                }
            )
            
            # Make the API call with timing
            start_time = time.time()
            logger.info("Sending request to LLM API...")
            
            response = self.client.chat.completions.create(
                model="local-model",  # This can be any identifier your backend accepts
                messages=[
                    {"role": "system", "content": "You are an AI agent's thought process. Respond with natural, introspective thoughts based on the current context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            elapsed = time.time() - start_time
            logger.info(f"LLM API response received in {elapsed:.2f}s")
            
            # Log the response
            response_content = response.choices[0].message.content.strip()
            logger.info(f"LLM response: '{response_content}'")
            
            # Update Langfuse with response
            generation.end(
                output=response_content,
                metadata={
                    "elapsed_time": elapsed,
                    "token_count": len(response_content.split())
                }
            )
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error generating thought: {e}", exc_info=True)
            error_message = "I'm having trouble processing my thoughts right now."
            
            # Log error in Langfuse if it was created
            if 'generation' in locals():
                generation.end(
                    error=str(e),
                    metadata={"fallback_message": error_message}
                )
                
            logger.info(f"Returning fallback message: '{error_message}'")
            return error_message
    
    def _create_prompt(self, context):
        emotional_state = context["emotional_state"]
        memory = context["short_term_memory"]
        subconscious = context["subconscious_thoughts"]
        
        prompt = f"""
Current emotional state: {emotional_state}
Recent memories: {memory[-3:] if memory else 'None'}
Subconscious thoughts: {subconscious}
Current stimuli: {context['stimuli']}
Current focus: {context['current_focus']}

Based on this context, what are my current thoughts?
"""
        return prompt

def test_connection(url=API_BASE_URL):
    """Test the connection to the API endpoint"""
    try:
        logger.info(f"Testing connection to {url}")
        response = requests.get(url)
        logger.info(f"Connection test result: {response.status_code}")
        return response.status_code
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return None

# Add this to your main function or API startup
test_connection()