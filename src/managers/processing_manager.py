"""Processing manager for handling agent processing cycles and stimuli."""

import logging
import queue
import json
from datetime import datetime
from agent.emotion import Emotion
from config import langfuse

logger = logging.getLogger('agent_simulation.managers.processing')

class ProcessingManager:
    """Manages processing cycles and stimuli for the agent.
    
    Handles:
    - Different types of processing cycles (normal, ego, emotion)
    - Stimuli queue management
    - Thought processing and emotional updates
    """
    
    def __init__(self, agent):
        """Initialize the processing manager.
        
        Args:
            agent: The agent instance to manage processing for
        """
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
        """Add raw stimuli to the queue for processing."""
        self.stimuli_queue.put(stimuli)
        
    def next_cycle(self):
        """Run the next processing cycle."""
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
        """Process raw stimuli and generate interpreted stimuli."""
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
        """Run a normal thinking cycle."""
        # Create a trace ID for this cycle
        trace_id = datetime.now().isoformat()
        
        # Check if there are any processed stimuli
        processed_stimuli = []
        try:
            while not self.processed_stimuli_queue.empty():
                processed_stimuli.append(self.processed_stimuli_queue.get(block=False))
        except queue.Empty:
            pass
            
        # Run subconscious processes
        subconscious_thoughts = self.agent.mind.subconscious.process(trace_id)
        
        # Format stimuli for consumption
        if processed_stimuli:
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
            formatted_stimuli, subconscious_thoughts, trace_id
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
        """Run an ego-focused cycle."""
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
        """Run an emotion-focused cycle - natural decay only."""
        # Apply natural decay of emotions over time
        for emotion in self.agent.mind.emotion_center.emotions.values():
            emotion.intensity *= (1 - emotion.decay_rate)
            emotion.intensity = max(0, min(1, emotion.intensity))
            
        # Recalculate mood
        emotional_state = self.agent.mind.emotion_center.get_state()
        
        # Calculate mood using the same formula as in the EmotionCenter class
        emotions = self.agent.mind.emotion_center.emotions
        # Use .get() with default Emotion objects to safely handle missing emotions
        happiness = emotions.get('happiness', Emotion('happiness')).intensity
        surprise = emotions.get('surprise', Emotion('surprise')).intensity
        focus = emotions.get('focus', Emotion('focus')).intensity
        curiosity = emotions.get('curiosity', Emotion('curiosity')).intensity
        sadness = emotions.get('sadness', Emotion('sadness')).intensity
        anger = emotions.get('anger', Emotion('anger')).intensity
        fear = emotions.get('fear', Emotion('fear')).intensity
        
        positive = happiness + surprise * 0.5 + focus * 0.3 + curiosity * 0.2
        negative = sadness + anger + fear
        mood = (positive - negative) / (positive + negative + 1e-6)
        
        # Update mood
        self.agent.mind.emotion_center.mood = mood
        
        return {
            "cycle_type": "emotion",
            "emotional_state": emotional_state,
            "mood": mood
        }
        
    def _extract_emotional_implications(self, thought):
        """Extract emotional implications from a thought."""
        implications = {}
        
        # Use the same logic as in Conscious._extract_emotional_implications
        if "happy" in thought.lower() or "joy" in thought.lower():
            implications["happiness"] = 0.1
        if "sad" in thought.lower() or "depress" in thought.lower():
            implications["sadness"] = 0.1
        if "angry" in thought.lower() or "frustrat" in thought.lower():
            implications["anger"] = 0.1
        if "fear" in thought.lower() or "afraid" in thought.lower() or "worry" in thought.lower():
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