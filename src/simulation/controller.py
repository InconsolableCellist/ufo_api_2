"""Simulation controller for managing agent execution."""

import logging
import time
from datetime import datetime

logger = logging.getLogger('agent_simulation.simulation.controller')

class SimulationController:
    """Controls and manages agent simulation execution.
    
    The SimulationController handles:
    - Running simulation steps
    - Managing simulation time
    - Coordinating stimuli sources
    - Processing agent state updates
    
    The thinking loop follows this structure:
    1. Subconscious runs and obtains long-term memories
    2. Conscious runs with context of last tool results
    3. Tool invocations are handled
    4. Ego cycle runs every nth loop
    5. Summary manager processes results
    """
    
    def __init__(self, agent, time_step=1.0, ego_cycle_frequency=10):
        """Initialize the simulation controller.
        
        Args:
            agent: The agent to simulate
            time_step (float): Time step size in simulation units (default: 1.0)
            ego_cycle_frequency (int): How often to run ego cycle (default: 5)
        """
        self.agent = agent
        self.time_step = time_step
        self.current_time = 0
        self.stimuli_sources = []
        self.is_running = False
        self.start_time = None
        self.cycle_count = 0
        self.ego_cycle_frequency = ego_cycle_frequency
        self.last_tool_results = None
        
    def add_stimuli_source(self, source):
        """Add a source of stimuli to the simulation.
        
        Args:
            source: Object with get_stimuli() method that returns dict of stimuli
        """
        self.stimuli_sources.append(source)
        logger.info(f"Added stimuli source: {source.__class__.__name__}")
        
    def run_step(self):
        """Run a single simulation step following the thinking loop structure.
        
        Returns:
            dict: Step results including time and agent state
        """
        try:
            # 1. Collect stimuli from all sources
            logger.info("Execution step 1: Collecting stimuli from all sources")
            stimuli = {}
            for source in self.stimuli_sources:
                try:
                    source_stimuli = source.get_stimuli()
                    if source_stimuli:
                        stimuli.update(source_stimuli)
                except Exception as e:
                    logger.error(f"Error getting stimuli from {source.__class__.__name__}: {e}")
            
            # Get current emotional state once for the entire step
            current_emotional_state = self.agent.mind.emotion_center.get_state()
            
            # 2. Run subconscious process
            # Get recent short-term memories for context
            logger.info("Execution step 2: Running subconscious process")
            recent_memories = list(self.agent.mind.memory.short_term)[-5:]  # Last 5 memories
            subconscious_result = self.agent.mind.subconscious.process(
                trace_id=f"step_{self.cycle_count}_{datetime.now().isoformat()}",
                recent_memories=recent_memories,
                emotional_state=current_emotional_state
            )
            
            # 3. Run conscious process
            logger.info("Execution step 3: Running conscious process")
            conscious_result = self.agent.mind.conscious.think(
                stimuli=stimuli,
                subconscious_thoughts=subconscious_result,
                last_tool_results=self.last_tool_results,
                emotional_state=current_emotional_state,
                trace_id=f"step_{self.cycle_count}_{datetime.now().isoformat()}"
            )
            
            # 4. Handle tool invocations if any
            logger.info("Execution step 4: Handling tool invocations")
            # Tool invocations are now handled internally within the LLM interface
            # during the conscious.think() process, so we don't need separate handling here
            # The tool results are already embedded in the conscious_result string
            # However, we can capture recent tool results for context in future cycles
            recent_tool_results = self.agent.llm.tool_registry.get_recent_results(3)
            self.last_tool_results = recent_tool_results if recent_tool_results else None
            
            # 5. Run ego cycle if it's time
            logger.info("Execution step 5: Running ego cycle")
            ego_thoughts = None
            if self.cycle_count % self.ego_cycle_frequency == 0:
                ego_thoughts = self.agent.mind.conscious.run_ego_cycle(
                    recent_memories=recent_memories,
                    emotional_state=current_emotional_state
                )
            
            # 6. Update physical state
            logger.info("Execution step 6: Updating physical state")
            self.agent.update_physical_state()
            
            # 7. Advance time and cycle count
            logger.info("Execution step 7: Advancing time and cycle count")
            self.current_time += self.time_step
            self.cycle_count += 1
            
            # 8. Let summary manager process results
            logger.info("Execution step 8: Letting summary manager process results")
            if hasattr(self.agent, 'thought_summary_manager'):
                # Add conscious result to summary database if it's a string
                if isinstance(conscious_result, str):
                    self.agent.thought_summary_manager.add_thought(
                        conscious_result,
                        thought_type="normal_thought"
                    )
                
                # Add ego thoughts if they exist
                if ego_thoughts:
                    self.agent.thought_summary_manager.add_thought(
                        ego_thoughts,
                        thought_type="ego_thought"
                    )
            
            # Log step completion
            logger.info(f"Completed simulation step {self.cycle_count} at time {self.current_time:.2f}")
            
            return {
                "time": self.current_time,
                "cycle": self.cycle_count,
                "agent_state": {
                    "physical": self.agent.physical_state,
                    "mental": {
                        "subconscious": subconscious_result,
                        "conscious": conscious_result,
                        "ego": ego_thoughts,
                        "tool_results": self.last_tool_results
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in simulation step: {e}", exc_info=True)
            return {
                "time": self.current_time,
                "cycle": self.cycle_count,
                "error": str(e)
            }
            
    def start(self):
        """Start the simulation."""
        if self.is_running:
            logger.warning("Simulation is already running")
            return False
            
        self.is_running = True
        self.start_time = time.time()
        logger.info("Simulation started")
        return True
        
    def stop(self):
        """Stop the simulation."""
        if not self.is_running:
            logger.warning("Simulation is not running")
            return False
            
        self.is_running = False
        duration = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Simulation stopped after {duration:.2f}s ({self.cycle_count} cycles)")
        return True
        
    def get_status(self):
        """Get current simulation status.
        
        Returns:
            dict: Current simulation status
        """
        duration = time.time() - self.start_time if self.start_time and self.is_running else 0
        return {
            "running": self.is_running,
            "current_time": self.current_time,
            "cycle_count": self.cycle_count,
            "real_duration": duration,
            "stimuli_sources": len(self.stimuli_sources)
        }
        
    def shutdown(self):
        """Properly shutdown the simulation.
        
        Returns:
            bool: Success status
        """
        try:
            logger.info("Simulation shutdown initiated")
            
            # Stop simulation if running
            if self.is_running:
                self.stop()
            
            # Shutdown agent
            if hasattr(self.agent, 'shutdown'):
                self.agent.shutdown()
            
            logger.info("Simulation shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during simulation shutdown: {e}", exc_info=True)
            return False 