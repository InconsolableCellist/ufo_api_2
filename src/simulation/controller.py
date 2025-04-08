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
    
    TODO: Enhance simulation control:
    - Add simulation speed control
    - Implement pause/resume functionality
    - Add simulation state persistence
    - Implement better error recovery
    - Add simulation metrics collection
    """
    
    def __init__(self, agent, time_step=1.0):
        """Initialize the simulation controller.
        
        Args:
            agent: The agent to simulate
            time_step (float): Time step size in simulation units (default: 1.0)
        """
        self.agent = agent
        self.time_step = time_step
        self.current_time = 0
        self.stimuli_sources = []
        self.is_running = False
        self.start_time = None
        self.cycle_count = 0
        
    def add_stimuli_source(self, source):
        """Add a source of stimuli to the simulation.
        
        Args:
            source: Object with get_stimuli() method that returns dict of stimuli
        """
        self.stimuli_sources.append(source)
        logger.info(f"Added stimuli source: {source.__class__.__name__}")
        
    def run_step(self):
        """Run a single simulation step.
        
        Returns:
            dict: Step results including time and agent state
        """
        try:
            # Collect stimuli from all sources
            stimuli = {}
            for source in self.stimuli_sources:
                try:
                    source_stimuli = source.get_stimuli()
                    if source_stimuli:
                        stimuli.update(source_stimuli)
                except Exception as e:
                    logger.error(f"Error getting stimuli from {source.__class__.__name__}: {e}")
            
            # Process agent step
            result = self.agent.mind.process_step(stimuli)
            
            # Update physical state
            self.agent.update_physical_state()
            
            # Advance time
            self.current_time += self.time_step
            self.cycle_count += 1
            
            # Log step completion
            logger.info(f"Completed simulation step {self.cycle_count} at time {self.current_time:.2f}")
            
            return {
                "time": self.current_time,
                "cycle": self.cycle_count,
                "agent_state": {
                    "physical": self.agent.physical_state,
                    "mental": result
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