"""
Main entry point for the agent simulation application.
"""

import os
from agent import Agent
from simulation import SimulationController
from services.llm_interface import LLMInterface
from config import logger, initialize_system

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
        
        # Start the simulation
        try:
            controller.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            controller.shutdown()
    else:
        logger.error("System initialization failed") 