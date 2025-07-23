"""
Central configuration module for the application.

This module contains all global configurations, constants, and environment variables
used throughout the application. All other modules should import configuration values
from here instead of defining their own global constants.
"""

import os
import logging
import sys
import signal
import atexit
import gc
import time
import requests
import importlib
from openai import OpenAI
from colorama import Fore, Back, Style, init
from langfuse import Langfuse
from helpers import ColoredFormatter

# Initialize colorama
init(autoreset=True)


# Define state directory
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
os.makedirs(STATE_DIR, exist_ok=True)

DB_FILE = os.path.join(STATE_DIR, "llm_messages.db")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(STATE_DIR, 'agent_simulation.log'))  # File handler without colors
    ]
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('agent_simulation')
logger.addHandler(console_handler)

# API Configuration
API_HOST = "cube"
API_PORT = "5000"
API_BASE_URL = f"http://{API_HOST}:{API_PORT}/v1"

# Summary Service Configuration
SUMMARY_HOST = "mlboy"
SUMMARY_PORT = "5000"
SUMMARY_BASE_URL = f"http://{SUMMARY_HOST}:{SUMMARY_PORT}/v1"

# Langfuse Configuration
os.environ["LANGFUSE_HOST"] = "http://zen:3000"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-c039333b-d33f-44a5-a33c-5827e783f4b2"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-3c59e794-7426-49ea-b9a1-2eae0999fadf"
langfuse = Langfuse()

# LLM Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETION_MODEL = "qwen/qwen3-235b-a22b:free"

LLM_CONFIG = {
    "model": COMPLETION_MODEL,
    "max_tokens": 750,
    "temperature": 1.75,
    "use_openrouter": False,  # Set to False to use local API
    "api_base": "https://openrouter.ai/api/v1",  # OpenRouter API endpoint
    #"api_key": "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",
    "api_key": "d10b4db7e9368401cd87ba6bea5ac072",
    "local_api_base": API_BASE_URL,  # Local API endpoint
    "local_model": "local-model"  # Model name for local API
}

MEMORY_PATH = os.path.join(STATE_DIR, "agent_memory.pkl")
EMOTION_PATH = os.path.join(STATE_DIR, "emotion_state.pkl")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def update_llm_config(new_config):
    """Update LLM configuration parameters"""
    global LLM_CONFIG
    LLM_CONFIG.update(new_config)
    logger.info(f"LLM configuration updated: {LLM_CONFIG}")

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
        # Access controller global if it exists in calling context
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
        gc.collect()
        
        logger.info("Shutdown cleanup completed, exiting now")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    # Use os._exit for a more forceful exit that doesn't wait for threads
    os._exit(0)

def reload_templates():
    """Reload templates from the templates.py file"""
    import templates
    importlib.reload(templates)
    logger.info("Templates reloaded")

# Register shutdown handlers
signal.signal(signal.SIGINT, handle_shutdown)  # Ctrl+C
signal.signal(signal.SIGTERM, handle_shutdown)  # Termination signal
atexit.register(lambda: handle_shutdown() if 'controller' in globals() else None)
