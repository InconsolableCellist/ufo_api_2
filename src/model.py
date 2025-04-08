from openai import OpenAI
import logging
import time
import sys
import requests
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from colorama import Fore, Back, Style, init
import importlib
import atexit
import signal
import os
import gc
from agent import Agent
from simulation import SimulationController
from services import LLMInterface
from helpers import ColoredFormatter



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
