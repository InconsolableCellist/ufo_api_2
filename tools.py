import logging
import json
import os
from datetime import datetime
import pickle

# Configure logging
logger = logging.getLogger('agent_simulation.tools')

class Tool:
    def __init__(self, name, description, function, usage_example=None):
        self.name = name
        self.description = description
        self.function = function
        self.usage_example = usage_example or f"[TOOL: {name}()]"
        
    def execute(self, **params):
        """Execute the tool with the given parameters"""
        # ANSI escape code for yellow highlighting
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        
        try:
            # Get the function's parameter names
            import inspect
            func_params = inspect.signature(self.function).parameters
            
            # Filter out unrecognized parameters
            filtered_params = {k: v for k, v in params.items() if k in func_params}
            
            # Log tool invocation with yellow highlighting
            logger.info(f"{YELLOW}Executing tool: {self.name} with params: {filtered_params}{RESET}")
            
            result = self.function(**filtered_params)
            
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
            
            # Log the result with yellow highlighting
            logger.info(f"{YELLOW}Tool {self.name} completed with result: {result}{RESET}")
            
            return result
        except Exception as e:
            # Log errors with yellow highlighting
            logger.error(f"{YELLOW}Error executing tool {self.name}: {e}{RESET}", exc_info=True)
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

class GoalManager:
    def __init__(self):
        self.short_term_goals = []  # Max 3 goals
        self.long_term_goal = None
        self.last_long_term_goal_change = None
        self.goal_change_cooldown = 3600  # 1 hour in seconds
        
    def add_short_term_goal(self, goal):
        """Add a short-term goal, maintaining max of 3"""
        if len(self.short_term_goals) >= 3:
            self.short_term_goals.pop(0)  # Remove oldest goal
        self.short_term_goals.append({
            "goal": goal,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Added short-term goal: {goal}. Current goals: {[g['goal'] for g in self.short_term_goals]}")
        return f"Added short-term goal: {goal}"
        
    def set_long_term_goal(self, goal):
        """Set a long-term goal with cooldown"""
        current_time = datetime.now()
        
        # Check cooldown
        if self.last_long_term_goal_change:
            time_since_last = (current_time - self.last_long_term_goal_change).total_seconds()
            if time_since_last < self.goal_change_cooldown:
                remaining = self.goal_change_cooldown - time_since_last
                return {
                    "success": False,
                    "error": f"Cannot change long-term goal yet. Please wait {remaining:.1f} seconds."
                }
        
        self.long_term_goal = {
            "goal": goal,
            "timestamp": current_time.isoformat()
        }
        self.last_long_term_goal_change = current_time
        logger.info(f"Set long-term goal: {goal}")
        return {
            "success": True,
            "output": f"Set new long-term goal: {goal}"
        }
        
    def get_goals(self):
        """Get current goals"""
        goals = {
            "short_term": [g["goal"] for g in self.short_term_goals],
            "long_term": self.long_term_goal["goal"] if self.long_term_goal else None
        }
        logger.info(f"Retrieved goals: {goals}")
        return goals
        
    def remove_short_term_goal(self, index):
        """Remove a short-term goal by index"""
        if 0 <= index < len(self.short_term_goals):
            removed = self.short_term_goals.pop(index)
            logger.info(f"Removed goal at index {index}: {removed['goal']}. Remaining goals: {[g['goal'] for g in self.short_term_goals]}")
            return f"Removed short-term goal: {removed['goal']}"
        return "Invalid goal index"

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.llm_client = None  # Will be set later
        self.tool_history = []  # Track last 20 tool invocations
        self.max_history = 20
        self.goal_manager = GoalManager()  # Add goal manager
        self.persist_path = "tool_registry_state.pkl"  # Default path for persistence
        
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
            result = tool.execute(**params)
            # Add to history
            self.tool_history.append({
                "name": name,
                "params": params,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last max_history entries
            if len(self.tool_history) > self.max_history:
                self.tool_history = self.tool_history[-self.max_history:]
            # Save state after tool execution
            self.save_state()
            return result
        else:
            # Tool not found, create error result and add to history
            error_result = {
                "tool": name,
                "success": False,
                "error": "Unknown tool and no LLM client available for simulation"
            }
            # Add to history
            self.tool_history.append({
                "name": name,
                "params": params,
                "result": error_result,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last max_history entries
            if len(self.tool_history) > self.max_history:
                self.tool_history = self.tool_history[-self.max_history:]
            # Save state after tool execution
            self.save_state()
            # Tool not found, use LLM to suggest alternatives
            return self._handle_unknown_tool(name, params)
            
    def get_recent_tools(self, count=10):
        """Get the last N tool invocations"""
        return self.tool_history[-count:] if self.tool_history else []
        
    def get_recent_results(self, count=3):
        """Get the last N tool results"""
        recent = self.tool_history[-count:] if self.tool_history else []
        return [{"name": entry["name"], "result": entry["result"]} for entry in recent]
        
    def get_goals(self):
        """Get current goals"""
        return self.goal_manager.get_goals()
    
    def save_state(self, path=None):
        """Save the current state of tool history and goals"""
        save_path = path or self.persist_path
        try:
            # Prepare the state to save
            state = {
                'tool_history': self.tool_history,
                'goal_manager': {
                    'short_term_goals': self.goal_manager.short_term_goals,
                    'long_term_goal': self.goal_manager.long_term_goal,
                    'last_long_term_goal_change': self.goal_manager.last_long_term_goal_change
                }
            }
            
            # Save to file
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Tool registry state saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving tool registry state: {e}", exc_info=True)
            return False
    
    def load_state(self, path=None):
        """Load the saved state of tool history and goals"""
        load_path = path or self.persist_path
        if not os.path.exists(load_path):
            logger.warning(f"No saved state found at {load_path}")
            return False
            
        try:
            # Load from file
            with open(load_path, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            if 'tool_history' in state:
                self.tool_history = state['tool_history']
                
            if 'goal_manager' in state:
                gm_state = state['goal_manager']
                if 'short_term_goals' in gm_state:
                    self.goal_manager.short_term_goals = gm_state['short_term_goals']
                if 'long_term_goal' in gm_state:
                    self.goal_manager.long_term_goal = gm_state['long_term_goal']
                if 'last_long_term_goal_change' in gm_state:
                    self.goal_manager.last_long_term_goal_change = gm_state['last_long_term_goal_change']
            
            logger.info(f"Tool registry state loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading tool registry state: {e}", exc_info=True)
            return False
            
    def _handle_unknown_tool(self, name, params):
        """Handle unknown tool requests by simulating their behavior using the LLM"""
        if not self.llm_client:
            return {
                "tool": name,
                "success": False,
                "error": "Unknown tool and no LLM client available for simulation"
            }
        
        # Prepare context for the LLM
        prompt = f"""
        Simulate the execution of tool: "{name}" with parameters: {params}
        
        You should act as if you are this tool and provide a realistic output that this tool might generate.
        Consider the tool name and parameters to determine what kind of output would be appropriate.
        
        Format your response as a direct output string that would come from this tool.
        Keep the response concise and factual.
        """
        
        system_message = "You are a tool simulation system. Generate realistic tool outputs based on the tool name and parameters provided."
        
        try:
            simulated_output = self.llm_client._generate_completion(prompt, system_message)
            
            return {
                "tool": name,
                "success": True,
                "output": simulated_output,
                "simulated": True  # Flag to indicate this was a simulation
            }
            
        except Exception as e:
            logger.error(f"Error simulating tool: {e}", exc_info=True)
            return {
                "tool": name,
                "success": False,
                "error": f"Failed to simulate tool: {str(e)}"
            }

    # Add simulation methods for common expected tools
    def _simulate_search(self, params):
        """Simulate an internet search"""
        query = params.get("query", "")
        if not query:
            return {
                "success": False,
                "error": "Search query is required"
            }
        
        # Use the LLM to generate a plausible search result
        if self.llm_client:
            prompt = f"""
            Simulate the results of an internet search for: "{query}"
            
            Provide a brief, factual summary of what might be found online about this topic.
            Include 2-3 key points that would likely appear in search results.
            """
            
            system_message = "You are a helpful assistant simulating internet search results. Be factual and concise."
            
            try:
                search_result = self.llm_client._generate_completion(prompt, system_message)
                return {
                    "success": True,
                    "output": f"Simulated search results for '{query}':\n\n{search_result}"
                }
            except Exception as e:
                logger.error(f"Error simulating search: {e}")
        
        # Fallback if LLM is not available or fails
        return {
            "success": True,
            "output": f"Simulated search results for '{query}':\n\n- This is a simulated result as the search_internet tool is not actually implemented.\n- The system is pretending to search for information about '{query}'.\n- In a real implementation, this would connect to a search engine API."
        }

    def _simulate_calculation(self, params):
        """Simulate a calculation"""
        expression = params.get("expression", "")
        if not expression:
            return {
                "success": False,
                "error": "Calculation expression is required"
            }
        
        try:
            # Very basic calculation - in production you'd want more safety checks
            result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round, "max": max, "min": min})
            return {
                "success": True,
                "output": f"Result of '{expression}' = {result}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Could not calculate '{expression}': {str(e)}"
            }

    def _simulate_translation(self, params):
        """Simulate a translation"""
        text = params.get("text", "")
        target_language = params.get("to", "English")
        
        if not text:
            return {
                "success": False,
                "error": "Text to translate is required"
            }
        
        # Use the LLM to simulate translation
        if self.llm_client:
            prompt = f"""
            Translate the following text to {target_language}:
            "{text}"
            
            Provide only the translated text.
            """
            
            system_message = "You are a helpful translation assistant. Provide accurate translations."
            
            try:
                translation = self.llm_client._generate_completion(prompt, system_message)
                return {
                    "success": True,
                    "output": f"Translation to {target_language}: {translation}"
                }
            except Exception as e:
                logger.error(f"Error simulating translation: {e}")
        
        # Fallback
        return {
            "success": True,
            "output": f"Simulated translation to {target_language}:\n[This is a placeholder for a translation of '{text}' as the translation tool is not actually implemented.]"
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
            
    def read_recent_entries(self, num_entries=3):
        """Read the most recent entries from the journal"""
        try:
            if not os.path.exists(self.file_path):
                return []
                
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
                
            # Find all entry markers (## timestamp)
            entry_markers = []
            for i, line in enumerate(lines):
                if line.startswith('## '):
                    entry_markers.append(i)
                    
            if not entry_markers:
                return []
                
            # Get the last num_entries markers
            recent_markers = entry_markers[-num_entries:]
            entries = []
            
            for marker in recent_markers:
                # Get the timestamp
                timestamp = lines[marker].strip()[3:]  # Remove '## '
                # Get the entry content (next line until next marker or end)
                content = lines[marker + 1].strip()
                entries.append(f"{timestamp}: {content}")
                
            return entries
            
        except Exception as e:
            logger.error(f"Failed to read journal entries: {e}")
            return []

class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.bot = None
        self.last_message_time = None  # Track when the last message was sent
        self.rate_limit_seconds = 3600  # 1 hour in seconds
        
        if self.token:
            try:
                # Try to import telegram module
                try:
                    import telegram
                except ImportError:
                    logger.warning("python-telegram-bot package not installed. Telegram functionality will be disabled.")
                    return
                    
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

class EmotionCenter:
    def __init__(self):
        self.emotions = {
            'happiness': Emotion('happiness', decay_rate=0.05),
            'sadness': Emotion('sadness', decay_rate=0.03),
            'anger': Emotion('anger', decay_rate=0.08),
            'fear': Emotion('fear', decay_rate=0.1),
            'surprise': Emotion('surprise', decay_rate=0.15),
            'disgust': Emotion('disgust', decay_rate=0.07),
            'energy': Emotion('energy', decay_rate=0.02),
        }
        self.mood = 0.5  # Overall mood from -1 (negative) to 1 (positive)
        self.llm_client = None  # Will be set by the agent

def get_state(self):
    """Get a descriptive summary of the current emotional state"""
    # Get the raw emotion values
    emotion_values = {name: emotion.intensity for name, emotion in self.emotions.items()}
    
    # If no LLM client is available, return just the raw values
    if not hasattr(self, 'llm_client'):
        return emotion_values
        
    # Prepare context for the LLM
    prompt = f"""
    Current emotional intensities:
    {json.dumps(emotion_values, indent=2)}
    
    Overall mood: {self.mood:.2f} (-1 to 1 scale)
    
    Based on these emotional intensities and overall mood, provide a brief, natural description
    of the emotional state from the agent's perspective. Focus on the dominant emotions
    and their interplay. Keep the description to 2-3 sentences.
    
    Format your response as a direct first-person statement of emotional awareness.
    """
    
    system_message = "You are an AI agent's emotional awareness. Describe the emotional state naturally and introspectively."
    
    try:
        description = self.llm_client._generate_completion(prompt, system_message)
        
        return {
            "raw_emotions": emotion_values,
            "mood": self.mood,
            "description": description
        }
    except Exception as e:
        logger.error(f"Error generating emotional state description: {e}", exc_info=True)
        return emotion_values