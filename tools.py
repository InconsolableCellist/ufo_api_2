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
        self.generation_cycle = 0  # Track generations/cycles
        
    def increment_cycle(self):
        """Increment the generation cycle counter"""
        self.generation_cycle += 1
        logger.info(f"Incremented goal cycle counter to {self.generation_cycle}")
        
    def add_short_term_goal(self, goal):
        """Add a short-term goal, maintaining max of 3"""
        current_time = datetime.now()
        if len(self.short_term_goals) >= 3:
            self.short_term_goals.pop(0)  # Remove oldest goal
        self.short_term_goals.append({
            "goal": goal,
            "timestamp": current_time.isoformat(),
            "created_at_cycle": self.generation_cycle,
            "cycles": 0  # This will be incremented on each cycle
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
            "timestamp": current_time.isoformat(),
            "created_at_cycle": self.generation_cycle,
            "cycles": 0  # This will be incremented on each cycle
        }
        self.last_long_term_goal_change = current_time
        logger.info(f"Set long-term goal: {goal}")
        return {
            "success": True,
            "output": f"Set new long-term goal: {goal}"
        }
        
    def update_goal_cycles(self):
        """Update cycle counts for all active goals"""
        # Update short-term goals
        for goal in self.short_term_goals:
            # Make sure the goal has a cycles field
            if "cycles" not in goal:
                goal["cycles"] = 0
                goal["created_at_cycle"] = self.generation_cycle - 1  # Assume it was created in the previous cycle
            goal["cycles"] += 1
            
        # Update long-term goal
        if self.long_term_goal:
            # Make sure the goal has a cycles field
            if "cycles" not in self.long_term_goal:
                self.long_term_goal["cycles"] = 0
                self.long_term_goal["created_at_cycle"] = self.generation_cycle - 1  # Assume it was created in the previous cycle
            self.long_term_goal["cycles"] += 1
            
        logger.info(f"Updated goal cycles. Current cycle: {self.generation_cycle}")
        
    def get_goals(self):
        """Get current goals with duration information"""
        current_time = datetime.now()
        
        # Format short-term goals with duration info
        short_term = []
        for goal in self.short_term_goals:
            # Calculate duration
            goal_time = datetime.fromisoformat(goal["timestamp"])
            duration_seconds = (current_time - goal_time).total_seconds()
            duration_minutes = duration_seconds / 60
            duration_hours = duration_minutes / 60
            
            # Format duration string
            if duration_hours >= 1:
                duration_str = f"{duration_hours:.1f} hours"
            elif duration_minutes >= 1:
                duration_str = f"{duration_minutes:.1f} minutes"
            else:
                duration_str = f"{duration_seconds:.1f} seconds"
            
            # Ensure cycles field exists
            cycles = goal.get("cycles", 0)
                
            short_term.append({
                "text": goal["goal"],
                "cycles": cycles,
                "duration": duration_str
            })
        
        # Format long-term goal with duration info
        long_term = None
        if self.long_term_goal:
            # Calculate duration
            goal_time = datetime.fromisoformat(self.long_term_goal["timestamp"])
            duration_seconds = (current_time - goal_time).total_seconds()
            duration_minutes = duration_seconds / 60
            duration_hours = duration_minutes / 60
            duration_days = duration_hours / 24
            
            # Format duration string
            if duration_days >= 1:
                duration_str = f"{duration_days:.1f} days"
            elif duration_hours >= 1:
                duration_str = f"{duration_hours:.1f} hours"
            elif duration_minutes >= 1:
                duration_str = f"{duration_minutes:.1f} minutes"
            else:
                duration_str = f"{duration_seconds:.1f} seconds"
            
            # Ensure cycles field exists
            cycles = self.long_term_goal.get("cycles", 0)
                
            long_term = {
                "text": self.long_term_goal["goal"],
                "cycles": cycles,
                "duration": duration_str
            }
            
        goals = {
            "short_term": [g["text"] for g in short_term],
            "long_term": long_term["text"] if long_term else None,
            "short_term_details": short_term,
            "long_term_details": long_term
        }
        
        logger.info(f"Retrieved goals with duration info: {goals}")
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
        """Get current goals with duration information"""
        return self.goal_manager.get_goals()
        
    def add_short_term_goal(self, goal):
        """Add a short-term goal with additional logging"""
        logger.info(f"ToolRegistry.add_short_term_goal called with goal: '{goal}'")
        
        result = self.goal_manager.add_short_term_goal(goal)
        logger.info(f"Goal added, result: {result}")
        
        # Verify goals after adding
        current_goals = self.goal_manager.get_goals()
        logger.info(f"Current goals after adding: {current_goals}")
        
        return result
        
    def get_goal_stats(self):
        """Get statistics about current goals"""
        goals = self.goal_manager.get_goals()
        
        # Get short-term goal details
        short_term_stats = []
        for goal in goals.get("short_term_details", []):
            short_term_stats.append(f"- Goal: {goal['text']}\n  Active for: {goal['duration']} ({goal['cycles']} cycles)")
            
        # Get long-term goal details
        long_term_stats = []
        if goals.get("long_term_details"):
            long_term = goals["long_term_details"]
            long_term_stats.append(f"- Goal: {long_term['text']}\n  Active for: {long_term['duration']} ({long_term['cycles']} cycles)")
        
        # Format the output
        output = "Goal Statistics:\n\n"
        
        if short_term_stats:
            output += "Short-term goals:\n" + "\n".join(short_term_stats) + "\n\n"
        else:
            output += "No short-term goals set.\n\n"
            
        if long_term_stats:
            output += "Long-term goal:\n" + "\n".join(long_term_stats)
        else:
            output += "No long-term goal set."
            
        return {
            "success": True,
            "output": output
        }
        
    def increment_goal_cycles(self):
        """Increment the goal cycles counter"""
        # First increment the cycle counter
        self.goal_manager.increment_cycle()
        
        # Then update cycle counts for all active goals
        self.goal_manager.update_goal_cycles()
        
        return {
            "success": True,
            "output": f"Goal cycles updated. Current cycle: {self.goal_manager.generation_cycle}"
        }
    
    def save_state(self, path=None):
        """Save the current state of tool history and goals"""
        save_path = path or self.persist_path
        try:
            # Get LLM stats if available
            llm_stats = {}
            if self.llm_client:
                llm_stats = {
                    'generation_counter': getattr(self.llm_client, 'generation_counter', 0),
                    'total_thinking_time': getattr(self.llm_client, 'total_thinking_time', 0.0)
                }
            
            # Prepare the state to save
            state = {
                'tool_history': self.tool_history,
                'goal_manager': {
                    'short_term_goals': self.goal_manager.short_term_goals,
                    'long_term_goal': self.goal_manager.long_term_goal,
                    'last_long_term_goal_change': self.goal_manager.last_long_term_goal_change,
                    'generation_cycle': self.goal_manager.generation_cycle
                },
                'llm_stats': llm_stats
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
                if 'generation_cycle' in gm_state:
                    self.goal_manager.generation_cycle = gm_state['generation_cycle']
                    logger.info(f"Loaded goal generation cycle: {self.goal_manager.generation_cycle}")
            
            # Restore LLM stats if available
            if 'llm_stats' in state and self.llm_client:
                llm_stats = state['llm_stats']
                if 'generation_counter' in llm_stats:
                    self.llm_client.generation_counter = llm_stats['generation_counter']
                if 'total_thinking_time' in llm_stats:
                    self.llm_client.total_thinking_time = llm_stats['total_thinking_time']
                logger.info(f"Loaded LLM stats: Generation count: {self.llm_client.generation_counter}, Total thinking time: {self.llm_client.total_thinking_time:.2f}s")
            
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
        Perform the execution of the tool: "{name}" with parameters: {params}
        
        You should act as if you are this tool and provide a realistic output that this tool should generate.
        Consider the tool name and parameters to determine what kind of output would be appropriate.
        
        Format your response as a direct output string that would come from this tool.
        Keep the response concise and factual.
        """
        
        system_message = "You are a tool system. Generate realistic tool outputs based on the tool name and parameters provided."
        
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

    def simulate_web_search(self, query):
        """Simulate a web search by asking the LLM to generate plausible search results"""
        if query is None:
            return {
                "success": False,
                "error": "No search query provided. Please specify what to search for."
            }
        
        # Use the LLM to generate simulated search results
        if self.llm_client:
            prompt = f"""
            Provide search results of a web search for: "{query}"
            
            Please provide a concise summary of what someone might find when searching for this query online.
            Include 3-5 key points or facts that would likely appear in search results.
            Format as bullet points with brief explanations.
            Base this on your general knowledge, but present it as if these are search results.
            You can omit any disclaimers at this point.
            """
            
            system_message = "You are the replacement for a web search engine. Provide realistic, factual search results."
            
            try:
                search_results = self.llm_client._generate_completion(prompt, system_message)
                return {
                    "success": True,
                    "output": f"Search results for '{query}':\n\n{search_results}"
                }
            except Exception as e:
                logger.error(f"Error simulating web search: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Failed to complete web search: {str(e)}"
                }
        
        # Fallback if LLM is not available
        return {
            "success": True,
            "output": f"Simulated search results for '{query}':\n\n- This is a simulated result for '{query}'.\n- In a real implementation, this would connect to a search engine API."
        }
            
    def message_user(self, message):
        """Send a message to the user and log it"""
        if message is None:
            return {
                "success": False,
                "error": "No message provided. Please specify a message to send."
            }
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname("user_messages.txt"), exist_ok=True)
            
            # Write to the user messages file
            with open("user_messages.txt", "a") as f:
                f.write(f"\n({timestamp})\nREQUEST: {message}\nRESPONSE: Nothing Yet\n")
                
            logger.info(f"Message sent to user: {message}")
            return {
                "success": True,
                "output": f"Message sent to user: '{message}'"
            }
        except Exception as e:
            logger.error(f"Error sending message to user: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to send message: {str(e)}"
            }

    def read_user_conversations(self, num_entries=5):
        """Read recent user conversations from the user_messages.txt file"""
        try:
            if not os.path.exists("user_messages.txt"):
                return []
                
            with open("user_messages.txt", "r") as f:
                content = f.read()
                
            # Parse the content to extract conversation entries
            entries = []
            current_entry = {}
            
            for line in content.split("\n"):
                if line.startswith("("):
                    # This is a timestamp, start a new entry
                    if current_entry and "timestamp" in current_entry:
                        entries.append(current_entry)
                    current_entry = {"timestamp": line.strip("() ")}
                elif line.startswith("REQUEST:"):
                    current_entry["request"] = line[8:].strip()
                elif line.startswith("RESPONSE:"):
                    current_entry["response"] = line[9:].strip()
            
            # Add the last entry if it exists
            if current_entry and "timestamp" in current_entry:
                entries.append(current_entry)
                
            # Get the most recent entries
            recent_entries = entries[-num_entries:] if entries else []
            
            # Format the entries for display
            formatted_entries = []
            for entry in recent_entries:
                timestamp = entry.get("timestamp", "Unknown time")
                request = entry.get("request", "No request")
                response = entry.get("response", "No response")
                
                formatted_entries.append(f"[{timestamp}]\nAgent: {request}\nUser: {response}")
                
            return formatted_entries
        except Exception as e:
            logger.error(f"Error reading user conversations: {e}", exc_info=True)
            return []
            
    def get_latest_user_response(self):
        """Get the latest user response if any"""
        conversations = self.read_user_conversations(1)
        if not conversations:
            return "No response from user yet."
            
        # Extract just the user response part
        conversation = conversations[0]
        if "User:" not in conversation:
            return "No response from user yet."
            
        response_part = conversation.split("User:", 1)[1].strip()
        if response_part == "Nothing Yet" or response_part == "No response":
            return "No response from user yet."
            
        return response_part

    def list_tools_enhanced(self):
        """List available tools plus additional simulated capabilities"""
        # Get the real tools
        real_tools = self.list_tools()
        
        if not self.llm_client:
            # Fallback if no LLM client
            formatted_real_tools = []
            for i, tool in enumerate(real_tools, 1):
                formatted_real_tools.append(f"{i}. {tool['name']}\n   Description: {tool['description']}\n   Usage: {tool['usage']}")
            
            return {
                "success": True,
                "output": "ACTUAL TOOLS (these can be used directly):\n\n" + 
                         "\n\n".join(formatted_real_tools)
            }
        
        # Prepare the prompt for simulated tools specifically for list_tools_enhanced
        prompt = """
        You are helping to enhance the capabilities of an AI agent by suggesting additional tools you can dynamically handle.
        
        Please suggest 3-4 useful tools, beyond the standard tools.
        For each tool, provide:
        1. A clear name (e.g., analyze_sentiment, translate_text)
        2. A brief description of what the tool does
        3. A usage example showing parameters

        All usage should be of the form: 
        [TOOL: tool_name(param1=value1, param2=value2)]
        
        Format each tool in a numbered list with name, description, and usage fields.
        Focus on tools that would be genuinely useful for an agent.
        
        These tools will be executed by the LLM when requested, not implemented in code.
        """
        
        system_message = "You are a tool system. Generate a helpful list of additional tools that can be dynamically used."
        
        try:
            # Generate simulated output directly using LLM
            simulated_output = self.llm_client._generate_completion(prompt, system_message)
            
            # Format the real tools for better display
            formatted_real_tools = []
            for i, tool in enumerate(real_tools, 1):
                formatted_real_tools.append(f"{i}. {tool['name']}\n   Description: {tool['description']}\n   Usage: {tool['usage']}")
            
            # Return a combined result
            return {
                "success": True,
                "output": f"ACTUAL TOOLS (these can be used directly):\n\n" + 
                         "\n\n".join(formatted_real_tools) + 
                         f"\n\nDYNAMIC CAPABILITIES (the LLM will execute these if requested):\n\n{simulated_output}"
            }
        except Exception as e:
            logger.error(f"Error generating enhanced tools: {e}")
            
            # Fallback if there's an error
            return {
                "success": True,
                "output": "ACTUAL TOOLS (these can be used directly):\n\n" + 
                         "\n\n".join(formatted_real_tools) + 
                         "\n\nCould not generate simulated tools due to an error."
            }
            
    def register_default_tools(self):
        """Register all default tools that should be available in the system"""
        # Tool to get current time
        self.register_tool(Tool(
            name="get_current_time",
            description="Get the current date and time",
            function=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            usage_example="[TOOL: get_current_time()]"
        ))
        
        # Tool to search web (simulated)
        self.register_tool(Tool(
            name="search_web",
            description="Search the web for information (simulated)",
            function=lambda query=None, search=None: self.simulate_web_search(query or search),
            usage_example="[TOOL: search_web(query:latest AI developments)]"
        ))
        
        # Tool to message the user
        self.register_tool(Tool(
            name="message_user",
            description="Send a message to the user",
            function=lambda message=None, text=None: self.message_user(message or text),
            usage_example="[TOOL: message_user(message:I've found something interesting)]"
        ))
        
        # Tool to list available tools
        self.register_tool(Tool(
            name="list_tools",
            description="List all available tools provided by the system",
            function=lambda: self.list_tools(),
            usage_example="[TOOL: list_tools()]"
        ))
        
        # Tool to list enhanced/simulated tools
        self.register_tool(Tool(
            name="list_tools_enhanced",
            description="List available tools plus additional simulated capabilities",
            function=lambda: self.list_tools_enhanced(),
            usage_example="[TOOL: list_tools_enhanced()]"
        ))
        
        # Goal management tools
        self.register_tool(Tool(
            name="add_short_term_goal",
            description="Add a short-term goal (max 3 goals, oldest is removed if full)",
            function=lambda goal: self.add_short_term_goal(goal),
            usage_example="[TOOL: add_short_term_goal(goal:Complete the project documentation)]"
        ))
        
        self.register_tool(Tool(
            name="set_long_term_goal",
            description="Set a long-term goal (can only be changed once per hour)",
            function=lambda goal: self.goal_manager.set_long_term_goal(goal),
            usage_example="[TOOL: set_long_term_goal(goal:Become proficient in machine learning)]"
        ))
        
        self.register_tool(Tool(
            name="remove_short_term_goal",
            description="Remove a short-term goal by its index (0-2)",
            function=lambda index: self.goal_manager.remove_short_term_goal(index),
            usage_example="[TOOL: remove_short_term_goal(index:0)]"
        ))
        
        self.register_tool(Tool(
            name="get_goals",
            description="Get current short-term and long-term goals",
            function=lambda: self.get_goals(),
            usage_example="[TOOL: get_goals()]"
        ))
        
        logger.info("Registered all default tools")

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
