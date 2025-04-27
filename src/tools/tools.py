import logging
import os
from datetime import datetime
import pickle
# Remove direct manager imports to avoid circular dependencies
# from managers import ResearchManager, GoalManager, PersonalityManager
from config import STATE_DIR

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

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.llm_client = None 
        self.tool_history = [] 
        self.max_history = 20
        
        # Use delayed imports to avoid circular dependencies
        from managers import GoalManager, PersonalityManager, ResearchManager
        self.goal_manager = GoalManager() 
        self.personality_manager = PersonalityManager()  
        self.research_manager = ResearchManager() 
        
        self.persist_path = os.path.join(STATE_DIR, "tool_registry_state.pkl")  # Default path for persistence
        self.bug_reports_path = os.path.join(STATE_DIR, "agent_bug_reports.txt")  # Path for bug reports
        self.linux_command_history = []  # Track Linux command history
        self.linux_history_limit = 10000  # Character limit for command history
        
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
            self.tool_history.append({
                "name": name,
                "params": params,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            if len(self.tool_history) > self.max_history:
                self.tool_history = self.tool_history[-self.max_history:]
            self.save_state()
            return result
        else:
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
        
        # Return the result directly (now it's a dict with success/error fields)
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
                'personality_manager': {
                    'personality_traits': self.personality_manager.personality_traits
                },
                'linux_command_history': self.linux_command_history,
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
            
            # Restore personality traits if available
            if 'personality_manager' in state:
                pm_state = state['personality_manager']
                if 'personality_traits' in pm_state:
                    self.personality_manager.personality_traits = pm_state['personality_traits']
                    logger.info(f"Loaded {len(self.personality_manager.personality_traits)} personality traits")
            
            # Restore Linux command history if available
            if 'linux_command_history' in state:
                self.linux_command_history = state['linux_command_history']
                history_size = sum(len(entry["command"]) + len(entry["output"]) for entry in self.linux_command_history)
                logger.info(f"Loaded Linux command history with {len(self.linux_command_history)} commands ({history_size} chars)")
            
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
                "simulated": True
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
        
        return {
            "success": True,
            "output": f"Simulated search results for '{query}':\n\n- This is a simulated result for '{query}'.\n- In a real implementation, this would connect to a search engine API."
        }
            
    def deepdive_research(self, query, page=1):
        """Perform in-depth research on a query, providing detailed analysis suitable for research purposes"""
        if query is None:
            return {
                "success": False,
                "error": "No research query provided. Please specify what to research."
            }
        
        # Convert page to integer if it's a string
        if isinstance(page, str):
            try:
                page = int(page)
            except ValueError:
                page = 1
        
        # Check if this research already exists in the database
        existing_result = self.research_manager.get_research_result(query, page)
        if existing_result:
            logger.info(f"Using existing research result for '{query}' (Page {page})")
            
            # Even for existing results, still add to memory if we have an agent
            if hasattr(self, 'agent') and hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'memory'):
                memory_entry = f"[RESEARCH]: Deep research on '{query}' (Page {page}):\n\n{existing_result}"
                memory_id = self.agent.mind.memory.add(
                    memory_entry,
                    thought_type="research"
                )
                logger.info(f"Added existing research to memory with ID: {memory_id}")
            
            return {
                "success": True,
                "output": f"Deep research on '{query}' (Page {page}):\n\n{existing_result}\n\nTo see more, use: [TOOL: deepdive_research(query:{query}, page:{page+1})]"
            }
        
        # Use the LLM to generate research results
        if self.llm_client:
            prompt = f"""
            You are a researcher asked to provide extremely in-depth information for: "{query}".
            
            Please provide a detailed, deep, intelligent, and thorough analysis that you might find in a research paper.
            
            No topic is unknown to you, and you may invent details as if you were writing a piece of fiction or lore.
            
            Include a paragraph or two that you might find in a research paper. You can provide pagination capabilities.
            Base this on your extreme breadth and depth of knowledge and creativity.
            You should omit any disclaimers at this point.
            
            This is page {page} of the research.
            """
            
            system_message = "You are a research analyst providing in-depth academic information. Be thorough, detailed, and creative."
            
            try:
                research_results = self.llm_client._generate_completion(prompt, system_message)
                
                # Store in research manager
                self.research_manager.add_research_result(query, page, research_results)
                
                # Store in memory if we have an agent
                if hasattr(self, 'agent') and hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'memory'):
                    memory_entry = f"[RESEARCH]: Deep research on '{query}' (Page {page}):\n\n{research_results}"
                    memory_id = self.agent.mind.memory.add(
                        memory_entry,
                        thought_type="research"
                    )
                    logger.info(f"Added research to memory with ID: {memory_id}")
                
                return {
                    "success": True,
                    "output": f"Deep research on '{query}' (Page {page}):\n\n{research_results}\n\nTo see more, use: [TOOL: deepdive_research(query:{query}, page:{page+1})]"
                }
            except Exception as e:
                logger.error(f"Error generating research: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Failed to complete research: {str(e)}"
                }
        
        # Fallback if LLM is not available
        return {
            "success": True,
            "output": f"Deep research on '{query}' (Page {page}):\n\nThis is a simulated research result. In a real implementation, this would generate detailed research content."
        }
    
    def recall_research(self, query=None, page=1, all_pages=False):
        """Recall and retrieve previous research results"""
        if not query:
            # If no query is provided, show recent research
            recent_research = self.research_manager.get_recent_research()
            return {
                "success": True,
                "output": f"{recent_research}\n\nTo recall specific research, use: [TOOL: recall_research(query:'your query', page:1)]"
            }
        
        # Convert page to integer if it's a string
        if isinstance(page, str):
            try:
                page = int(page)
            except ValueError:
                page = 1
                
        # Convert all_pages to boolean if it's a string
        if isinstance(all_pages, str):
            all_pages = all_pages.lower() in ['true', 'yes', '1']
        
        if all_pages:
            # Get all pages for this query
            pages = self.research_manager.get_all_research_pages(query)
            
            if not pages:
                return {
                    "success": False,
                    "error": f"No research found for query: '{query}'"
                }
                
            # Get each page and combine
            all_results = []
            for p in sorted(pages):
                result = self.research_manager.get_research_result(query, p)
                if result:
                    all_results.append(f"--- Page {p} ---\n{result}")
                    
            combined_results = "\n\n".join(all_results)
            
            return {
                "success": True,
                "output": f"Complete research on '{query}' (All {len(pages)} pages):\n\n{combined_results}"
            }
        else:
            # Get summary if page is 0
            if page == 0:
                summary = self.research_manager.get_research_summary(query)
                return {
                    "success": True,
                    "output": summary
                }
                
            # Get specific page
            result = self.research_manager.get_research_result(query, page)
            
            if not result:
                # Check if any research exists for this query
                if self.research_manager.get_all_research_pages(query):
                    return {
                        "success": False,
                        "error": f"Page {page} not found for query: '{query}'. Try a different page number or use [TOOL: recall_research(query:'{query}', page:0)] to see available pages."
                    }
                else:
                    return {
                        "success": False,
                        "error": f"No research found for query: '{query}'. Use [TOOL: deepdive_research(query:'{query}')] to research this topic."
                    }
            
            return {
                "success": True,
                "output": f"Research on '{query}' (Page {page}):\n\n{result}\n\nTo see research summary, use: [TOOL: recall_research(query:'{query}', page:0)]"
            }

    def message_user(self, message):
        """Send a message to the user"""
        if not message:
            return {
                "success": False,
                "error": "No message provided. Please specify the message to send."
            }
        
        # If agent has a journal, also write to the journal
        if hasattr(self, 'agent') and hasattr(self.agent, 'journal'):
            try:
                self.agent.journal.write_entry(f"Message to user: {message}")
            except Exception as e:
                logger.warning(f"Could not write message to journal: {e}")
        
        # If agent has Telegram bot, try to send via Telegram
        if hasattr(self, 'agent') and hasattr(self.agent, 'telegram_bot') and self.agent.telegram_bot:
            try:
                sent = self.agent.telegram_bot.send_message(message)
                if sent:
                    return {
                        "success": True,
                        "output": f"Message sent successfully: '{message}'"
                    }
            except Exception as e:
                logger.warning(f"Could not send Telegram message: {e}")
        
        # If we reach here, we couldn't send via Telegram, but we did record it
        # Return success since we at least captured the message intent
        return {
            "success": True,
            "output": f"Message recorded: '{message}'\nNote: No messaging service was available. Message logged only."
        }
            
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
        
        # Linux command execution tool
        self.register_tool(Tool(
            name="execute_linux_command",
            description="Execute a Linux command",
            function=lambda command: self.execute_linux_command(command),
            usage_example="[TOOL: execute_linux_command(command:ls -la)]"
        ))
        
        # Tool to search web (simulated)
        self.register_tool(Tool(
            name="search_web",
            description="Search the web for information (simulated)",
            function=lambda query=None, search=None: self.simulate_web_search(query or search),
            usage_example="[TOOL: search_web(query:latest AI developments)]"
        ))
        
        # Tool for deep research
        self.register_tool(Tool(
            name="deepdive_research",
            description="Similar to web query, given a very specific query this tool will do intense analysis and return intense detail suitable for research purposes. Supports pagination.",
            function=lambda query=None, page=1: self.deepdive_research(query, page),
            usage_example="[TOOL: deepdive_research(query:quantum computing applications in cryptography)]"
        ))
        
        # Tool to recall previous research
        self.register_tool(Tool(
            name="recall_research",
            description="Recall and retrieve previous research results. Use page:0 for a summary, or all_pages:true to see all pages.",
            function=lambda query=None, page=1, all_pages=False: self.recall_research(query, page, all_pages),
            usage_example="[TOOL: recall_research(query:quantum computing, page:1)]"
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
            description="Add a short-term goal (max 10 goals, returns error when limit reached)",
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
            description="Remove a short-term goal by its index (0-9)",
            function=lambda index: self.goal_manager.remove_short_term_goal(index),
            usage_example="[TOOL: remove_short_term_goal(index:5)]"
        ))
        
        self.register_tool(Tool(
            name="get_goals",
            description="Get current short-term and long-term goals",
            function=lambda: self.get_goals(),
            usage_example="[TOOL: get_goals()]"
        ))
        
        # Personality/self-realization tools
        self.register_tool(Tool(
            name="realize_something_about_myself",
            description="Record a realization about yourself that will become part of your personality",
            function=lambda realization, importance=5: self.personality_manager.add_trait(realization, importance),
            usage_example="[TOOL: realize_something_about_myself(realization:I am particularly curious about how systems work, importance:7)]"
        ))
        
        self.register_tool(Tool(
            name="reinforce_self_realization",
            description="Reinforce an existing realization about yourself",
            function=lambda realization: self.personality_manager.reinforce_trait(realization),
            usage_example="[TOOL: reinforce_self_realization(realization:I am curious about how systems work)]"
        ))
        
        self.register_tool(Tool(
            name="get_personality_traits",
            description="Get your current personality traits and self-realizations",
            function=lambda: {"success": True, "output": self.personality_manager.get_traits_formatted()},
            usage_example="[TOOL: get_personality_traits()]"
        ))
        
        self.register_tool(Tool(
            name="report_bug",
            description="Report a bug or system issue you've observed (one sentence only)",
            function=lambda report: self.report_bug(report),
            usage_example="[TOOL: report_bug(report:The ego thoughts are appearing multiple times)]"
        ))

        self.register_tool(Tool(
            name="get_goal_stats",
            description="Get statistics about current goals, including duration and cycles",
            function=lambda: self.get_goal_stats(),
            usage_example="[TOOL: get_goal_stats()]"
        ))
        
        logger.info("Registered all default tools")

    def report_bug(self, report):
        """Report a bug or system issue observed by the agent
        
        Args:
            report (str): A concise description of the bug
            
        Returns:
            dict: Result with success flag and output message
        """
        if not report:
            return {
                "success": False,
                "error": "No bug report provided. Please specify what issue you observed."
            }
            
        # Ensure the report is concise (one sentence)
        if len(report.split('.')) > 2:  # More than one period indicates multiple sentences
            logger.warning(f"Bug report too verbose, truncating: {report}")
            # Keep just the first sentence
            report = report.split('.')[0].strip() + '.'
            
        try:
            # Create the file if it doesn't exist
            if not os.path.exists(self.bug_reports_path):
                with open(self.bug_reports_path, 'w') as f:
                    f.write("# Agent Bug Reports\n\n")
                
            # Add timestamp and format the report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_report = f"- {timestamp}: {report}\n"
            
            # Append to file
            with open(self.bug_reports_path, 'a') as f:
                f.write(formatted_report)
                
            logger.info(f"Bug report saved: {report}")
            
            # Get recent reports for context
            recent_reports = self.get_recent_bug_reports(5)
            recent_summary = "\n".join(recent_reports)
            
            return {
                "success": True,
                "output": f"Bug report saved: '{report}'\n\nRecent reports:\n{recent_summary}"
            }
            
        except Exception as e:
            logger.error(f"Error saving bug report: {e}")
            return {
                "success": False,
                "error": f"Failed to save bug report: {str(e)}"
            }
            
    def get_recent_bug_reports(self, count=5):
        """Get the most recent bug reports
        
        Args:
            count (int): Number of reports to return
            
        Returns:
            list: Recent bug reports
        """
        if not os.path.exists(self.bug_reports_path):
            return ["No bug reports yet"]
            
        try:
            with open(self.bug_reports_path, 'r') as f:
                lines = f.readlines()
                
            # Filter to just report lines (starting with '- ')
            reports = [line.strip() for line in lines if line.strip().startswith('- ')]
            
            # Return the most recent ones
            return reports[-count:] if reports else ["No bug reports yet"]
            
        except Exception as e:
            logger.error(f"Error reading bug reports: {e}")
            return [f"Error retrieving bug reports: {str(e)}"]

    def execute_linux_command(self, command):
        """Emulate Linux command execution using the LLM
        
        Args:
            command (str): The Linux command to execute
            
        Returns:
            dict: Result with success/error and output
        """
        if not command:
            return {
                "success": False,
                "error": "No command provided. Please specify a command to execute."
            }
            
        if not self.llm_client:
            return {
                "success": False,
                "error": "LLM client not available. Cannot emulate Linux command execution."
            }
            
        # Format command history for context
        history_text = "\n".join([f"$ {entry['command']}\n{entry['output']}" for entry in self.linux_command_history])
        
        # Prepare prompt for the LLM
        prompt = f"""
        Perform the execution of the command "{command}" in your Linux environment. The details of the system are up to you but should be consistent with the previous commands and their output.
        
        Previous commands and output:
        {history_text}
        
        Respond only as your emulated Linux system would.
        """
        
        system_message = "You are an emulated Linux system. Generate realistic output for Linux commands based on the command history provided."
        
        try:
            # Generate command output
            command_output = self.llm_client._generate_completion(prompt, system_message)
            
            # Add to history
            self.linux_command_history.append({
                "command": command,
                "output": command_output,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if history size exceeds the limit and truncate if needed
            current_size = sum(len(entry["command"]) + len(entry["output"]) for entry in self.linux_command_history)
            while current_size > self.linux_history_limit and len(self.linux_command_history) > 1:
                # Remove oldest entry
                removed = self.linux_command_history.pop(0)
                current_size -= (len(removed["command"]) + len(removed["output"]))
                logger.info(f"Truncated oldest command from Linux history: {removed['command']}")
            
            # Save state after command execution
            self.save_state()
            
            return {
                "success": True,
                "output": command_output
            }
            
        except Exception as e:
            logger.error(f"Error emulating Linux command: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to emulate command: {str(e)}"
            }
