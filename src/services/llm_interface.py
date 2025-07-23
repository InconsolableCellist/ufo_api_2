"""LLM interface service for interacting with language models."""

import json
import re
import time
import random
import logging
from datetime import datetime
from openai import OpenAI
from langfuse.decorators import observe
from config import LLM_CONFIG, update_llm_config, langfuse
from templates import AGENT_SYSTEM_INSTRUCTIONS, THOUGHT_PROMPT_TEMPLATE, TOOL_DOCUMENTATION_TEMPLATE, EGO_SYSTEM_INSTRUCTIONS, EGO_THOUGHT_PROMPT_TEMPLATE
from tools.tools import Tool, ToolRegistry
import uuid

logger = logging.getLogger('agent_simulation.services.llm')

class LLMInterface:
    def __init__(self, base_url=None):
        logger.info(f"Initializing LLM interface")
        self.client = None
        self.initialize_client()
        self.agent = None 
        self.tool_registry = ToolRegistry()
        self.tool_registry.llm_client = self 
        
        self.generation_counter = 0
        self.total_thinking_time = 0.0
        
        # Register default tools from the tool registry first
        self.tool_registry.register_default_tools()
        
        # Then register any LLM-specific tools
        self._register_agent_specific_tools()
        
    def initialize_client(self):
        try:
            if LLM_CONFIG["use_openrouter"]:
                # Initialize OpenRouter client
                self.client = OpenAI(
                    base_url=LLM_CONFIG["api_base"],
                    api_key=LLM_CONFIG["api_key"],
                    default_headers={
                        "HTTP-Referer": "https://github.com/yourusername/ufo_ai_2",
                        "X-Title": "UFO AI 2"
                    }
                )
                logger.info("OpenRouter client initialized successfully")
            else:
                # Initialize local client
                self.client = OpenAI(
                    base_url=LLM_CONFIG["local_api_base"],
                    api_key=LLM_CONFIG["api_key"]
                )
                logger.info("Local API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}", exc_info=True)
            self.client = None
    
    def _generate_completion(self, prompt, system_message):
        """Generate a completion using either OpenRouter or local API"""
        try:
            if not self.client:
                logger.warning("LLM client not initialized, attempting to initialize")
                self.initialize_client()
                if not self.client:
                    logger.error("Failed to initialize LLM client")
                    return "Error: Could not connect to LLM API"
            
            # Increment the generation counter
            self.generation_counter += 1
            
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Select model based on configuration
            model = LLM_CONFIG["model"] if LLM_CONFIG["use_openrouter"] else LLM_CONFIG["local_model"]
            
            # Log the request payload
            request_payload = {
                "model": model,
                "messages": messages,
                "max_tokens": LLM_CONFIG["max_tokens"],
                "temperature": LLM_CONFIG["temperature"]
            }
            request_id = str(uuid.uuid4())
            logger.info(f"Request payload: {json.dumps(request_payload, indent=2)}")
            
            # Use delayed import to avoid circular dependency
            from services.sqlite_db import insert_request
            insert_request(request_id, request_payload)

            # Start tracking with Langfuse
            generation = langfuse.generation(
                name="llm-completion",
                model=model,
                input={
                    "messages": messages,
                    "max_tokens": LLM_CONFIG["max_tokens"], 
                    "temperature": LLM_CONFIG["temperature"]
                },
                metadata={
                    "system_message": system_message,
                    "prompt_length": len(prompt),
                    "timestamp": datetime.now().isoformat(),
                    "generation_number": self.generation_counter,
                    "api_type": "openrouter" if LLM_CONFIG["use_openrouter"] else "local"
                }
            )
            
            # Send the request
            logger.info(f"Sending request to {'OpenRouter' if LLM_CONFIG['use_openrouter'] else 'local'} API...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=LLM_CONFIG["max_tokens"],
                temperature=LLM_CONFIG["temperature"]
            )
            
            elapsed = time.time() - start_time
            # Update total thinking time
            self.total_thinking_time += elapsed
            logger.info(f"LLM API response received in {elapsed:.2f}s. Total thinking time: {self.total_thinking_time:.2f}s")
            
            # Extract the response text
            result = response.choices[0].message.content
            # Use delayed import to avoid circular dependency
            from services.sqlite_db import update_with_response
            update_with_response(request_id, result)

            # Log the full response (without truncating)
            logger.info(f"LLM response: '{result}'")
            
            # Update the Langfuse generation with the result
            generation.end(
                output=result,
                metadata={
                    "elapsed_time": elapsed,
                    "response_time": elapsed,
                    "output_length": len(result),
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None,
                    "total_thinking_time": self.total_thinking_time,
                    "total_tokens": len(result) // 4,  # Very rough approximation
                    "generation_counter": self.generation_counter
                }
            )
            
            # Save the state to persist thinking time and generation count
            if self.tool_registry:
                self.tool_registry.save_state()
            
            # If we have an agent with a thought_summary_manager, and this is not an ego thought or emotional awareness, save to summary db
            if hasattr(self, 'agent') and hasattr(self.agent, 'thought_summary_manager'):
                # Clean system message for comparison (trim whitespace)
                clean_system_msg = system_message.strip()
                
                # For debugging, log the entire system message occasionally
                if random.random() < 0.05:  # Log 5% of system messages for debugging
                    logger.debug(f"FULL SYSTEM MESSAGE: '{clean_system_msg}'")
                
                # Identify different types of messages
                is_ego_thought = any([
                    "your name is simon and you are his ego" in clean_system_msg.lower()
                ])
                
                # Identify pure tool responses (not agent thoughts that include tool invocations)
                is_tool_response = any([
                    "search engine" in clean_system_msg.lower(),
                    "research analyst" in clean_system_msg.lower(),
                    "web_scrape" in prompt and "response" in prompt.lower(),
                    "search_web" in prompt and "response" in prompt.lower()
                ])
                
                # Identify agent thought process messages
                is_agent_thought = any([
                    "your name is simon and this is your thought process" in clean_system_msg.lower(),
                ])
                
                is_emotional_awareness = any([
                    "emotional awareness" in clean_system_msg.lower()
                ])

                # As a fallback, check if it looks like AGENT_SYSTEM_INSTRUCTIONS
                if not is_agent_thought and len(clean_system_msg) > 100:
                    # Check for key phrases in the default agent system instructions
                    agent_instruction_markers = [
                        "high-agency",
                        "introspective",
                        "emotional state",
                        "your responses should be natural",
                        "your personality"
                    ]
                    
                    # Count how many markers are present
                    marker_count = sum(1 for marker in agent_instruction_markers 
                                      if marker in clean_system_msg.lower())
                    
                    # If at least 2 markers are present, it's likely the agent instructions
                    is_agent_thought = marker_count >= 2
                
                if is_agent_thought and len(result) > 100:
                    # This is the agent's main thought process
                    # Pass the request_id to the thought summary manager to use when storing the summary
                    self.agent.thought_summary_manager.add_thought(result, thought_type="normal_thought", request_id=request_id)
                    logger.info(f"Added agent thought to summary database, length: {len(result)}, request_id: {request_id}")
                elif is_tool_response:
                    # This is a tool response, not an agent thought
                    logger.info("Skipping tool response from summary database")
                elif is_emotional_awareness:
                    # This is emotional awareness, not an agent thought
                    logger.info("Skipping emotional awareness from summary database")
                else:
                    # Log detailed message for debugging
                    logger.info(f"Skipping message - agent:{is_agent_thought}, ego:{is_ego_thought}, tool:{is_tool_response}, emotional:{is_emotional_awareness}")
                    logger.debug(f"System message (first 100 chars): '{clean_system_msg[:100]}'")
                    
                    # To debug issues, log the first part of the result too
                    logger.debug(f"Result preview: '{result[:100]}...'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}", exc_info=True)
            
            # Log error in Langfuse if it was created
            if 'generation' in locals():
                generation.end(
                    error=str(e),
                    metadata={"error_type": "llm_api_error"}
                )
                
            return f"Error occurred while generating thought: {str(e)}"

    @observe()
    def generate_thought(self, context):
        # Create a structured trace for the thought generation process
        trace = langfuse.trace(
            name="thought-generation",
            input=json.dumps(context)
        )
        
        try:
            # Get the original short-term memory
            if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'memory'):
                recent_memories = list(self.agent.mind.memory.short_term)
                # Reverse to show most recent first
                recent_memories.reverse()
                
                # Format memories with timestamps if available
                formatted_memories = []
                for memory in recent_memories:
                    if isinstance(memory, dict) and 'timestamp' in memory:
                        timestamp = memory['timestamp']
                        content = memory.get('content', '')
                        formatted_memories.append(f"[{timestamp}] {content}")
                    else:
                        formatted_memories.append(str(memory))
            else:
                # Fall back to context memories if direct memory access not available
                recent_memories = context.get("recent_memories", [])
                formatted_memories = [str(m) for m in recent_memories]
            
            recent_memories = "\n".join(formatted_memories)
            
            # Get long-term memories using vector search
            long_term_memories = []
            
            # Use short-term memories as queries
            for memory in formatted_memories:
                # Remove timestamps for better search
                clean_memory = re.sub(r'\[\d{2}:\d{2}:\d{2}\]\s*', '', memory)
                retrieved = self.agent.mind.memory.recall(clean_memory, n=2)
                if retrieved:
                    long_term_memories.extend(retrieved)
            
            # Use goals as additional queries
            goals = self.tool_registry.get_goals()
            if goals.get("short_term"):
                for goal in goals["short_term"]:
                    retrieved = self.agent.mind.memory.recall(goal, n=2)
                    if retrieved:
                        long_term_memories.extend(retrieved)
            
            if goals.get("long_term"):
                retrieved = self.agent.mind.memory.recall(goals["long_term"], n=3)
                if retrieved:
                    long_term_memories.extend(retrieved)
            
            # Remove duplicates and format long-term memories
            seen = set()
            formatted_long_term = []
            for memory in long_term_memories:
                if memory not in seen:
                    seen.add(memory)
                    formatted_long_term.append(memory)
            
            long_term_memories = "\n".join(formatted_long_term) if formatted_long_term else "No relevant long-term memories found."
            
            # Get recent journal entries
            journal_context = ""
            if hasattr(self.agent, 'thought_summary_manager'):
                recent_entries = self.agent.thought_summary_manager.get_recent_entries(3)
                if recent_entries:
                    journal_context = "\nRecent journal entries:\n" + "\n".join(recent_entries)
            
            # Format available tools documentation
            available_tools = self.tool_registry.get_available_tools()
            available_tools_text = "\nAvailable tools:\n" + "\n".join([
                f"- {name}: {tool['description']}" 
                for name, tool in available_tools.items()
            ])
            
            # Get recent tool usage
            recent_tools = self.tool_registry.get_recent_tools(3)
            recent_tools_text = "\nRecent tools used:\n" + "\n".join([
                f"- {name}" for name in recent_tools
            ]) if recent_tools else "\nNo recent tools used."
            
            # Get recent tool results
            recent_results = self.tool_registry.get_recent_results(3)
            recent_results_text = "\nRecent tool results:\n" + "\n".join([
                f"- {result}" for result in recent_results
            ]) if recent_results else "\nNo recent tool results."
            
            # Format goals
            short_term_goals = "\n".join(goals.get("short_term", [])) if goals.get("short_term") else "No short-term goals"
            long_term_goal = goals.get("long_term", "No long-term goal")
            
            # Get generation stats
            generation_stats = f"Thought {self.generation_counter} times for {self.total_thinking_time:.2f}s"
            
            # Format ego thoughts
            ego_thoughts = context.get("ego_thoughts", [])
            if isinstance(ego_thoughts, str):
                # If it's already a string, use it directly
                formatted_ego_thoughts = ego_thoughts
            else:
                # If it's a list, join them with newlines
                formatted_ego_thoughts = "\n".join([
                    f"- {thought}" for thought in ego_thoughts
                ]) if ego_thoughts else "No recent ego thoughts"
            
            # Get pending messages
            pending_messages = context.get("pending_messages", [])
            pending_messages = "\n".join([
                f"- {msg}" for msg in pending_messages
            ]) if pending_messages else "No pending messages"
            
            # Generate the response
            if context.get("is_ego_thought"):
                prompt = EGO_THOUGHT_PROMPT_TEMPLATE.format(
                    emotional_state=context.get("emotional_state", {}),
                    recent_memories=recent_memories if recent_memories else "None",
                    short_term_goals=short_term_goals,
                    long_term_goal=long_term_goal,
                    generation_stats=generation_stats
                )
                response = self._generate_completion(prompt, EGO_SYSTEM_INSTRUCTIONS)
            else:
                prompt = THOUGHT_PROMPT_TEMPLATE.format(
                    emotional_state=context.get("emotional_state", {}),
                    recent_memories=recent_memories if recent_memories else "None",
                    long_term_memories=long_term_memories,
                    subconscious_thoughts=context.get("subconscious_thoughts", []),
                    stimuli=context.get("stimuli", {}),
                    current_focus=context.get("current_focus"),
                    available_tools=available_tools_text + journal_context,
                    recent_tools=recent_tools_text,
                    recent_results=recent_results_text,
                    short_term_goals=short_term_goals,
                    long_term_goal=long_term_goal,
                    pending_messages=pending_messages,
                    generation_stats=generation_stats,
                    ego_thoughts=formatted_ego_thoughts
                )
                response = self._generate_completion(prompt, AGENT_SYSTEM_INSTRUCTIONS)
            
            # Ensure response is a string
            if isinstance(response, tuple):
                response = response[0] if response else "Error: Empty response"
            elif not isinstance(response, str):
                response = str(response)
            
            # Handle tool invocations if any are present in the response
            if "[TOOL:" in response:
                logger.info("Tool invocations detected in response, processing...")
                response, tool_results = self._handle_tool_invocations(response, context)
                logger.info(f"Processed {len(tool_results)} tool invocations")
            
            # Update the trace with the response
            trace.update(output=response)
            return response
            
        except Exception as e:
            trace.update(error=str(e))
            logger.error(f"Error in generate_thought: {e}", exc_info=True)
            return f"Error in thought generation: {str(e)}"

    # Registers tools with the tool_registry that affect LLM state rather than act upon LLM output
    # [set_focus, system_instruction, get_generation_stats]
    def _register_agent_specific_tools(self):
        """Register tools specific to this agent"""
        # Tool to set focus - LLM specific since it affects LLM context
        self.tool_registry.register_tool(Tool(
            name="set_focus",
            description="Set the current focus of attention",
            function=lambda value=None, focus=None: self._set_focus(value or focus),
            usage_example="[TOOL: set_focus(data analysis)]"
        ))
        
        # Tool for system instructions - LLM specific since it affects LLM context
        self.tool_registry.register_tool(Tool(
            name="system_instruction",
            description="Add a system instruction to be followed during thinking",
            function=lambda instruction=None, value=None: self._add_system_instruction(instruction or value),
            usage_example="[TOOL: system_instruction(Think more creatively)]"
        ))
        
        # Tool to get generation stats
        self.tool_registry.register_tool(Tool(
            name="get_generation_stats",
            description="Get statistics about LLM generations, including count and total thinking time",
            function=self._get_generation_stats,
            usage_example="[TOOL: get_generation_stats()]"
        ))

        self.tool_registry.register_tool(Tool(
            name="get_generation_stats",
            description="Get statistics about LLM generations, including count and total thinking time",
            function=self._get_generation_stats,
            usage_example="[TOOL: get_generation_stats()]"
        ))
        
        
    def _set_focus(self, value):
        """Set the agent's focus"""
        if value is None:
            return {
                "success": False,
                "error": "No focus value provided. Please specify a focus."
            }
            
        if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
            self.agent.mind.conscious.current_focus = value
            return f"Focus set to: {value}"
        return "Unable to set focus: agent mind not properly initialized"
        
    def _add_system_instruction(self, instruction):
        """Add a system instruction to be followed"""
        if instruction is None:
            return {
                "success": False,
                "error": "No instruction provided. Please specify an instruction."
            }
            
        if not hasattr(self, 'system_instructions'):
            self.system_instructions = []
        self.system_instructions.append(instruction)
        return f"System instruction added: {instruction}"
    
    def _get_generation_stats(self):
        """Get statistics about generations"""
        stats = {
            "generation_counter": self.generation_counter,
            "total_thinking_time": self.total_thinking_time,
            "avg_thinking_time": self.total_thinking_time / max(1, self.generation_counter)
        }
        
        return {
            "success": True,
            "output": f"Generation statistics:\n- Total generations: {stats['generation_counter']}\n- Total thinking time: {stats['total_thinking_time']:.2f}s\n- Average thinking time: {stats['avg_thinking_time']:.2f}s per generation"
        }
    
    def _format_result(self, result):
        """Format the result to prevent truncation of complex objects"""
        if isinstance(result, (dict, list, tuple, set)):
            try:
                import json
                return json.dumps(result, indent=2, default=str)
            except:
                pass
        return str(result)

    def _handle_tool_invocations(self, response, context):
        """Parse and handle tool invocations in the response"""
        # Updated regex to handle quoted parameters and nested tool invocations
        tool_pattern = r'\[TOOL:\s*(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)\]'
        matches = re.findall(tool_pattern, response)
        
        if not matches:
            return response, []
        
        tool_results = []
        parsed_response = response
        
        tools_trace = langfuse.trace(
            name="tool-invocations-handling",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_tools_found": len(matches),
                "tool_names": [match[0] for match in matches]
            }
        )
        
        for tool_match in matches:
            tool_name = tool_match[0]
            params_str = tool_match[1]
            
            # Langfuse span
            tool_span = tools_trace.span(name=f"tool-{tool_name}")
            
            params = {}
            
            logger.info(f"Parsing parameters for tool {tool_name}: '{params_str}'")
            
            params_str = self.parse_tool_params(tool_name, params_str, params)
            
            tool_span.update(
                input=json.dumps(params),
                metadata={
                    "tool_name": tool_name,
                    "raw_params_str": params_str
                }
            )
            
            try:
                result = self.tool_registry.execute_tool(tool_name, **params)
                
                if isinstance(result, dict):
                    if "success" not in result:
                        result = {
                            "success": True,
                            "output": self._format_result(result)
                        }
                else:
                    result = {
                        "success": True,
                        "output": self._format_result(result)
                    }
                
                tool_span.update(
                    output=json.dumps(result),
                    metadata={
                        "success": True,
                        "output_length": len(str(result.get("output", "")))
                    }
                )
                
            except Exception as e:
                result = {
                    "success": False,
                    "error": str(e),
                    "tool": tool_name
                }

                tool_span.update(
                    error=str(e),
                    metadata={
                        "success": False,
                        "error_type": "tool_execution_error"
                    }
                )
                
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            
            tool_span.end()
            
            tool_results.append((tool_name, result))
            
            # Replace the tool invocation with its result in the response
            tool_invocation = f"[TOOL: {tool_name}({params_str})]"
            result_text = f"Tool '{tool_name}' result: {result.get('output', '')}"
            parsed_response = parsed_response.replace(tool_invocation, result_text)
        
        # If we have multiple tool results, format them together
        if len(tool_results) > 1:
            results_summary = "\n".join([
                f"Tool '{name}' result: {result.get('output', '')}"
                for name, result in tool_results
            ])
            
            parsed_response += f"\n\nAll tool results:\n{results_summary}"
        
        tools_trace.update(
            output=json.dumps([{"tool": name, "success": result.get("success", False)} for name, result in tool_results]),
            metadata={
                "successful_tools": sum(1 for _, result in tool_results if result.get("success", False)),
                "failed_tools": sum(1 for _, result in tool_results if not result.get("success", False))
            }
        )
        
        return parsed_response, tool_results

    # Takes a string of parameters and returns a dictionary of parameters
    def parse_tool_params(self, tool_name, params_str, params):
        if params_str:
            # Handle parameters with quoted values that may contain nested tool invocations
            if '=' in params_str:
                    # Match key=value pairs where value can be quoted and contain nested content
                param_pairs = []
                # First try to extract parameters with quoted values
                quoted_params = re.findall(r'(\w+)=(["\'])((?:(?!\2).|\\\2)*)\2', params_str)
                for key, quote, value in quoted_params:
                    params[key] = value
                    # Mark this part as processed by replacing it in the params_str
                    params_str = params_str.replace(f"{key}={quote}{value}{quote}", "", 1)
                    
                # Then process any remaining key=value pairs without quotes
                remaining_params = re.findall(r'(\w+)=([^,"\'][^,]*)', params_str)
                for key, value in remaining_params:
                    key = key.strip()
                    value = value.strip()
                    params[key] = value
                
            # If we still don't have parameters and there are colons, try key:value format
            if not params and ':' in params_str:
                # Handle key:value format with possible quoted values
                param_pairs = re.findall(r'([^,:]+?):([^,]+?)(?:,|$)', params_str)
                    
                for key, value in param_pairs:
                    key = key.strip()
                    value = value.strip()
                        
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                        
                    # Try to convert value to appropriate type
                    try:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif '.' in value and value.replace('.', '', 1).isdigit():
                            value = float(value)
                        elif value.isdigit():
                            value = int(value)
                    except (ValueError, AttributeError):
                        # Keep as string if conversion fails
                        pass
                        
                    params[key] = value
                
            # If we still don't have parameters, treat the whole string as a single value
            if not params:
                params["value"] = params_str.strip()
            
        logger.info(f"Final parsed parameters for {tool_name}: {params}")
        return params_str
    
    def _execute_tool(self, tool_name, params, context):
        """Execute a tool and return the result"""
        result = self.tool_registry.execute_tool(tool_name, **params)
        
        # Format the result for display
        if isinstance(result, dict):
            if result.get("success", False):
                return self._format_result(result.get("output", ""))
            else:
                return f"Error: {result.get('error', 'Unknown error')}"
        
        return self._format_result(result)

    def attach_to_agent(self, agent):
        """Attach this LLM interface to an agent"""
        self.agent = agent

    def _generate_ego_thoughts(self, context):
        """Generate ego thoughts as a higher-level perspective on the agent's state and actions"""
        try:
            logger.info("Generating ego thoughts...")
            
            # Get the templates
            from templates import EGO_SYSTEM_INSTRUCTIONS
            
            emotional_state = context.get("emotional_state", {})
            
            # Get more recent memories (10 instead of the default 5)
            if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'memory'):
                # Retrieve 10 memories for ego's broader perspective
                recent_memories = self.agent.mind.memory.recall(emotional_state, n=10)
                if isinstance(recent_memories, list) and recent_memories and hasattr(recent_memories[0], 'content'):
                    recent_memories = [m.content for m in recent_memories if hasattr(m, 'content')]
                logger.info(f"Retrieved {len(recent_memories)} memories for ego perspective")
            else:
                # Fall back to context memories if direct memory access not available
                recent_memories = context.get("recent_memories", [])
                if isinstance(recent_memories, list) and recent_memories and hasattr(recent_memories[0], 'content'):
                    recent_memories = [m.content for m in recent_memories if hasattr(m, 'content')]
            
            # Get goals with duration information
            goals = self.tool_registry.get_goals()
            logger.info(f"Retrieved goals in _generate_ego_thoughts: {goals}")
            logger.info(f"Number of short-term goals in ego thoughts: {len(goals.get('short_term_details', []))}")
            
            # Format short-term goals with duration
            if goals.get("short_term_details"):
                short_term_goals = []
                for i, goal_detail in enumerate(goals["short_term_details"]):
                    text = goal_detail["text"]
                    duration = goal_detail["duration"]
                    cycles = goal_detail.get("cycles", 0)
                    short_term_goals.append(f"[{i}] {text} (Active for: {duration}, {cycles} cycles)")
                    logger.info(f"Ego thoughts - formatted goal {i}: {text}")
                short_term_goals = "\n".join(short_term_goals)
                logger.info(f"Ego thoughts - total formatted goals: {len(short_term_goals.split('\n'))}")
            else:
                short_term_goals = "No short-term goals"
                logger.info("Ego thoughts - no short-term goals found")
            
            # Format long-term goal with duration
            if goals.get("long_term_details"):
                long_term_detail = goals["long_term_details"]
                text = long_term_detail["text"]
                duration = long_term_detail["duration"]
                cycles = long_term_detail.get("cycles", 0)
                long_term_goal = f"{text} (Active for: {duration}, {cycles} cycles)"
            else:
                long_term_goal = "No long-term goal"
            
            # Get generation stats
            generation_stats = f"Thought {self.generation_counter} times for {self.total_thinking_time:.2f}s"
            
            ego_prompt = EGO_THOUGHT_PROMPT_TEMPLATE.format(
                emotional_state=emotional_state,
                recent_memories=recent_memories if recent_memories else "None",
                short_term_goals=short_term_goals,
                long_term_goal=long_term_goal,
                generation_stats=generation_stats
            )
            
            # Generate the ego thoughts
            ego_thoughts = self._generate_completion(ego_prompt, EGO_SYSTEM_INSTRUCTIONS)
            
            # Log and return the ego thoughts
            logger.info(f"Ego thoughts generated: {ego_thoughts[:100]}...")
            return ego_thoughts
            
        except Exception as e:
            logger.error(f"Error generating ego thoughts: {e}", exc_info=True)
            return ""

    def _recall_research_memories(self, query=None, count=3):
        """Recall memories specifically from research"""
        try:
            # Convert count to integer if it's a string
            if isinstance(count, str):
                try:
                    count = int(count)
                except ValueError:
                    count = 3
                    
            # Get research memories
            memories = self.mind.memory.recall_research(query, count)
            
            if not memories:
                if query:
                    return {
                        "success": True,
                        "output": f"No research memories found related to '{query}'. Try using [TOOL: deepdive_research(query:{query})] to perform research on this topic."
                    }
                else:
                    return {
                        "success": True,
                        "output": "No research memories found. Try using [TOOL: deepdive_research(query:your topic)] to perform research."
                    }
                    
            # Format the result
            result = "Research memories:\n\n"
            for i, memory in enumerate(memories, 1):
                result += f"{i}. {memory}\n\n"
                
            result += "To perform new research, use: [TOOL: deepdive_research(query:your topic)]"
            
            return {
                "success": True,
                "output": result
            }
            
        except Exception as e:
            logger.error(f"Error recalling research memories: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to recall research memories: {str(e)}"
            }


def update_llm_config(new_config):
    """Update the LLM configuration.
    
    Args:
        new_config: New configuration values
        
    Returns:
        dict: Updated configuration
    """
    global LLM_CONFIG
    
    # Update only the provided values
    for key, value in new_config.items():
        if key in LLM_CONFIG:
            LLM_CONFIG[key] = value
            
    logger.info(f"Updated LLM config: {LLM_CONFIG}")
    return LLM_CONFIG 
