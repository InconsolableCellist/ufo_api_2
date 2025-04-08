"""LLM interface service for interacting with language models."""

import json
import re
import time
import random
import logging
from datetime import datetime
from openai import OpenAI
from langfuse.decorators import observe
from config import API_BASE_URL, LLM_CONFIG, update_llm_config, langfuse
from templates import AGENT_SYSTEM_INSTRUCTIONS, THOUGHT_PROMPT_TEMPLATE, TOOL_DOCUMENTATION_TEMPLATE
from tools.tools import Tool, ToolRegistry

logger = logging.getLogger('agent_simulation.services.llm')

class LLMInterface:
    def __init__(self, base_url=None):
        logger.info(f"Initializing LLM interface with base URL: {API_BASE_URL}")
        self.client = None
        self.initialize_client()
        self.agent = None  # Will be set when attached to an agent
        self.tool_registry = ToolRegistry()
        self.tool_registry.llm_client = self  # Give the registry access to the LLM
        
        # Add generation counter and thinking time tracking
        self.generation_counter = 0
        self.total_thinking_time = 0.0
        
        # Register default tools from the tool registry first
        self.tool_registry.register_default_tools()
        
        # Then register any LLM-specific tools
        self._register_agent_specific_tools()
        
    def initialize_client(self):
        try:
            self.client = OpenAI(
                base_url=API_BASE_URL,
                api_key="not-needed"  # Since it's your local endpoint
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            self.client = None
    
    def _generate_completion(self, prompt, system_message):
        """Generate a completion using the OpenAI API"""
        try:
            if not self.client:
                logger.warning("OpenAI client not initialized, attempting to initialize")
                self.initialize_client()
                if not self.client:
                    logger.error("Failed to initialize OpenAI client")
                    return "Error: Could not connect to LLM API"
            
            # Increment the generation counter
            self.generation_counter += 1
            
            # Prepare the messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            
            # Log the request payload
            request_payload = {
                "model": LLM_CONFIG["model"],
                "messages": messages,
                "max_tokens": LLM_CONFIG["max_tokens"],
                "temperature": LLM_CONFIG["temperature"]
            }
            logger.info(f"Request payload: {json.dumps(request_payload, indent=2)}")
            
            # Start tracking with Langfuse
            generation = langfuse.generation(
                name="llm-completion",
                model=LLM_CONFIG["model"],
                input={
                    "messages": messages,
                    "max_tokens": LLM_CONFIG["max_tokens"], 
                    "temperature": LLM_CONFIG["temperature"]
                },
                metadata={
                    "system_message": system_message,
                    "prompt_length": len(prompt),
                    "timestamp": datetime.now().isoformat(),
                    "generation_number": self.generation_counter
                }
            )
            
            # Send the request
            logger.info("Sending request to LLM API...")
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=LLM_CONFIG["model"],
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
            
            # If we have an agent with a thought_summary_manager, and this is not an ego thought, save to summary db
            if hasattr(self, 'agent') and hasattr(self.agent, 'thought_summary_manager'):
                # Clean system message for comparison (trim whitespace)
                clean_system_msg = system_message.strip()
                
                # For debugging, log the entire system message occasionally
                if random.random() < 0.05:  # Log 5% of system messages for debugging
                    logger.debug(f"FULL SYSTEM MESSAGE: '{clean_system_msg}'")
                
                # Identify different types of messages
                is_ego_thought = any([
                    "agent's ego" in clean_system_msg.lower(),
                    "you are his ego" in clean_system_msg.lower()
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
                    "your name is simon" in clean_system_msg.lower(),
                    "name is simon" in clean_system_msg.lower(),
                    "you are simon" in clean_system_msg.lower(),
                    "thought process" in clean_system_msg.lower(),
                    "agent system instructions" in clean_system_msg.lower()
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
                    self.agent.thought_summary_manager.add_thought(result, thought_type="normal_thought")
                    logger.info(f"Added agent thought to summary database, length: {len(result)}")
                elif is_tool_response:
                    # This is a tool response, not an agent thought
                    logger.info("Skipping tool response from summary database")
                else:
                    # Log detailed message for debugging
                    logger.info(f"Skipping message - agent:{is_agent_thought}, ego:{is_ego_thought}, tool:{is_tool_response}")
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
        # Langfuse context is automatically created by the @observe decorator
        # Create a structured trace for the thought generation process
        trace = langfuse.trace(
            name="thought-generation",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "has_ego": bool(context.get("ego_thoughts", "")),
                "has_stimuli": bool(context.get("stimuli")),
                "generation_counter": self.generation_counter,
                "total_thinking_time": self.total_thinking_time
            }
        )
        
        try:
            # Debug context sizes to identify potential issues
            logger.info("Analyzing context sizes for thought generation:")
            for key, value in context.items():
                if isinstance(value, str):
                    logger.info(f"  - Context[{key}]: {len(value)} characters")
                elif isinstance(value, list):
                    logger.info(f"  - Context[{key}]: {len(value)} items")
                    # Check if any list items are extremely large
                    for i, item in enumerate(value[:5]):  # First 5 items only
                        if isinstance(item, str) and len(item) > 1000:
                            logger.warning(f"    - Large item at index {i}: {len(item)} characters")
                elif isinstance(value, dict):
                    logger.info(f"  - Context[{key}]: {len(value)} keys")
                else:
                    logger.info(f"  - Context[{key}]: {type(value)}")

            # Track generation time
            start_time = time.time()
            
            # Get ego thoughts from previous cycle if any
            ego_thoughts = context.get("ego_thoughts", "")
            
            # Get original short-term memory
            orig_short_term_memory = context.get("short_term_memory", [])
            
            # Keep a working copy of short-term memory that will be updated during the conversation
            # This ensures each response can see previous responses in the same thinking cycle
            working_short_term_memory = list(orig_short_term_memory)
            
            # Check if memories are strings or objects with content attribute
            if working_short_term_memory and isinstance(working_short_term_memory[0], str):
                recent_memories = working_short_term_memory
            else:
                recent_memories = [m.content for m in working_short_term_memory if hasattr(m, 'content')]
            
            # Get recent journal entries if available
            recent_journal_entries = []
            if hasattr(self.agent, 'journal'):
                recent_journal_entries = self.agent.journal.read_recent_entries(num_entries=10)
            
            # Generate the available tools documentation
            available_tools_docs = []
            for i, tool_doc in enumerate(self.tool_registry.list_tools(), 1):
                tool_text = TOOL_DOCUMENTATION_TEMPLATE.format(
                    index=i,
                    name=tool_doc["name"],
                    description=tool_doc["description"],
                    usage=tool_doc["usage"]
                )
                available_tools_docs.append(tool_text)
            
            available_tools_text = "\n".join(available_tools_docs)
            
            # Add journal entries to the context if available
            journal_context = ""
            if recent_journal_entries:
                journal_context = "\nRecent journal entries:\n" + "\n".join(recent_journal_entries)
            
            # Format recent tool usage
            recent_tools = self.tool_registry.get_recent_tools(10)
            if recent_tools:
                tool_entries = []
                for tool in recent_tools:
                    # Format parameters as key:value pairs
                    params_str = ", ".join([f"{k}:{v}" for k, v in tool['params'].items()])
                    # Include timestamp in a readable format
                    timestamp = datetime.fromisoformat(tool['timestamp']).strftime("%H:%M:%S")
                    tool_entries.append(f"- {timestamp} | {tool['name']}({params_str})")
                recent_tools_text = "\n".join(tool_entries)
            else:
                recent_tools_text = "No recent tool usage"
            
            # Format recent tool results
            recent_results = self.tool_registry.get_recent_results(3)
            if recent_results:
                results_entries = []
                for result in recent_results:
                    # Format the result with success/failure indicator
                    res_obj = result['result']
                    if res_obj.get('success', False):
                        output = res_obj.get('output', 'No output')
                        # Truncate long outputs
                        if len(output) > 200:
                            output = output[:200] + "..."
                        results_entries.append(f"- {result['name']}: SUCCESS - {output}")
                    else:
                        error = res_obj.get('error', 'Unknown error')
                        results_entries.append(f"- {result['name']}: FAILED - {error}")
                recent_results_text = "\n".join(results_entries)
            else:
                recent_results_text = "No recent results"
            
            # Get current goals with duration information
            goals = self.tool_registry.get_goals()
            logger.info(f"Retrieved goals in generate_thought: {goals}")
            logger.info(f"Number of short-term goals: {len(goals.get('short_term_details', []))}")
            logger.info(f"Raw short-term goals data: {goals.get('short_term_details')}")
            
            # Format short-term goals with bullet points, numbering and duration
            if goals["short_term"]:
                numbered_goals = []
                for i, goal_detail in enumerate(goals.get("short_term_details", [])):
                    goal_text = goal_detail["text"]
                    duration = goal_detail["duration"]
                    cycles = goal_detail["cycles"]
                    numbered_goals.append(f"[{i}] {goal_text} (Active for: {duration}, {cycles} cycles)")
                    logger.info(f"Formatted goal {i}: {goal_text}")
                short_term_goals = "\n".join(numbered_goals)
                logger.info(f"Formatted short-term goals: {short_term_goals}")
                logger.info(f"Total formatted goals: {len(numbered_goals)}")
            else:
                short_term_goals = "No short-term goals"
                logger.info("No short-term goals found")
                
            # Format long-term goal with emphasis and duration
            if goals["long_term"] and goals.get("long_term_details"):
                long_term_detail = goals["long_term_details"]
                duration = long_term_detail["duration"]
                cycles = long_term_detail["cycles"]
                long_term_goal = f">>> {goals['long_term']} <<< (Active for: {duration}, {cycles} cycles)"
                logger.info(f"Formatted long-term goal: {long_term_goal}")
            else:
                long_term_goal = "No long-term goal"
                logger.info("No long-term goal found")
            
            # Add generation statistics
            generation_stats = f"You've thought {self.generation_counter} times for {self.total_thinking_time:.2f}s"
            
            # Check for pending Telegram messages
            pending_messages = "No pending messages."
            if hasattr(self, 'agent') and self.agent and self.agent.telegram_bot:
                # Just check if there are unread messages, don't mark as read yet
                unread_messages = self.agent.telegram_bot.get_unread_messages()
                if unread_messages:
                    pending_messages = f"You have {len(unread_messages)} unread message(s). Use the receive_telegram tool to read them."
            
            # Format ego thoughts if there are any
            if ego_thoughts:
                # Check if these ego thoughts have already been processed
                processed_ego_marker = "[PROCESSED]"
                if processed_ego_marker in ego_thoughts:
                    logger.warning("Ego thoughts already processed - skipping formatting")
                    formatted_ego_thoughts = ""
                else:
                    # Format ego thoughts with a dramatic presentation
                    formatted_ego_thoughts = f"!!!\n***Suddenly, the following thought(s) occur to you. You try to ignore them but cannot, they echo in your mind for a full minute, completely diverting your attention before they fade, and you can think again:\n{ego_thoughts}\n***\n!!!"
                    
                    # Mark as processed to prevent re-formatting in future cycles
                    if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
                        self.agent.mind.conscious.ego_thoughts = f"{processed_ego_marker}{ego_thoughts}"
                        logger.info("Marked ego thoughts as processed")
            else:
                formatted_ego_thoughts = ""
                
            # Initialize the response and tool results
            current_response = ""
            last_tool_results = context.get("last_tool_results", [])
            all_tool_calls = []
            iteration_count = 0
            ego_thoughts_refresh_interval = 10  # Increased from 3 to 10 - Generate ego thoughts less frequently
            intermediate_ego_thoughts = ""
            max_iterations = 10  # Maximum number of iterations to prevent infinite loops
            
            # Create a local variable to track whether to show ego thoughts in this iteration
            # We only want to show them in the first iteration
            show_ego_thoughts_this_iteration = formatted_ego_thoughts != ""
            
            while iteration_count < max_iterations:
                iteration_count += 1
                # Create span for each thought iteration
                thought_span = trace.span(name=f"thought-iteration-{iteration_count}")
                
                # Logger to trace memory usage
                logger.info(f"Iteration {iteration_count} - Working with {len(recent_memories)} memories")
                
                # Check if it's time to generate intermediate ego thoughts
                if iteration_count > 1 and (iteration_count - 1) % ego_thoughts_refresh_interval == 0:
                    # Create Langfuse span for ego thoughts generation
                    ego_thoughts_span = thought_span.span(name="intermediate-ego-thoughts")
                    
                    logger.info(f"Generating intermediate ego thoughts at iteration {iteration_count}")
                    # Create updated context with current state
                    interim_context = dict(context)
                    interim_context.update({
                        "recent_memories": recent_memories,
                        "recent_response": current_response
                    })
                    
                    # Generate intermediate ego thoughts
                    intermediate_ego_thoughts = self._generate_ego_thoughts(interim_context)
                    logger.info(f"Generated intermediate ego thoughts: {intermediate_ego_thoughts[:100]}...")
                    
                    # Update Langfuse span with the generated ego thoughts
                    ego_thoughts_span.update(
                        output=intermediate_ego_thoughts[:200],
                        metadata={
                            "ego_thoughts_length": len(intermediate_ego_thoughts),
                            "iteration": iteration_count
                        }
                    )
                    ego_thoughts_span.end()
                    
                    # Format ego thoughts for the next prompt
                    if intermediate_ego_thoughts:
                        formatted_ego_thoughts = f"!!!\n***Suddenly, the following thought(s) occur to you. You try to ignore them but cannot, they echo in your mind for a full minute, completely diverting your attention before they fade, and you can think again:\n{intermediate_ego_thoughts}\n***\n!!!"
                    
                    # Store the ego thoughts for the next cycle
                    if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
                        # Mark as processed to prevent re-formatting in future cycles
                        processed_ego_marker = "[PROCESSED]"
                        self.agent.mind.conscious.ego_thoughts = f"{processed_ego_marker}{intermediate_ego_thoughts}"
                        logger.info("Marked intermediate ego thoughts as processed")
                        
                        # Also add ego thoughts to short-term memory
                        if hasattr(self.agent.mind, 'memory'):
                            memory_entry = f"[EGO THOUGHTS]: {intermediate_ego_thoughts}"
                            self.agent.mind.memory.short_term.append(memory_entry)
                            logger.info(f"Added intermediate ego thoughts to short-term memory")
                            
                            # Update recent_memories to include the ego thoughts
                            if isinstance(working_short_term_memory[0], str):
                                working_short_term_memory.append(memory_entry)
                                recent_memories = working_short_term_memory[-10:]  # Keep only last 10
                            else:
                                class MemoryEntry:
                                    def __init__(self, content):
                                        self.content = content
                                working_short_term_memory.append(MemoryEntry(memory_entry))
                                recent_memories = [m.content for m in working_short_term_memory[-10:] if hasattr(m, 'content')]
                
                # Prepare the prompt
                if last_tool_results:
                    # Format all tool results for the prompt
                    results_text = "\n".join([
                        f"Tool '{name}' result: {result.get('output', '')}"
                        for name, result in last_tool_results
                    ])
                    
                    # Get recent bug reports if available
                    recent_bug_reports = ""
                    if hasattr(self, 'tool_registry') and hasattr(self.tool_registry, 'get_recent_bug_reports'):
                        try:
                            bug_reports = self.tool_registry.get_recent_bug_reports(3)  # Get last 3 reports
                            if bug_reports and bug_reports[0] != "No bug reports yet":
                                recent_bug_reports = "\nRecent bug reports:\n" + "\n".join(bug_reports)
                        except Exception as e:
                            logger.warning(f"Error getting bug reports: {e}")
                    
                    # Only show ego thoughts in the first iteration
                    current_ego_thoughts = formatted_ego_thoughts if show_ego_thoughts_this_iteration else ""
                    
                    prompt = THOUGHT_PROMPT_TEMPLATE.format(
                        emotional_state=context.get("emotional_state", {}),
                        recent_memories=recent_memories if recent_memories else "None",
                        subconscious_thoughts=context.get("subconscious_thoughts", []),
                        stimuli=context.get("stimuli", {}),
                        current_focus=context.get("current_focus"),
                        available_tools=available_tools_text + journal_context + recent_bug_reports,  # Add bug reports to available tools
                        recent_tools=recent_tools_text,
                        recent_results=recent_results_text,
                        short_term_goals=short_term_goals,
                        long_term_goal=long_term_goal,
                        generation_stats=generation_stats,
                        ego_thoughts=current_ego_thoughts,
                        pending_messages=pending_messages
                    )
                else:
                    # Get recent bug reports if available
                    recent_bug_reports = ""
                    if hasattr(self, 'tool_registry') and hasattr(self.tool_registry, 'get_recent_bug_reports'):
                        try:
                            bug_reports = self.tool_registry.get_recent_bug_reports(3)  # Get last 3 reports
                            if bug_reports and bug_reports[0] != "No bug reports yet":
                                recent_bug_reports = "\nRecent bug reports:\n" + "\n".join(bug_reports)
                        except Exception as e:
                            logger.warning(f"Error getting bug reports: {e}")
                    
                    # Only show ego thoughts in the first iteration
                    current_ego_thoughts = formatted_ego_thoughts if show_ego_thoughts_this_iteration else ""
                    
                    prompt = THOUGHT_PROMPT_TEMPLATE.format(
                        emotional_state=context.get("emotional_state", {}),
                        recent_memories=recent_memories if recent_memories else "None",
                        subconscious_thoughts=context.get("subconscious_thoughts", []),
                        stimuli=context.get("stimuli", {}),
                        current_focus=context.get("current_focus"),
                        available_tools=available_tools_text + journal_context + recent_bug_reports,  # Add bug reports
                        recent_tools=recent_tools_text,
                        recent_results=recent_results_text,
                        short_term_goals=short_term_goals,
                        long_term_goal=long_term_goal,
                        generation_stats=generation_stats,
                        ego_thoughts=current_ego_thoughts,
                        pending_messages=pending_messages
                    )
                
                # After the first iteration, don't show ego thoughts anymore
                show_ego_thoughts_this_iteration = False
                
                # Generate the response
                response = self._generate_completion(prompt, AGENT_SYSTEM_INSTRUCTIONS)

                # Parse and handle any tool invocations
                parsed_response, tool_results = self._handle_tool_invocations(response, context)
                
                # Track tool usage in Langfuse
                for tool_name, tool_result in tool_results:
                    # Create span for each tool call
                    tool_span = thought_span.span(name=f"tool-{tool_name}")
                    
                    # Add tool details to Langfuse
                    tool_span.update(
                        input=json.dumps(self.tool_registry.get_recent_tools(1)[0]['params'] if self.tool_registry.get_recent_tools(1) else {}),
                        output=json.dumps(tool_result),
                        metadata={
                            "tool_name": tool_name,
                            "success": tool_result.get("success", False)
                        }
                    )
                    tool_span.end()
                    
                    # Keep track of all tools used
                    all_tool_calls.append({
                        "name": tool_name,
                        "result": tool_result
                    })
                
                # Add current iteration's response to the ongoing conversation context
                # Important: Update recent_memories for the next iteration to include this response
                if parsed_response:
                    # Add the parsed response to our current response
                    if current_response:
                        current_response += "\n\n"
                    current_response += parsed_response
                    
                    # Add this response to the working memory for the next iteration
                    working_short_term_memory.append(parsed_response)
                    
                    # Update recent_memories list for next iteration's context
                    if isinstance(working_short_term_memory[0], str):
                        recent_memories = working_short_term_memory[-10:]  # Keep only last 10
                    else:
                        # If we had objects with content attribute, maintain that structure
                        class MemoryEntry:
                            def __init__(self, content):
                                self.content = content
                        
                        working_short_term_memory.append(MemoryEntry(parsed_response))
                        recent_memories = [m.content for m in working_short_term_memory[-10:] if hasattr(m, 'content')]
                
                # End this thought iteration span
                thought_span.end()
                
                # If no tools were used, we're done
                if not tool_results:
                    break
                    
                # Update the last tool results for the next iteration
                last_tool_results = tool_results
                
                # Refresh recent tool usage and results for the next iteration's prompt
                recent_tools = self.tool_registry.get_recent_tools(10)
                if recent_tools:
                    tool_entries = []
                    for tool in recent_tools:
                        # Format parameters as key:value pairs
                        params_str = ", ".join([f"{k}:{v}" for k, v in tool['params'].items()])
                        # Include timestamp in a readable format
                        timestamp = datetime.fromisoformat(tool['timestamp']).strftime("%H:%M:%S")
                        tool_entries.append(f"- {timestamp} | {tool['name']}({params_str})")
                    recent_tools_text = "\n".join(tool_entries)
                else:
                    recent_tools_text = "No recent tool usage"
                
                # Update recent tool results too
                recent_results = self.tool_registry.get_recent_results(3)
                if recent_results:
                    results_entries = []
                    for result in recent_results:
                        res_obj = result['result']
                        if res_obj.get('success', False):
                            output = res_obj.get('output', 'No output')
                            if len(output) > 200:
                                output = output[:200] + "..."
                            results_entries.append(f"- {result['name']}: SUCCESS - {output}")
                        else:
                            error = res_obj.get('error', 'Unknown error')
                            results_entries.append(f"- {result['name']}: FAILED - {error}")
                    recent_results_text = "\n".join(results_entries)
                else:
                    recent_results_text = "No recent results"
                    
                # Check if we're about to reach the maximum iterations
                if iteration_count == max_iterations - 1:
                    # Add a message to inform that we're stopping due to too many iterations
                    loop_warning = "\n\n[SYSTEM: Maximum tool invocation loop reached. Forcing stop of tool chain. Please complete your thought without additional tools.]"
                    current_response += loop_warning
                    logger.warning(f"Maximum iteration limit reached ({max_iterations}). Forcing end of tool chain.")
            
            # Generate ego thoughts about this thinking cycle
            updated_context = dict(context)
            updated_context.update({
                "recent_memories": recent_memories,
                "recent_response": current_response,
                "previous_ego_thoughts": intermediate_ego_thoughts
            })
            
            # Create Langfuse span for final ego thoughts generation
            final_ego_span = trace.span(name="final-ego-thoughts")
            
            new_ego_thoughts = self._generate_ego_thoughts(updated_context)
            logger.info(f"Generated new ego thoughts: {new_ego_thoughts[:100]}...")
            
            # Update Langfuse span with the generated ego thoughts
            final_ego_span.update(
                output=new_ego_thoughts[:200],
                metadata={
                    "ego_thoughts_length": len(new_ego_thoughts),
                    "previous_ego_thoughts_length": len(intermediate_ego_thoughts) if intermediate_ego_thoughts else 0,
                    "generation_counter": self.generation_counter
                }
            )
            final_ego_span.end()
            
            # End the trace with full results
            trace.update(
                output=current_response,
                metadata={
                    "tool_calls_count": len(all_tool_calls),
                    "tools_used": [t["name"] for t in all_tool_calls],
                    "thought_length": len(current_response),
                    "iterations": iteration_count,
                    "generation_counter": self.generation_counter,
                    "total_thinking_time": self.total_thinking_time,
                    "has_ego_thoughts": bool(new_ego_thoughts)
                }
            )
            
            # Store the ego thoughts for the next cycle
            if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'conscious'):
                # Mark as processed to prevent re-formatting in future cycles
                processed_ego_marker = "[PROCESSED]"
                self.agent.mind.conscious.ego_thoughts = f"{processed_ego_marker}{new_ego_thoughts}"
                logger.info("Marked final ego thoughts as processed")
                
                # Also add ego thoughts to short-term memory
                if hasattr(self.agent.mind, 'memory'):
                    memory_entry = f"[FINAL EGO THOUGHTS]: {new_ego_thoughts}"
                    self.agent.mind.memory.short_term.append(memory_entry)
                    logger.info(f"Added final ego thoughts to short-term memory")
            
            # Track total thinking time
            end_time = time.time()
            thinking_time = end_time - start_time
            self.total_thinking_time += thinking_time
            
            # Add the final thought response to short-term memory
            if hasattr(self.agent, 'mind') and hasattr(self.agent.mind, 'memory'):
                memory_entry = f"[FINAL THOUGHT]: {current_response}"
                self.agent.mind.memory.short_term.append(memory_entry)
                logger.info(f"Added final thought response to short-term memory")
            
            # Add thought to the summary database for summarization
            if hasattr(self.agent, 'thought_summary_manager'):
                # Check if the thought contains content (not just tool invocations)
                # We want to include thoughts that contain tool invocations as part of reasoning
                if len(current_response) > 100:
                    self.agent.thought_summary_manager.add_thought(current_response, thought_type="normal_thought")
                    logger.info(f"Added thought to summary database, length: {len(current_response)}")
                else:
                    logger.info("Skipping short thought from summary database")
            
            return current_response, new_ego_thoughts
            
        except Exception as e:
            logger.error(f"Error in generate_thought: {e}", exc_info=True)
            
            # Update trace with error information
            if 'trace' in locals():
                trace.update(
                    error=str(e),
                    metadata={"error_type": "thought_generation_error"}
                )
            
            return f"Error generating thought: {str(e)}", ""

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
        
        # Tool to get goal statistics
        self.tool_registry.register_tool(Tool(
            name="get_goal_stats",
            description="Get statistics about current goals, including duration and cycles",
            function=lambda: self.tool_registry.get_goal_stats(),
            usage_example="[TOOL: get_goal_stats()]"
        ))
        
    def _set_focus(self, value):
        """Set the agent's focus"""
        # Handle None value
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
        # Handle None value
        if instruction is None:
            return {
                "success": False,
                "error": "No instruction provided. Please specify an instruction."
            }
            
        # Store the instruction for later use
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
        
        # Create a Langfuse trace for tool handling
        tools_trace = langfuse.trace(
            name="tool-invocations-handling",
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_tools_found": len(matches),
                "tool_names": [match[0] for match in matches]
            }
        )
        
        # Process all tool invocations in order
        for tool_match in matches:
            tool_name = tool_match[0]
            params_str = tool_match[1]
            
            # Create a span for this specific tool
            tool_span = tools_trace.span(name=f"tool-{tool_name}")
            
            # Parse parameters - improved parameter parsing
            params = {}
            
            logger.info(f"Parsing parameters for tool {tool_name}: '{params_str}'")
            
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
            
            # Update Langfuse with parsed parameters
            tool_span.update(
                input=json.dumps(params),
                metadata={
                    "tool_name": tool_name,
                    "raw_params_str": params_str
                }
            )
            
            try:
                # Execute the tool using the registry
                result = self.tool_registry.execute_tool(tool_name, **params)
                
                # Ensure result is in the correct format
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
                
                # Update Langfuse with successful result
                tool_span.update(
                    output=json.dumps(result),
                    metadata={
                        "success": True,
                        "output_length": len(str(result.get("output", "")))
                    }
                )
                
            except Exception as e:
                # Handle tool execution errors
                result = {
                    "success": False,
                    "error": str(e),
                    "tool": tool_name
                }
                
                # Update Langfuse with error
                tool_span.update(
                    error=str(e),
                    metadata={
                        "success": False,
                        "error_type": "tool_execution_error"
                    }
                )
                
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            
            # End the tool span
            tool_span.end()
            
            tool_results.append((tool_name, result))
            
            # Replace the tool invocation with its result in the response
            tool_invocation = f"[TOOL: {tool_name}({params_str})]"
            result_text = f"Tool '{tool_name}' result: {result.get('output', '')}"
            parsed_response = parsed_response.replace(tool_invocation, result_text)
        
        # If we have multiple tool results, format them together
        if len(tool_results) > 1:
            # Create a summary of all tool results
            results_summary = "\n".join([
                f"Tool '{name}' result: {result.get('output', '')}"
                for name, result in tool_results
            ])
            
            # Add the summary to the parsed response
            parsed_response += f"\n\nAll tool results:\n{results_summary}"
        
        # End the tools trace with final results
        tools_trace.update(
            output=json.dumps([{"tool": name, "success": result.get("success", False)} for name, result in tool_results]),
            metadata={
                "successful_tools": sum(1 for _, result in tool_results if result.get("success", False)),
                "failed_tools": sum(1 for _, result in tool_results if not result.get("success", False))
            }
        )
        
        return parsed_response, tool_results
    
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
            from templates import EGO_SYSTEM_2_INSTRUCTIONS, THOUGHT_PROMPT_TEMPLATE
            
            # Check for previous ego thoughts
            previous_ego_thoughts = context.get("previous_ego_thoughts", "")
            # Check if ego_thoughts is also in context (could cause duplication)
            existing_ego_thoughts = context.get("ego_thoughts", "")
            
            # Debug logging for ego thoughts sources
            logger.info(f"Previous ego thoughts length: {len(previous_ego_thoughts)} chars")
            logger.info(f"Existing ego thoughts length: {len(existing_ego_thoughts)} chars")
            
            # If both are present, log a warning as this could indicate duplication
            if previous_ego_thoughts and existing_ego_thoughts:
                logger.warning("Both previous_ego_thoughts and ego_thoughts found in context - potential duplication risk")
                
                # Check if they're identical
                if previous_ego_thoughts == existing_ego_thoughts:
                    logger.warning("Duplicate detected: previous_ego_thoughts and ego_thoughts are identical")
                
            # Always prefer previous_ego_thoughts if available
            if previous_ego_thoughts:
                logger.info(f"Found previous ego thoughts: {previous_ego_thoughts[:100]}...")
            elif existing_ego_thoughts:
                # If no previous_ego_thoughts but ego_thoughts exists, use that instead
                previous_ego_thoughts = existing_ego_thoughts
                logger.info(f"Using existing ego_thoughts as previous: {previous_ego_thoughts[:100]}...")
            
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
            
            # Limit subconscious thoughts to just a few key ones
            subconscious_thoughts = context.get("subconscious_thoughts", [])
            if isinstance(subconscious_thoughts, list) and len(subconscious_thoughts) > 3:
                # Just keep the 3 most important subconscious thoughts
                subconscious_thoughts = subconscious_thoughts[:3]
                logger.info("Limited subconscious thoughts to 3 for ego perspective")
            
            stimuli = context.get("stimuli", {})
            current_focus = context.get("current_focus", "Nothing in particular")
            
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
            
            # Get available tools
            available_tools_docs = []
            for i, tool_doc in enumerate(self.tool_registry.list_tools(), 1):
                tool_text = TOOL_DOCUMENTATION_TEMPLATE.format(
                    index=i,
                    name=tool_doc["name"],
                    description=tool_doc["description"],
                    usage=tool_doc["usage"]
                )
                available_tools_docs.append(tool_text)
            available_tools_text = "\n".join(available_tools_docs)
            
            # Get recent tool usage and results
            recent_tools = self.tool_registry.get_recent_tools(10)
            if recent_tools:
                tool_entries = []
                for tool in recent_tools:
                    params_str = ", ".join([f"{k}:{v}" for k, v in tool['params'].items()])
                    timestamp = datetime.fromisoformat(tool['timestamp']).strftime("%H:%M:%S")
                    tool_entries.append(f"- {timestamp} | {tool['name']}({params_str})")
                recent_tools_text = "\n".join(tool_entries)
            else:
                recent_tools_text = "No recent tool usage"
            
            recent_results = self.tool_registry.get_recent_results(3)
            if recent_results:
                results_entries = []
                for result in recent_results:
                    res_obj = result['result']
                    if res_obj.get('success', False):
                        output = res_obj.get('output', 'No output')
                        if len(output) > 200:
                            output = output[:200] + "..."
                        results_entries.append(f"- {result['name']}: SUCCESS - {output}")
                    else:
                        error = res_obj.get('error', 'Unknown error')
                        results_entries.append(f"- {result['name']}: FAILED - {error}")
                recent_results_text = "\n".join(results_entries)
            else:
                recent_results_text = "No recent results"
            
            # Also include the current thought response if available
            recent_response = context.get("recent_response", "")
            if recent_response:
                recent_memories.append(f"[MOST RECENT THOUGHT]: {recent_response}")
            
            # Format the prompt with all the information
            # Get recent bug reports if available
            recent_bug_reports = ""
            if hasattr(self, 'tool_registry') and hasattr(self.tool_registry, 'get_recent_bug_reports'):
                try:
                    bug_reports = self.tool_registry.get_recent_bug_reports(3)  # Get last 3 reports
                    if bug_reports and bug_reports[0] != "No bug reports yet":
                        recent_bug_reports = "\nRecent bug reports:\n" + "\n".join(bug_reports)
                except Exception as e:
                    logger.warning(f"Error getting bug reports for ego thoughts: {e}")
            
            ego_prompt = THOUGHT_PROMPT_TEMPLATE.format(
                emotional_state=emotional_state,
                recent_memories=recent_memories if recent_memories else "None",
                subconscious_thoughts=subconscious_thoughts,
                stimuli=stimuli,
                current_focus=current_focus,
                short_term_goals=short_term_goals,
                long_term_goal=long_term_goal,
                generation_stats=generation_stats,
                available_tools=available_tools_text + recent_bug_reports,
                recent_tools=recent_tools_text,
                recent_results=recent_results_text,
                ego_thoughts="",
                pending_messages="No pending messages."
            )
            
            # If we have previous ego thoughts, add them to the prompt
            if previous_ego_thoughts:
                ego_prompt += f"\n\nYour previous ego thoughts were:\n{previous_ego_thoughts}\n\nConsider these thoughts as you develop new insights, but don't repeat them exactly."
            
            # Generate the ego thoughts
            ego_thoughts = self._generate_completion(ego_prompt, EGO_SYSTEM_2_INSTRUCTIONS)
            
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