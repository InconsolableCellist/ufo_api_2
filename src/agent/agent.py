"""Main Agent class implementation."""

import logging
import os
from datetime import datetime
import json

from tools.tools import Tool
from services import Journal, TelegramBot
from managers.processing_manager import ProcessingManager
from agent.mind import Mind
from managers.thought_summary_manager import ThoughtSummaryManager
from config import STATE_DIR

logger = logging.getLogger('agent_simulation.agent')

class Agent:
    """The main Agent class that coordinates all agent components and behaviors.
    
    The Agent class serves as the central coordinator for all agent functionality,
    managing the mind, memory, emotions, tools, and external services.
    
    Args:
        llm: The LLM interface for language model interactions
        memory_path (str, optional): Path to persist memory state. Defaults to None.
        emotion_path (str, optional): Path to persist emotional state. Defaults to None.
        tool_registry_path (str, optional): Path to persist tool registry state. Defaults to None.
        telegram_token (str, optional): Telegram bot token for messaging. Defaults to None.
        telegram_chat_id (str, optional): Telegram chat ID for messaging. Defaults to None.
        journal_path (str, optional): Path to persist journal entries. Defaults to None.
    """
    
    def __init__(self, llm, memory_path=None, emotion_path=None, 
                 tool_registry_path=None,
                 telegram_token=None, telegram_chat_id=None,
                 journal_path=None):
        self.llm = llm
        self.llm.attach_to_agent(self)  # Connect the LLM to the agent
        self.mind = Mind(self.llm, memory_path, emotion_path)
        self.physical_state = {
            "energy": 1.0,
            "fatigue": 0.0
        }
        
        # Set default paths if not provided
        if tool_registry_path is None:
            tool_registry_path = os.path.join(STATE_DIR, "tool_registry_state.pkl")
        if journal_path is None:
            journal_path = os.path.join(STATE_DIR, "agent_journal.txt")
        
        # Save paths for shutdown
        self.memory_path = memory_path
        self.emotion_path = emotion_path
        self.tool_registry_path = tool_registry_path
        
        # Initialize thought summary manager
        self.thought_summary_manager = ThoughtSummaryManager()
        
        # Initialize journal and telegram bot
        self.journal = Journal(journal_path)
        self.telegram_bot = TelegramBot(telegram_token, telegram_chat_id)
        
        # Initialize processing manager
        self.processing_manager = ProcessingManager(self)
        
        # Register agent-specific tools
        self._register_agent_tools()
        
        # Load tool registry state if available
        if tool_registry_path:
            self.llm.tool_registry.load_state(tool_registry_path)
            
        logger.info("Agent initialized successfully")
        
    def update_physical_state(self):
        """Update the agent's physical state based on emotions and vice versa."""
        # Physical state affects emotions and vice versa
        emotions = self.mind.emotion_center.get_state()
        self.physical_state["energy"] -= 0.01  # Base energy drain
        self.physical_state["energy"] += emotions["energy"] * 0.05
        self.physical_state["energy"] = max(0, min(1, self.physical_state["energy"]))
        
        # If very low energy, increase desire to rest
        if self.physical_state["energy"] < 0.2:
            self.mind.emotion_center.emotions["energy"].intensity += 0.1

    def invoke_tool(self, tool_name, **params):
        """Allow the agent to invoke tools that can affect its state.
        
        Args:
            tool_name (str): Name of the tool to invoke
            **params: Tool parameters
            
        Returns:
            str: Result of the tool invocation
        """
        if tool_name == "adjust_emotion":
            emotion_name = params.get("emotion")
            intensity_change = params.get("change", 0)
            
            if emotion_name in self.mind.emotion_center.emotions:
                emotion = self.mind.emotion_center.emotions[emotion_name]
                emotion.intensity += intensity_change
                emotion.intensity = max(0, min(1, emotion.intensity))
                return f"Adjusted {emotion_name} by {intensity_change}"
            return f"Unknown emotion: {emotion_name}"
        
        elif tool_name == "rest":
            self.physical_state["energy"] += params.get("amount", 0.1)
            self.physical_state["energy"] = min(1.0, self.physical_state["energy"])
            return "Rested and recovered some energy"
        
        return f"Unknown tool: {tool_name}"

    def _register_agent_tools(self):
        """Register tools specific to this agent."""
        # Tool to adjust emotions
        self.llm.tool_registry.register_tool(Tool(
            name="adjust_emotion",
            description="Adjust the intensity of an emotion",
            function=lambda emotion, change: self._adjust_emotion(emotion, change),
            usage_example="[TOOL: adjust_emotion(emotion:happiness, change:0.1)]"
        ))
        
        # Tool to rest and recover energy
        self.llm.tool_registry.register_tool(Tool(
            name="rest",
            description="Rest to recover energy",
            function=lambda amount=0.1: self._rest(amount),
            usage_example="[TOOL: rest(amount:0.1)]"
        ))
        
        # Tool to recall memories
        self.llm.tool_registry.register_tool(Tool(
            name="recall_memories",
            description="Recall memories related to a query or emotional state",
            function=lambda query=None, count=3: self._recall_memories(query, count),
            usage_example="[TOOL: recall_memories(query:happiness, count:3)]"
        ))
        
        # Tool to find memories related to a specific thought
        self.llm.tool_registry.register_tool(Tool(
            name="find_related_memories",
            description="Find memories related to a specific thought or concept",
            function=lambda thought=None, count=3: self._find_related_memories(thought, count),
            usage_example="[TOOL: find_related_memories(thought:artificial intelligence, count:3)]"
        ))
        
        # Tool to set subconscious focus
        self.llm.tool_registry.register_tool(Tool(
            name="set_subconscious_focus",
            description="Set a thought for the subconscious to focus on when finding related memories",
            function=lambda thought: self._set_subconscious_focus(thought),
            usage_example="[TOOL: set_subconscious_focus(thought:I want to understand more about machine learning)]"
        ))
        
        # Tool to get current emotional state
        self.llm.tool_registry.register_tool(Tool(
            name="get_emotional_state",
            description="Get a natural description of the current emotional state",
            function=lambda: self._get_emotional_state(),
            usage_example="[TOOL: get_emotional_state()]"
        ))
        
        # Tool to send telegram messages
        self.llm.tool_registry.register_tool(Tool(
            name="send_telegram",
            description="Send a message",
            function=lambda message: self._send_telegram_message(message),
            usage_example="[TOOL: send_telegram(message:This is an urgent message)]"
        ))
        
        # Tool to read pending Telegram messages
        self.llm.tool_registry.register_tool(Tool(
            name="receive_telegram",
            description="Read pending messages",
            function=lambda: self._receive_telegram_messages(),
            usage_example="[TOOL: receive_telegram()]"
        ))
        
        # Tool to write journal entries
        self.llm.tool_registry.register_tool(Tool(
            name="write_journal",
            description="Write an entry to the agent's journal for future reference",
            function=lambda entry=None, text=None, message=None, value=None: self._write_journal_entry(entry or text or message or value),
            usage_example="[TOOL: write_journal(entry:Today I learned something interesting)]"
        ))
        
        # Tool to report bugs
        self.llm.tool_registry.register_tool(Tool(
            name="report_bug",
            description="Report a bug to the agent's bug tracking system",
            function=lambda report: self._report_bug(report),
            usage_example="[TOOL: report_bug(report:The agent is experiencing a critical issue with memory recall)]"
        ))
        
        # Tool to recall research memories
        self.llm.tool_registry.register_tool(Tool(
            name="recall_research_memories",
            description="Recall memories specifically from research, optionally filtered by query",
            function=lambda query=None, count=3: self._recall_research_memories(query, count),
            usage_example="[TOOL: recall_research_memories(query:quantum computing, count:3)]"
        ))

    def _adjust_emotion(self, emotion, change):
        """Adjust an emotion's intensity."""
        if emotion in self.mind.emotion_center.emotions:
            emotion_obj = self.mind.emotion_center.emotions[emotion]
            emotion_obj.intensity += float(change)
            emotion_obj.intensity = max(0, min(1, emotion_obj.intensity))
            return f"Adjusted {emotion} by {change}, new intensity: {emotion_obj.intensity:.2f}"
        return f"Unknown emotion: {emotion}"
        
    def _rest(self, amount):
        """Rest to recover energy."""
        amount = float(amount)
        self.physical_state["energy"] += amount
        self.physical_state["energy"] = min(1.0, self.physical_state["energy"])
        return f"Rested and recovered {amount:.2f} energy. Current energy: {self.physical_state['energy']:.2f}"
        
    def _recall_memories(self, query, count):
        """Recall memories based on query or emotional state."""
        count = int(count)
        
        # If a query is provided, use semantic search
        if query:
            logger.info(f"Recalling memories with direct query: {query}")
            memories = self.mind.memory.recall(
                emotional_state=self.mind.emotion_center.get_state(),
                query=query,
                n=count
            )
        else:
            # If no query, fall back to emotional state only
            logger.info("Recalling memories based on emotional state only")
            memories = self.mind.memory.recall(
                emotional_state=self.mind.emotion_center.get_state(),
                n=count
            )
            
        return memories
        
    def _find_related_memories(self, thought, count):
        """Find memories related to a specific thought using the subconscious."""
        if not thought:
            return "A thought query must be provided"
            
        # Handle potentially very long inputs
        if len(thought) > 500:
            logger.info(f"Truncating very long thought query from {len(thought)} chars to 500 chars")
            thought = thought[:500] + "..."
            
        count = int(count)
        logger.info(f"Finding memories related to thought: '{thought[:50]}...'")
        
        # Use the subconscious to find related memories
        memories = self.mind.subconscious.find_related_memories(thought, count)
        
        if not memories:
            return "No related memories found"
            
        return memories
        
    def _set_subconscious_focus(self, thought):
        """Set a thought for the subconscious to focus on."""
        if not thought:
            return "A thought must be provided to focus on"
            
        success = self.mind.subconscious.set_focus_thought(thought)
        
        if success:
            return f"Subconscious now focusing on: '{thought[:50]}...'"
        else:
            return "Failed to set subconscious focus"

    def _send_telegram_message(self, message):
        """Send a message via Telegram."""
        if not self.telegram_bot.token:
            return "Telegram bot not configured. Message not sent."
        
        success = self.telegram_bot.send_message(message)
        if success:
            return f"Message sent successfully: '{message}'"
        else:
            return "Failed to send message. Check logs for details."

    def _receive_telegram_messages(self):
        """Read unread messages from Telegram and mark them as read."""
        if not self.telegram_bot.token:
            return {
                "success": False,
                "output": "Telegram bot not configured. Cannot receive messages."
            }
        
        try:
            # Get unread messages
            unread_messages = self.telegram_bot.get_unread_messages()
            
            if not unread_messages:
                return {
                    "success": True,
                    "output": "No new messages."
                }
            
            # Format messages for display
            formatted_messages = []
            for msg in unread_messages:
                timestamp = msg.get("timestamp", "Unknown time")
                sender = msg.get("from", "Unknown sender")
                text = msg.get("text", "(No text)")
                
                formatted_messages.append(f"[{timestamp}] {sender}: {text}")
            
            # Mark messages as read
            self.telegram_bot.mark_messages_as_read()
            
            return {
                "success": True,
                "output": "New messages:\n\n" + "\n\n".join(formatted_messages)
            }
        except Exception as e:
            logger.error(f"Error receiving Telegram messages: {e}", exc_info=True)
            return {
                "success": False,
                "output": f"Error receiving messages: {str(e)}"
            }

    def _write_journal_entry(self, entry=None, value=None):
        """Write an entry to the journal."""
        try:
            # Use entry or value parameter, whichever is provided
            content = entry if entry is not None else value
            
            # Handle various input formats that might be passed in by the LLM
            if content is None:
                content = "Empty journal entry"
                
            # Handle cases where entry is passed with quotes
            if isinstance(content, str):
                # Remove surrounding quotes if they exist
                if (content.startswith('"') and content.endswith('"')) or (content.startswith("'") and content.endswith("'")):
                    content = content[1:-1]
            
            # Convert non-string entries to string
            if not isinstance(content, str):
                content = str(content)
                
            # Log the entry type and content for debugging
            logger.info(f"Writing journal entry of type {type(content)}: '{content[:50]}...'")
            
            # Ensure we have an entry
            if not content.strip():
                content = "Blank journal entry recorded at this timestamp"
                
            # Write the entry
            success = self.journal.write_entry(content)
            
            if success:
                return f"Journal entry recorded: '{content[:50]}...'" if len(content) > 50 else f"Journal entry recorded: '{content}'"
            else:
                # If writing failed, try once more with a simplified entry
                fallback_entry = f"Journal entry (simplified): {content[:100]}"
                fallback_success = self.journal.write_entry(fallback_entry)
                
                if fallback_success:
                    return f"Journal entry recorded (simplified format): '{fallback_entry[:50]}...'"
                else:
                    return "Failed to write journal entry. Check logs for details."
                    
        except Exception as e:
            # Catch any exceptions to prevent failure
            logger.error(f"Error in _write_journal_entry: {e}", exc_info=True)
            
            # Try one last time with an error note
            try:
                error_entry = f"Journal entry attempted but encountered error: {str(e)[:100]}"
                self.journal.write_entry(error_entry)
                return "Journal entry recorded with error note."
            except:
                return "Journal entry attempted but failed. Will keep trying."

    def _get_emotional_state(self):
        """Get a natural description of the current emotional state using the LLM."""
        if not hasattr(self, 'mind'):
            return "Unable to get emotional state: agent mind not properly initialized"
            
        # Get the raw emotion values
        emotion_values = self.mind.emotion_center.get_state()
        
        # Prepare context for the LLM
        prompt = f"""
        Current emotional intensities:
        {json.dumps(emotion_values, indent=2)}
        
        Overall mood: {self.mind.emotion_center.mood:.2f} (-1 to 1 scale)
        
        Based on these emotional intensities and overall mood, provide a brief, natural description
        of the emotional state from the agent's perspective. Focus on the dominant emotions
        and their interplay. Keep the description to 2-3 sentences.
        
        Format your response as a direct first-person statement of emotional awareness.
        """
        
        system_message = "You are an AI agent's emotional awareness. Describe the emotional state naturally and introspectively."
        
        try:
            # Use the agent's LLM interface to generate the description
            description = self.llm._generate_completion(prompt, system_message)
            return description
        except Exception as e:
            logger.error(f"Error generating emotional state description: {e}", exc_info=True)
            return f"Raw emotional state: {json.dumps(emotion_values, indent=2)}"

    def _report_bug(self, report):
        """Report a bug to the agent's bug tracking system."""
        try:
            # Ensure the report is a string
            if not isinstance(report, str):
                report = str(report)
            
            # Write the report to the bug tracking file
            with open("bug_reports.txt", "a") as f:
                f.write(f"{datetime.now().isoformat()}: {report}\n")
            
            return f"Bug reported successfully: '{report[:50]}...'" if len(report) > 50 else f"Bug reported successfully: '{report}'"
        except Exception as e:
            logger.error(f"Error reporting bug: {e}", exc_info=True)
            return f"Failed to report bug: {str(e)}"

    def _recall_research_memories(self, query=None, count=3):
        """Recall memories specifically from research."""
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

    def shutdown(self):
        """Properly shutdown the agent, saving all states."""
        logger.info("Agent shutdown initiated")
        
        # Save memory
        if self.memory_path:
            self.mind.memory.save()
            logger.info(f"Memory saved to {self.memory_path}")
            
        # Save emotional state
        if self.emotion_path:
            self.mind.save_emotional_state(self.emotion_path)
            
        # Save tool registry state
        if self.tool_registry_path:
            self.llm.tool_registry.save_state(self.tool_registry_path)
            logger.info(f"Tool registry state saved to {self.tool_registry_path}")
            
        # Stop thought summarization process
        if hasattr(self, 'thought_summary_manager'):
            self.thought_summary_manager.stop_summarization()
            logger.info("Thought summarization process stopped")
            
        # Write final journal entry
        self.journal.write_entry("Agent shutdown completed. Goodbye for now.")
        
        logger.info("Agent shutdown completed")
        return True 