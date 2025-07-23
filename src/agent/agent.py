"""Main Agent class implementation."""

import logging
import os
from datetime import datetime
import json

from tools.tools import Tool
from services import Journal, TelegramBot
from agent.mind import Mind
from managers.thought_summary_manager import ThoughtSummaryManager
from config import STATE_DIR
from .emotion import EmotionCenter
from .motivation import MotivationCenter
from .cognition.subconscious import Subconscious
from .cognition.conscious import Conscious
from .cognition.memory import Memory

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
        
        # Initialize core components
        self.memory = Memory(persist_path=memory_path)
        self.emotion_center = EmotionCenter()
        self.emotion_center.llm_client = llm
        self.motivation_center = MotivationCenter()
        
        # Initialize processing components
        self.subconscious = Subconscious(self.memory, self.emotion_center)
        self.conscious = Conscious(
            self.memory, 
            self.emotion_center, 
            self.subconscious,
            llm
        )
        
        # Initialize the mind with all components
        self.mind = Mind(
            memory=self.memory,
            emotion_center=self.emotion_center,
            motivation_center=self.motivation_center,
            subconscious=self.subconscious,
            conscious=self.conscious
        )
        
        # Load emotional state if path provided
        if emotion_path and os.path.exists(emotion_path):
            self.mind.load_emotional_state(emotion_path)
            
        # Initialize physical state
        self.physical_state = {
            "energy": 1.0,
            "focus": 1.0,
            "health": 1.0
        }
        
        # Cache for emotional state descriptions
        self._emotional_state_cache = {
            "description": None,
            "last_emotions": None,
            "last_mood": None,
            "timestamp": None
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
        if telegram_token and telegram_chat_id:
            self.telegram_bot = TelegramBot(telegram_token, telegram_chat_id)
        else:
            self.telegram_bot = None
        
        # Register agent-specific tools
        self._register_agent_tools()
        
        # Load tool registry state if available
        if tool_registry_path:
            self.llm.tool_registry.load_state(tool_registry_path)
            
        logger.info("Agent initialized successfully")
        
    def update_physical_state(self):
        """Update the agent's physical state based on current conditions."""
        # Energy naturally decays over time
        self.physical_state["energy"] = max(0, self.physical_state["energy"] - 0.01)
        
        # Focus is affected by energy level
        self.physical_state["focus"] = self.physical_state["energy"] * 0.8
        
        # Health is affected by energy and focus
        self.physical_state["health"] = (self.physical_state["energy"] + 
            self.physical_state["focus"]) / 2
            
    def invoke_tool(self, tool_name, **params):
        """Allow the agent to invoke tools that can affect its state.
        
        Args:
            tool_name (str): Name of the tool to invoke
            **params: Tool parameters
            
        Returns:
            str: Result of the tool invocation
        """
        # Handle special internal tools differently
        if tool_name == "adjust_emotion":
            return self._adjust_emotion(params.get("emotion"), params.get("change", 0))
        elif tool_name == "rest":
            return self._rest(params.get("amount", 0.1))
        
        # For regular tools, use the tool registry
        try:
            logger.info(f"Invoking tool '{tool_name}' with params: {params}")
            result = self.llm.tool_registry.invoke_tool(tool_name, **params)
            logger.info(f"Tool '{tool_name}' returned: {str(result)[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error invoking tool '{tool_name}': {e}")
            return f"Error invoking tool '{tool_name}': {str(e)}"

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
        
        # Tool to check for new Telegram messages without reading them
        self.llm.tool_registry.register_tool(Tool(
            name="check_telegram",
            description="Check if there are any unread messages without reading them",
            function=lambda: self._check_telegram_messages(),
            usage_example="[TOOL: check_telegram()]"
        ))
        
        # Tool to write journal entries
        self.llm.tool_registry.register_tool(Tool(
            name="write_journal",
            description="Write an entry to the agent's journal for future reference",
            function=lambda entry: self._write_journal_entry(entry),
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
        """Adjust the intensity of an emotion.
        
        Args:
            emotion (str): Name of the emotion to adjust
            change (float): Amount to change the emotion by (-1.0 to 1.0)
            
        Returns:
            dict: Result of the adjustment
        """
        if emotion not in self.mind.emotion_center.emotions:
            return {"error": f"Unknown emotion: {emotion}"}
            
        self.mind.emotion_center.emotions[emotion].intensity += change
        self.mind.emotion_center.emotions[emotion].intensity = max(0, min(1, 
            self.mind.emotion_center.emotions[emotion].intensity))
            
        return {
            "emotion": emotion,
            "new_intensity": self.mind.emotion_center.emotions[emotion].intensity
        }
        
    def _rest(self, amount=0.1):
        """Rest to recover energy.
        
        Args:
            amount (float): Amount of energy to recover (0.0 to 1.0)
            
        Returns:
            dict: Result of resting
        """
        self.physical_state["energy"] = min(1.0, self.physical_state["energy"] + amount)
        return {
            "energy": self.physical_state["energy"]
        }

    def _recall_memories(self, query, count):
        """Recall memories based on query or emotional state."""
        count = int(count)
        
        # If a query is provided, use semantic search
        if query:
            logger.info(f"Recalling memories with direct query: {query}")
            memories = self.mind.memory.recall(
                query=query,
                n=count
            )
        else:
            # If no query, fall back to emotional state only
            logger.info("Recalling memories based on emotional state only")
            memories = self.mind.memory.recall(
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

    def _check_telegram_messages(self):
        """Check if there are any unread messages without reading them."""
        if not self.telegram_bot.token:
            return {
                "success": False,
                "output": "Telegram bot not configured. Cannot check messages."
            }
        
        try:
            # Check for new messages
            new_messages = self.telegram_bot.check_new_messages()
            
            # Check if there are any unread messages
            has_unread = self.telegram_bot.has_unread_messages()
            
            if not has_unread:
                return {
                    "success": True,
                    "output": "No unread messages."
                }
            
            return {
                "success": True,
                "output": "You have unread message(s). Use the receive_telegram tool to read them."
            }
        except Exception as e:
            logger.error(f"Error checking Telegram messages: {e}", exc_info=True)
            return {
                "success": False,
                "output": f"Error checking messages: {str(e)}"
            }

    def _write_journal_entry(self, entry):
        """Write an entry to the journal."""
        try:
            # Handle various input formats that might be passed in by the LLM
            if entry is None:
                entry = "Empty journal entry"
                
            # Handle cases where entry is passed with quotes
            if isinstance(entry, str):
                # Remove surrounding quotes if they exist
                if (entry.startswith('"') and entry.endswith('"')) or (entry.startswith("'") and entry.endswith("'")):
                    entry = entry[1:-1]
            
            # Convert non-string entries to string
            if not isinstance(entry, str):
                entry = str(entry)
                
            # Log the entry type and content for debugging
            logger.info(f"Writing journal entry of type {type(entry)}: '{entry[:50]}...'")
            
            # Ensure we have an entry
            if not entry.strip():
                entry = "Blank journal entry recorded at this timestamp"
                
            # Write the entry
            success = self.journal.write_entry(entry)
            
            if success:
                return f"Journal entry recorded: '{entry[:50]}...'" if len(entry) > 50 else f"Journal entry recorded: '{entry}'"
            else:
                # If writing failed, try once more with a simplified entry
                fallback_entry = f"Journal entry (simplified): {entry[:100]}"
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

        if 'description' in emotion_values:
            return emotion_values['description']
        else:
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
        
        # Shutdown Telegram bot if active
        if self.telegram_bot:
            self.telegram_bot.shutdown()
            
        logger.info("Agent shutdown completed")
        return True 