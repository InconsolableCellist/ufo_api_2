"""Telegram integration service for the agent."""

import os
import json
import logging
from datetime import datetime
from config import STATE_DIR

logger = logging.getLogger('agent_simulation.services.telegram')

class TelegramBot:
    """Service for interacting with Telegram messaging platform."""
    
    def __init__(self, token=None, chat_id=None):
        """Initialize the Telegram bot service.
        
        Args:
            token: Telegram bot token (or uses TELEGRAM_BOT_TOKEN environment variable)
            chat_id: Chat ID to send messages to (or uses TELEGRAM_CHAT_ID environment variable)
        """
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.bot = None
        self.last_message_time = None  # Track when the last message was sent
        self.rate_limit_seconds = 3600  # 1 hour in seconds
        self.messages_file = os.path.join(STATE_DIR, "telegram_messages.json")
        self.last_update_id = 0  # To keep track of processed messages
        
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
                
                # Load existing messages and last update ID
                self._load_message_state()
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
        else:
            logger.warning("Telegram bot token not provided. Messaging will be disabled.")
    
    def send_message(self, message):
        """Send a message to the configured chat ID with rate limiting.
        
        Args:
            message: The message text to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
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
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            # Run the coroutine to send the message
            loop.run_until_complete(self.bot.send_message(chat_id=self.chat_id, text=message))
            
            self.last_message_time = current_time  # Update the last message time
            logger.info(f"Telegram message sent: {message[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
            
    def _load_message_state(self):
        """Load the message state from file (internal method)."""
        try:
            if os.path.exists(self.messages_file):
                with open(self.messages_file, 'r') as f:
                    state = json.load(f)
                    self.last_update_id = state.get("last_update_id", 0)
                    logger.info(f"Loaded message state with last update ID: {self.last_update_id}")
            else:
                logger.info("No message state file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading message state: {e}")
            # Create a fresh state file
            self._save_message_state()
            
    def _save_message_state(self, messages=None):
        """Save the message state to file (internal method).
        
        Args:
            messages: List of messages to save (optional)
            
        Returns:
            bool: True if state was saved successfully, False otherwise
        """
        try:
            state = {
                "last_update_id": self.last_update_id,
                "messages": messages or []
            }
            
            with open(self.messages_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved message state with last update ID: {self.last_update_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving message state: {e}")
            return False
            
    def check_new_messages(self):
        """Check for new messages from Telegram and save them.
        
        Returns:
            list: List of new messages received
        """
        if not self.bot:
            logger.warning("Telegram bot not configured. Cannot check messages.")
            return []
            
        try:
            # Get updates with offset (to avoid getting the same message twice)
            try:
                import asyncio
                # Create a new event loop for this thread if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Check if we're already in an event loop
                if loop.is_running():
                    # If we're already in an event loop, use a different approach
                    # Create a future to run the coroutine
                    future = asyncio.run_coroutine_threadsafe(
                        self.bot.get_updates(offset=self.last_update_id + 1, timeout=5),
                        loop
                    )
                    # Wait for the result with a timeout
                    try:
                        updates = future.result(timeout=30)  
                    except TimeoutError:
                        logger.warning("Timeout waiting for Telegram updates. This may be due to network issues or the Telegram API being slow.")
                        # Cancel the future to avoid resource leaks
                        future.cancel()
                        return []
                else:
                    # Run the coroutine to get updates
                    try:
                        updates = loop.run_until_complete(
                            asyncio.wait_for(
                                self.bot.get_updates(offset=self.last_update_id + 1, timeout=5),
                                timeout=15  # Increased timeout to 15 seconds
                            )
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for Telegram updates. This may be due to network issues or the Telegram API being slow.")
                        return []
            except (ImportError, AttributeError):
                # Fall back to synchronous behavior for older versions
                updates = self.bot.get_updates(offset=self.last_update_id + 1, timeout=5)
            
            if not updates:
                logger.info("No new messages found")
                return []
                
            # Process new messages
            new_messages = []
            
            for update in updates:
                # Update the last processed update ID
                if update.update_id > self.last_update_id:
                    self.last_update_id = update.update_id
                
                # Only process messages from allowed chat IDs
                if hasattr(update, 'message') and update.message:
                    message = update.message
                    
                    # Only process messages from authorized users
                    if str(message.chat_id) == str(self.chat_id):
                        # Create a message object
                        message_obj = {
                            "id": update.update_id,
                            "text": message.text or "(non-text content)",
                            "from": message.from_user.username or message.from_user.first_name if message.from_user else "Unknown",
                            "timestamp": message.date.isoformat() if hasattr(message, 'date') else datetime.now().isoformat(),
                            "read": False
                        }
                        
                        new_messages.append(message_obj)
                        logger.info(f"New message received: {message_obj['text'][:30]}...")
            
            # If we found new messages, save them
            if new_messages:
                # Load existing unread messages
                existing_messages = self._get_saved_messages()
                
                # Add new messages
                all_messages = existing_messages + new_messages
                
                # Save the updated state
                self._save_message_state(all_messages)
                
            return new_messages
        except Exception as e:
            logger.error(f"Error checking for new messages: {e}", exc_info=True)
            return []
            
    def has_unread_messages(self):
        """Check if there are any unread messages without marking them as read.
        
        Returns:
            bool: True if there are unread messages, False otherwise
        """
        if not self.bot:
            logger.warning("Telegram bot not configured. Cannot check messages.")
            return False
            
        try:
            # First check for any new messages
            self.check_new_messages()
            
            # Then check if there are any unread messages
            messages = self._get_saved_messages()
            unread = [m for m in messages if not m.get("read", False)]
            
            return len(unread) > 0
        except Exception as e:
            logger.error(f"Error checking for unread messages: {e}")
            return False
            
    def _get_saved_messages(self):
        """Get saved messages from the state file (internal method).
        
        Returns:
            list: List of saved messages
        """
        try:
            if os.path.exists(self.messages_file):
                with open(self.messages_file, 'r') as f:
                    state = json.load(f)
                    return state.get("messages", [])
            return []
        except Exception as e:
            logger.error(f"Error getting saved messages: {e}")
            return []
            
    def get_unread_messages(self):
        """Get all unread messages.
        
        Returns:
            list: List of unread messages
        """
        if not self.bot:
            logger.warning("Telegram bot not configured. Cannot check messages.")
            return []
            
        try:
            # First check for any new messages
            self.check_new_messages()
            
            # Then get all unread messages
            messages = self._get_saved_messages()
            unread = [m for m in messages if not m.get("read", False)]
            
            logger.info(f"Found {len(unread)} unread messages")
            return unread
        except Exception as e:
            logger.error(f"Error getting unread messages: {e}")
            return []
            
    def mark_messages_as_read(self):
        """Mark all unread messages as read.
        
        Returns:
            bool: True if messages were marked as read successfully, False otherwise
        """
        try:
            messages = self._get_saved_messages()
            
            # Mark all as read
            for message in messages:
                message["read"] = True
                
            # Save the updated state
            self._save_message_state(messages)
            
            logger.info(f"Marked {len(messages)} messages as read")
            return True
        except Exception as e:
            logger.error(f"Error marking messages as read: {e}")
            return False 