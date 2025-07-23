"""Thought summary manager for storing and summarizing agent thoughts."""

import logging
import os
import pickle
import queue
import threading
import time
import json
import requests
from datetime import datetime
from config import STATE_DIR
# Remove the direct imports to break circular dependency
# from services.sqlite_db import update_with_summary, get_messages_without_summary

logger = logging.getLogger('agent_simulation.managers.thought_summary')

class ThoughtSummaryManager:
    """Manages the storage and summarization of thoughts.
    
    Features:
    - Persistent storage of thoughts
    - Asynchronous summarization
    - Thought categorization
    - Summary retrieval
    """
    
    def __init__(self, db_path=None):
        """Initialize the thought summary manager.
        
        Args:
            db_path (str): Path to persist thought summaries
        """
        if db_path is None:
            db_path = os.path.join(STATE_DIR, "thought_summaries.pkl")
            
        self.db_path = db_path
        self.thoughts_db = self._load_db()
        self.summary_queue = queue.Queue()
        self.summarization_active = True
        self.summary_thread = None
        self.summary_available = True  # Track if summary API is available
        
        # Start the summarization thread
        self.start_summarization()
    
    def _load_db(self):
        """Load the database from disk or create a new one."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    return pickle.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Error loading thought database: {e}")
            return []
    
    def _save_db(self):
        """Save the database to disk."""
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.thoughts_db, f)
        except Exception as e:
            logger.error(f"Error saving thought database: {e}")
    
    def add_thought(self, thought, thought_type="normal_thought", request_id=None):
        """Add a thought to the database and queue it for summarization.
        
        Args:
            thought (str): The thought content to add
            thought_type (str): Type of thought (e.g., "normal_thought", "ego_thought")
            request_id (str, optional): The request ID for updating SQLite DB
            
        Returns:
            dict: The created thought entry
        """
        if thought_type != "normal_thought":
            # Skip ego and emotional thoughts
            logger.info(f"Skipping non-normal thought of type {thought_type}")
            return
        
        # Validate the thought
        if not thought or len(thought) < 50:
            logger.warning(f"Thought too short ({len(thought) if thought else 0} chars), not adding to summary database")
            return
        
        # Log the thought being added
        logger.info(f"Adding thought to summary database - Type: {thought_type}, Length: {len(thought)}")
        logger.info(f"Thought preview: {thought[:100]}...")
            
        timestamp = time.time()
        entry = {
            "timestamp": timestamp,
            "thought": thought,
            "summary": None,
            "timestamp_formatted": datetime.fromtimestamp(timestamp).isoformat(),
            "request_id": request_id  # Store the request_id for direct DB updates
        }
        
        # Add to database
        self.thoughts_db.append(entry)
        self._save_db()
        
        # Queue for summarization
        if self.summarization_active:
            self.summary_queue.put(entry)
            logger.info(f"Queued thought for summarization, queue size: {self.summary_queue.qsize()}")
        
        return entry
    
    def start_summarization(self):
        """Start the summarization thread.
        
        Returns:
            bool: True if thread started, False if already running
        """
        if self.summary_thread is None or not self.summary_thread.is_alive():
            self.summarization_active = True
            self.summary_thread = threading.Thread(target=self._summarization_worker, daemon=True)
            self.summary_thread.start()
            logger.info("Thought summarization thread started")
            return True
        return False
    
    def stop_summarization(self):
        """Stop the summarization thread.
        
        Returns:
            bool: True if thread stopped, False if not running
        """
        self.summarization_active = False
        if self.summary_thread and self.summary_thread.is_alive():
            logger.info("Stopping thought summarization thread")
            return True
        return False
    
    def get_summaries(self, limit=10, offset=0):
        """Get the most recent thought summaries.
        
        Args:
            limit (int): Maximum number of summaries to return
            offset (int): Number of summaries to skip
            
        Returns:
            list: List of thought summary entries
        """
        sorted_entries = sorted(self.thoughts_db, key=lambda x: x["timestamp"], reverse=True)
        return sorted_entries[offset:offset+limit]
    
    def get_summary_status(self):
        """Get the status of the summarization process.
        
        Returns:
            dict: Status information including queue size and counts
        """
        return {
            "active": self.summarization_active,
            "api_available": self.summary_available,
            "queue_size": self.summary_queue.qsize(),
            "total_entries": len(self.thoughts_db),
            "summarized_entries": sum(1 for entry in self.thoughts_db if entry.get("summary") is not None)
        }
    
    def _summarize_thought(self, entry):
        """Summarize a thought using the summary API.
        
        Args:
            entry (dict): The thought entry to summarize
            
        Returns:
            bool: True if summarization successful, False otherwise
        """
        try:
            # Get the template from templates.py
            from templates import SUMMARY_TEMPLATE
            
            # Replace {thoughts} with the actual thought content
            prompt = SUMMARY_TEMPLATE.replace("{thoughts}", entry["thought"])
            
            # Make API request to summary endpoint
            logger.info(f"Sending request to summary API for thought {entry['timestamp_formatted']}")
            response = requests.post(
                "http://mlboy:5000/v1/chat/completions",
                json={
                    "model": "summary-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 300
                },
                timeout=60  # Increased timeout to 60 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Update entry with summary
                entry["summary"] = summary
                
                # If we have a request_id, update the SQLite database directly
                if entry.get("request_id"):
                    try:
                        # Delay import to avoid circular dependency
                        from services.sqlite_db import update_with_summary
                        logger.info(f"Updating SQLite DB with summary for request_id: {entry['request_id']}")
                        update_with_summary(entry["request_id"], summary)
                    except Exception as e:
                        logger.error(f"Error updating SQLite database with summary: {e}")
                else:
                    # Fallback to content matching for older entries without request_id
                    try:
                        # Delay import to avoid circular dependency
                        from services.sqlite_db import get_messages_without_summary, update_with_summary
                        unsummarized_messages = get_messages_without_summary()
                        for request_id, response_text in unsummarized_messages:
                            if response_text and entry["thought"] in response_text:
                                logger.info(f"Found matching content for request_id: {request_id}")
                                update_with_summary(request_id, summary)
                                break
                    except Exception as e:
                        logger.error(f"Error updating SQLite database with summary using content matching: {e}")
                
                # Mark API as available
                self.summary_available = True
                logger.info(f"Successfully summarized thought: {summary[:50]}...")
                
                # Save the updated database
                self._save_db()
                return True
            else:
                logger.error(f"Failed to summarize thought: {response.status_code}, {response.text}")
                if response.status_code in (500, 502, 503, 504):
                    # Mark API as unavailable on server errors
                    self.summary_available = False
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Summary API error: {e}")
            self.summary_available = False
            return False
        except Exception as e:
            logger.error(f"Error summarizing thought: {e}")
            return False
    
    def _summarization_worker(self):
        """Worker thread for processing the summary queue."""
        while self.summarization_active:
            try:
                # If API is not available, wait before retrying
                if not self.summary_available:
                    logger.warning("Summary API not available, pausing summarization")
                    time.sleep(30)  # Wait 30 seconds before checking again
                    continue
                
                # Get next entry from queue
                entry = self.summary_queue.get(block=True, timeout=1)
                
                # Skip if already summarized
                if entry.get("summary") is not None:
                    self.summary_queue.task_done()
                    continue
                
                # Summarize the thought synchronously
                logger.info(f"Processing thought from {entry['timestamp_formatted']} for summarization")
                success = self._summarize_thought(entry)
                
                # If unsuccessful and API is available, re-queue with backoff
                if not success and self.summary_available:
                    logger.warning("Failed to summarize thought, will retry later")
                    time.sleep(5)  # Short delay before re-queuing
                    self.summary_queue.put(entry)
                
                self.summary_queue.task_done()
                
            except queue.Empty:
                # No items in queue, just continue
                pass
            except Exception as e:
                logger.error(f"Error in summarization worker: {e}")
                time.sleep(5)  # Delay before next iteration on error
                
        logger.info("Summarization worker stopped")

    def force_summarize_all(self):
        """Force immediate summarization of all thoughts in the queue.
        
        Returns:
            dict: Results of the forced summarization
        """
        logger.info(f"Forcing summarization of {self.summary_queue.qsize()} pending thoughts")
        
        # Process all thoughts in the queue synchronously
        processed_count = 0
        failed_count = 0
        
        # Make a copy of the queue to avoid concurrent modification
        thoughts_to_process = []
        while not self.summary_queue.empty():
            try:
                thoughts_to_process.append(self.summary_queue.get(block=False))
                self.summary_queue.task_done()
            except queue.Empty:
                break
        
        # Process all thoughts
        for entry in thoughts_to_process:
            # Skip if already summarized
            if entry.get("summary") is not None:
                continue
                
            logger.info(f"Force summarizing thought from {entry['timestamp_formatted']}")
            success = self._summarize_thought(entry)
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
                # Put back in queue for later processing
                self.summary_queue.put(entry)
        
        return {
            "processed": processed_count,
            "failed": failed_count,
            "remaining": self.summary_queue.qsize()
        }

    def get_recent_entries(self, limit=3):
        """Get the most recent thought entries.
        
        Args:
            limit (int): Maximum number of entries to return
            
        Returns:
            list: List of recent thought entries
        """
        # Sort entries by timestamp in descending order
        sorted_entries = sorted(self.thoughts_db, key=lambda x: x["timestamp"], reverse=True)
        
        # Get the most recent entries
        recent_entries = sorted_entries[:limit]
        
        # Format entries for display
        formatted_entries = []
        for entry in recent_entries:
            formatted_entry = f"[{entry['timestamp_formatted']}] {entry['thought']}"
            formatted_entries.append(formatted_entry)
            
        return formatted_entries 