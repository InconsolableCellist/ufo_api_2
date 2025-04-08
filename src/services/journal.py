"""Journal service for writing and reading agent journal entries."""

import os
import logging
from datetime import datetime
from config import STATE_DIR

logger = logging.getLogger('agent_simulation.services.journal')

class Journal:
    """Manages the agent's journal for recording thoughts and reflections."""
    
    def __init__(self, file_path=None):
        """Initialize the journal service.
        
        Args:
            file_path: Path to the journal file
        """
        if file_path is None:
            file_path = os.path.join(STATE_DIR, "agent_journal.txt")
            
        self.file_path = file_path
        # Create the file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("# Agent Journal\n\n")
            logger.info(f"Created new journal file at {file_path}")
    
    def write_entry(self, entry):
        """Write a new entry to the journal with timestamp.
        
        Args:
            entry: The journal entry text
            
        Returns:
            bool: True if entry was written successfully, False otherwise
        """
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
        """Read the most recent entries from the journal.
        
        Args:
            num_entries: Number of entries to read
            
        Returns:
            list: List of recent journal entries with timestamps
        """
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
                
                # Get the entry content (from next line until next marker or end)
                content_lines = []
                i = marker + 1
                while i < len(lines) and (i + 1 >= len(lines) or not lines[i + 1].startswith('## ')):
                    content_lines.append(lines[i].strip())
                    i += 1
                
                content = "\n".join(content_lines)
                entries.append(f"{timestamp}: {content}")
                
            return entries
            
        except Exception as e:
            logger.error(f"Failed to read journal entries: {e}")
            return [] 