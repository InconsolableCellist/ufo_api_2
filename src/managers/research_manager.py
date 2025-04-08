"""Research manager for handling research results and providing retrieval capabilities."""

import os
import logging
import pickle
from datetime import datetime

logger = logging.getLogger('agent_simulation.managers.research')

class ResearchManager:
    """Manages research results and provides retrieval capabilities."""
    
    def __init__(self, storage_path="research_database.pkl"):
        """Initialize the research manager.
        
        Args:
            storage_path (str): Path to store the research database
        """
        self.storage_path = storage_path
        self.research_database = {}  # Format: {query: {pages: [], timestamps: [], results: []}}
        self.research_history = []  # Most recent research queries
        self.max_history = 20
        self.load_state()
        
    def add_research_result(self, query, page, result):
        """Add a research result to the database.
        
        Args:
            query (str): The research query
            page (int): The page number
            result (str): The research result text
            
        Returns:
            bool: True if successful
        """
        # Normalize query for consistent lookups
        normalized_query = query.strip().lower()
        timestamp = datetime.now().isoformat()
        
        # Initialize query record if it doesn't exist
        if normalized_query not in self.research_database:
            self.research_database[normalized_query] = {
                "pages": [],
                "timestamps": [],
                "results": [],
                "first_researched": timestamp
            }
        
        # Update research database
        if page in self.research_database[normalized_query]["pages"]:
            # Replace existing page
            idx = self.research_database[normalized_query]["pages"].index(page)
            self.research_database[normalized_query]["results"][idx] = result
            self.research_database[normalized_query]["timestamps"][idx] = timestamp
        else:
            # Add new page
            self.research_database[normalized_query]["pages"].append(page)
            self.research_database[normalized_query]["results"].append(result)
            self.research_database[normalized_query]["timestamps"].append(timestamp)
            
        # Update research history
        research_entry = {"query": query, "page": page, "timestamp": timestamp}
        
        # Remove existing entries for this query+page to avoid duplicates
        self.research_history = [entry for entry in self.research_history 
                               if not (entry["query"].lower() == normalized_query and entry["page"] == page)]
        
        # Add to history
        self.research_history.append(research_entry)
        
        # Keep only max_history entries
        if len(self.research_history) > self.max_history:
            self.research_history = self.research_history[-self.max_history:]
            
        # Save state
        self.save_state()
        
        return True
        
    def get_research_result(self, query, page=1):
        """Get a specific research result.
        
        Args:
            query (str): The research query
            page (int): The page number
            
        Returns:
            str: The research result or None if not found
        """
        normalized_query = query.strip().lower()
        
        if normalized_query not in self.research_database:
            return None
            
        if page not in self.research_database[normalized_query]["pages"]:
            return None
            
        idx = self.research_database[normalized_query]["pages"].index(page)
        return self.research_database[normalized_query]["results"][idx]
        
    def get_all_research_pages(self, query):
        """Get all research pages for a query.
        
        Args:
            query (str): The research query
            
        Returns:
            list: List of page numbers
        """
        normalized_query = query.strip().lower()
        
        if normalized_query not in self.research_database:
            return []
            
        # Return list of page numbers
        return self.research_database[normalized_query]["pages"]
        
    def get_research_summary(self, query):
        """Get a summary of research results for a query.
        
        Args:
            query (str): The research query
            
        Returns:
            str: A formatted summary of research results
        """
        normalized_query = query.strip().lower()
        
        if normalized_query not in self.research_database:
            return f"No research found for query: '{query}'"
            
        pages = self.research_database[normalized_query]["pages"]
        timestamps = self.research_database[normalized_query]["timestamps"]
        first_researched = self.research_database[normalized_query].get("first_researched", timestamps[0])
        
        # Format timestamps to be more readable
        formatted_timestamps = []
        for ts in timestamps:
            dt = datetime.fromisoformat(ts)
            formatted_timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            
        # Create summary
        summary = f"Research on '{query}':\n"
        summary += f"First researched: {datetime.fromisoformat(first_researched).strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Total pages: {len(pages)}\n"
        summary += "Pages available:\n"
        
        for i, (page, ts) in enumerate(zip(pages, formatted_timestamps)):
            summary += f"  Page {page}: Last updated {ts}\n"
            
        summary += "\nTo view a specific page, use: [TOOL: recall_research(query:'{query}', page:N)]\n"
        summary += "To see all pages combined, use: [TOOL: recall_research(query:'{query}', all_pages:true)]"
        
        return summary
        
    def get_recent_research(self, count=5):
        """Get most recent research queries.
        
        Args:
            count (int): Number of recent research entries to retrieve
            
        Returns:
            str: Formatted list of recent research queries
        """
        if not self.research_history:
            return "No research history available."
            
        recent = self.research_history[-count:]
        recent.reverse()  # Most recent first
        
        result = "Recent research queries:\n"
        for entry in recent:
            dt = datetime.fromisoformat(entry["timestamp"])
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            result += f"- '{entry['query']}' (Page {entry['page']}, {formatted_time})\n"
            
        return result
        
    def save_state(self):
        """Save the research database to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump({
                    "database": self.research_database,
                    "history": self.research_history
                }, f)
            return True
        except Exception as e:
            logger.error(f"Error saving research database: {e}")
            return False
            
    def load_state(self):
        """Load the research database from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.storage_path):
            logger.info(f"No research database found at {self.storage_path}")
            return False
            
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
                
            self.research_database = data.get("database", {})
            self.research_history = data.get("history", [])
            
            logger.info(f"Loaded research database with {len(self.research_database)} queries")
            return True
        except Exception as e:
            logger.error(f"Error loading research database: {e}")
            return False 