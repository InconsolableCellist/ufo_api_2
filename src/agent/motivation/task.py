"""Task management for the agent's motivation system."""

import logging
from datetime import datetime

logger = logging.getLogger('agent_simulation.motivation.task')

class Task:
    """Represents a task or goal for the agent to pursue.
    
    The Task class manages:
    - Task description and metadata
    - Priority and status tracking
    - Subtask relationships
    - Progress monitoring
    
    TODO: Enhance task management:
    - Add task dependencies
    - Implement task scheduling
    - Add task difficulty estimation
    - Track task completion metrics
    - Add task categorization
    """
    
    def __init__(self, description, priority=0.5, status="pending", deadline=None):
        """Initialize a new task.
        
        Args:
            description (str): Description of the task
            priority (float): Priority from 0 to 1
            status (str): Current status ("pending", "in_progress", "completed", "blocked")
            deadline (datetime, optional): When the task needs to be completed by
        """
        self.description = description
        self.priority = priority
        self.status = status
        self.deadline = deadline
        self.subtasks = []
        self.created_at = datetime.now()
        self.completed_at = None
        self.progress = 0.0  # Progress from 0 to 1
        self.metadata = {}  # For storing additional task-specific data
        
    def add_subtask(self, subtask):
        """Add a subtask to this task.
        
        Args:
            subtask (Task): The subtask to add
            
        Returns:
            bool: True if subtask was added successfully
        """
        if isinstance(subtask, Task):
            self.subtasks.append(subtask)
            self._update_progress()
            return True
        return False
        
    def update_status(self, new_status, progress=None):
        """Update task status and optionally progress.
        
        Args:
            new_status (str): New status value
            progress (float, optional): New progress value (0 to 1)
        """
        valid_statuses = ["pending", "in_progress", "completed", "blocked"]
        if new_status not in valid_statuses:
            logger.warning(f"Invalid status: {new_status}")
            return
            
        self.status = new_status
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
            
        if new_status == "completed":
            self.completed_at = datetime.now()
            self.progress = 1.0
            
        self._update_progress()
        
    def _update_progress(self):
        """Update progress based on subtask completion."""
        if not self.subtasks:
            return
            
        # Progress is average of own progress and subtask progress
        subtask_progress = sum(task.progress for task in self.subtasks) / len(self.subtasks)
        self.progress = (self.progress + subtask_progress) / 2
        
    def get_urgency(self):
        """Calculate task urgency based on deadline and priority.
        
        Returns:
            float: Urgency score from 0 to 1
        """
        if not self.deadline:
            return self.priority
            
        time_remaining = (self.deadline - datetime.now()).total_seconds()
        if time_remaining <= 0:
            return 1.0
            
        # Increase urgency as deadline approaches
        urgency = min(1.0, max(0.0, 1.0 - (time_remaining / (24 * 3600))))  # 24 hour scale
        return (urgency + self.priority) / 2
        
    def to_dict(self):
        """Convert task to dictionary for serialization.
        
        Returns:
            dict: Task data
        """
        return {
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "subtasks": [task.to_dict() for task in self.subtasks],
            "metadata": self.metadata
        } 