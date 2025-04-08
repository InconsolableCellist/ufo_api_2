"""Motivation center for managing agent's drives and goals."""

import logging
import json
import os
import pickle
from datetime import datetime
from .task import Task

logger = logging.getLogger('agent_simulation.motivation.motivation_center')

class MotivationCenter:
    """Manages the agent's motivational drives, goals, and tasks.
    
    The MotivationCenter handles:
    - Core motivational drives
    - Task management and prioritization
    - Goal setting and tracking
    - Drive-based behavior modulation
    
    TODO: Enhance motivation system:
    - Add hierarchical goal structure
    - Implement drive satisfaction tracking
    - Add motivation-based decision making
    - Implement adaptive goal adjustment
    - Add motivation-emotion interactions
    """
    
    def __init__(self, persist_path=None):
        """Initialize the motivation center.
        
        Args:
            persist_path (str, optional): Path to persist motivation state
        """
        self.tasks = []
        self.core_drives = {
            "curiosity": 0.5,      # Drive to learn and explore
            "achievement": 0.5,    # Drive to complete tasks and goals
            "connection": 0.5,     # Drive for social interaction
            "autonomy": 0.5,       # Drive for independence
            "competence": 0.5      # Drive to improve skills
        }
        
        self.core_values = [
            "Understand myself",
            "Grow emotionally",
            "Explore ideas",
            "Help others",
            "Learn continuously"
        ]
        
        self.persist_path = persist_path
        if persist_path and os.path.exists(persist_path):
            self.load()
            
    def update_drives(self, emotional_state, recent_achievements):
        """Update motivational drives based on emotional state and achievements.
        
        Args:
            emotional_state (dict): Current emotional state
            recent_achievements (list): Recently completed tasks/goals
        """
        # Update curiosity based on emotional state
        if emotional_state.get("curiosity", 0) > 0.7:
            self.core_drives["curiosity"] = min(1.0, self.core_drives["curiosity"] + 0.1)
            
        # Update achievement drive based on task completion
        if recent_achievements:
            self.core_drives["achievement"] = min(1.0, self.core_drives["achievement"] + 0.05 * len(recent_achievements))
            
        # Decay drives over time
        for drive in self.core_drives:
            self.core_drives[drive] = max(0.1, self.core_drives[drive] - 0.01)
            
    def add_task(self, description, priority=None, deadline=None):
        """Add a new task with optional priority calculation.
        
        Args:
            description (str): Task description
            priority (float, optional): Manual priority override
            deadline (datetime, optional): Task deadline
            
        Returns:
            Task: The created task
        """
        if priority is None:
            # Calculate priority based on drives and values
            priority = self._calculate_task_priority(description)
            
        task = Task(description, priority=priority, deadline=deadline)
        self.tasks.append(task)
        return task
        
    def _calculate_task_priority(self, description):
        """Calculate task priority based on drives and values.
        
        Args:
            description (str): Task description
            
        Returns:
            float: Calculated priority (0 to 1)
        """
        priority = 0.5  # Base priority
        
        # Adjust based on core drives
        if "learn" in description.lower() or "study" in description.lower():
            priority += self.core_drives["curiosity"] * 0.2
        if "achieve" in description.lower() or "complete" in description.lower():
            priority += self.core_drives["achievement"] * 0.2
        if "interact" in description.lower() or "communicate" in description.lower():
            priority += self.core_drives["connection"] * 0.2
            
        # Adjust based on core values
        for value in self.core_values:
            if value.lower() in description.lower():
                priority += 0.1
                
        return min(1.0, priority)
        
    def get_current_task(self):
        """Get the highest priority active task.
        
        Returns:
            Task: Highest priority task or None
        """
        active_tasks = [t for t in self.tasks if t.status in ["pending", "in_progress"]]
        if not active_tasks:
            return None
            
        # Sort by urgency (combines priority and deadline)
        return max(active_tasks, key=lambda t: t.get_urgency())
        
    def update_tasks(self, emotional_state):
        """Update task priorities based on emotional state.
        
        Args:
            emotional_state (dict): Current emotional state
        """
        for task in self.tasks:
            if emotional_state.get("stress", 0) > 0.7:
                # Lower priority of non-urgent tasks when stressed
                if not task.deadline:
                    task.priority *= 0.8
            elif emotional_state.get("curiosity", 0) > 0.7:
                # Increase priority of learning-related tasks
                if "learn" in task.description.lower():
                    task.priority = min(1.0, task.priority * 1.2)
                    
    def get_motivation_state(self):
        """Get current motivation system state.
        
        Returns:
            dict: Current motivation state
        """
        return {
            "core_drives": self.core_drives,
            "active_tasks": len([t for t in self.tasks if t.status != "completed"]),
            "completed_tasks": len([t for t in self.tasks if t.status == "completed"]),
            "current_task": self.get_current_task().description if self.get_current_task() else None
        }
        
    def save(self):
        """Save motivation state to disk."""
        if not self.persist_path:
            return
            
        data = {
            'core_drives': self.core_drives,
            'core_values': self.core_values,
            'tasks': [task.to_dict() for task in self.tasks]
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self):
        """Load motivation state from disk."""
        if not os.path.exists(self.persist_path):
            return
            
        with open(self.persist_path, 'rb') as f:
            data = pickle.load(f)
            
        self.core_drives = data.get('core_drives', self.core_drives)
        self.core_values = data.get('core_values', self.core_values)
        
        # Reconstruct tasks from dict representation
        self.tasks = []
        for task_data in data.get('tasks', []):
            task = Task(
                description=task_data['description'],
                priority=task_data['priority'],
                status=task_data['status']
            )
            if task_data.get('deadline'):
                task.deadline = datetime.fromisoformat(task_data['deadline'])
            task.progress = task_data.get('progress', 0.0)
            task.metadata = task_data.get('metadata', {})
            self.tasks.append(task) 