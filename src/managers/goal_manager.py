from datetime import datetime
import logging

logger = logging.getLogger('agent_simulation.managers.goal_manager')

class GoalManager:
    def __init__(self):
        self.short_term_goals = []  # Max 10 goals
        self.long_term_goal = None
        self.last_long_term_goal_change = None
        self.goal_change_cooldown = 3600  # 1 hour in seconds
        self.generation_cycle = 0  # Track generations/cycles
        self.max_short_term_goals = 10  # Maximum number of short-term goals allowed
        
    def increment_cycle(self):
        """Increment the generation cycle counter"""
        self.generation_cycle += 1
        logger.info(f"Incremented goal cycle counter to {self.generation_cycle}")
        
    def add_short_term_goal(self, goal):
        """Add a short-term goal, with a maximum of 10 goals"""
        current_time = datetime.now()
        
        # Check if we've reached the maximum number of goals
        if len(self.short_term_goals) >= self.max_short_term_goals:
            error_msg = f"Maximum number of short-term goals ({self.max_short_term_goals}) reached. Please remove a goal before adding another."
            logger.warning(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
            
        # Add the new goal
        self.short_term_goals.append({
            "goal": goal,
            "timestamp": current_time.isoformat(),
            "created_at_cycle": self.generation_cycle,
            "cycles": 0  # This will be incremented on each cycle
        })
        logger.info(f"Added short-term goal: {goal}. Current goals: {[g['goal'] for g in self.short_term_goals]}")
        return {
            "success": True,
            "output": f"Added short-term goal: {goal}"
        }
        
    def set_long_term_goal(self, goal):
        """Set a long-term goal with cooldown"""
        current_time = datetime.now()
        
        # Check cooldown
        if self.last_long_term_goal_change:
            time_since_last = (current_time - self.last_long_term_goal_change).total_seconds()
            if time_since_last < self.goal_change_cooldown:
                remaining = self.goal_change_cooldown - time_since_last
                return {
                    "success": False,
                    "error": f"Cannot change long-term goal yet. Please wait {remaining:.1f} seconds."
                }
        
        self.long_term_goal = {
            "goal": goal,
            "timestamp": current_time.isoformat(),
            "created_at_cycle": self.generation_cycle,
            "cycles": 0  # This will be incremented on each cycle
        }
        self.last_long_term_goal_change = current_time
        logger.info(f"Set long-term goal: {goal}")
        return {
            "success": True,
            "output": f"Set new long-term goal: {goal}"
        }
        
    def update_goal_cycles(self):
        """Update cycle counts for all active goals"""
        # Update short-term goals
        for goal in self.short_term_goals:
            # Make sure the goal has a cycles field
            if "cycles" not in goal:
                goal["cycles"] = 0
                goal["created_at_cycle"] = self.generation_cycle - 1  # Assume it was created in the previous cycle
            goal["cycles"] += 1
            
        # Update long-term goal
        if self.long_term_goal:
            # Make sure the goal has a cycles field
            if "cycles" not in self.long_term_goal:
                self.long_term_goal["cycles"] = 0
                self.long_term_goal["created_at_cycle"] = self.generation_cycle - 1  # Assume it was created in the previous cycle
            self.long_term_goal["cycles"] += 1
            
        logger.info(f"Updated goal cycles. Current cycle: {self.generation_cycle}")
        
    def get_goals(self):
        """Get current goals with duration information"""
        current_time = datetime.now()
        
        # Format short-term goals with duration info
        short_term = []
        for goal in self.short_term_goals:
            # Calculate duration
            goal_time = datetime.fromisoformat(goal["timestamp"])
            duration_seconds = (current_time - goal_time).total_seconds()
            duration_minutes = duration_seconds / 60
            duration_hours = duration_minutes / 60
            
            # Format duration string
            if duration_hours >= 1:
                duration_str = f"{duration_hours:.1f} hours"
            elif duration_minutes >= 1:
                duration_str = f"{duration_minutes:.1f} minutes"
            else:
                duration_str = f"{duration_seconds:.1f} seconds"
            
            # Ensure cycles field exists
            cycles = goal.get("cycles", 0)
                
            short_term.append({
                "text": goal["goal"],
                "cycles": cycles,
                "duration": duration_str
            })
        
        # Format long-term goal with duration info
        long_term = None
        if self.long_term_goal:
            # Calculate duration
            goal_time = datetime.fromisoformat(self.long_term_goal["timestamp"])
            duration_seconds = (current_time - goal_time).total_seconds()
            duration_minutes = duration_seconds / 60
            duration_hours = duration_minutes / 60
            duration_days = duration_hours / 24
            
            # Format duration string
            if duration_days >= 1:
                duration_str = f"{duration_days:.1f} days"
            elif duration_hours >= 1:
                duration_str = f"{duration_hours:.1f} hours"
            elif duration_minutes >= 1:
                duration_str = f"{duration_minutes:.1f} minutes"
            else:
                duration_str = f"{duration_seconds:.1f} seconds"
            
            # Ensure cycles field exists
            cycles = self.long_term_goal.get("cycles", 0)
                
            long_term = {
                "text": self.long_term_goal["goal"],
                "cycles": cycles,
                "duration": duration_str
            }
            
        goals = {
            "short_term": [g["text"] for g in short_term],
            "long_term": long_term["text"] if long_term else None,
            "short_term_details": short_term,
            "long_term_details": long_term
        }
        
        logger.info(f"Retrieved goals with duration info: {goals}")
        return goals
        
    def remove_short_term_goal(self, index):
        """Remove a short-term goal by index"""
        if 0 <= index < len(self.short_term_goals):
            removed = self.short_term_goals.pop(index)
            logger.info(f"Removed goal at index {index}: {removed['goal']}. Remaining goals: {[g['goal'] for g in self.short_term_goals]}")
            return f"Removed short-term goal: {removed['goal']}"
        return "Invalid goal index"
