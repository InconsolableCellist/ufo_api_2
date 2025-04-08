from datetime import datetime
import logging

logger = logging.getLogger('agent_simulation.managers.personality_manager')

class PersonalityManager:
    def __init__(self):
        self.personality_traits = []  # List of traits with timestamps and importance
        self.max_traits = 20  # Maximum number of traits to maintain
        
    def add_trait(self, realization, importance=5):
        """Add a new personality trait or realization
        
        Args:
            realization (str): The trait or realization about self
            importance (int): How important this is, 1-10 scale (default: 5)
        
        Returns:
            dict: Result with success flag and output message
        """
        current_time = datetime.now()
        
        # Validate importance
        if not isinstance(importance, (int, float)) or importance < 1 or importance > 10:
            importance = 5  # Default to medium importance
            
        # Add the new trait
        self.personality_traits.append({
            "trait": realization,
            "importance": importance,
            "timestamp": current_time.isoformat(),
            "reinforcement_count": 1  # Start with 1, will increase if the trait is reinforced
        })
        
        # Sort traits by importance (descending)
        self.personality_traits.sort(key=lambda x: x["importance"], reverse=True)
        
        # Remove oldest least important traits if we exceed max_traits
        if len(self.personality_traits) > self.max_traits:
            # Sort by importance (ascending) and timestamp (ascending) for removal
            temp_list = sorted(self.personality_traits, key=lambda x: (x["importance"], x["timestamp"]))
            # Remove the least important oldest trait
            removed_trait = temp_list[0]
            self.personality_traits.remove(removed_trait)
            
            logger.info(f"Removed least important trait: {removed_trait['trait']} (importance: {removed_trait['importance']})")
            
        logger.info(f"Added personality trait: '{realization}' with importance {importance}")
        return {
            "success": True,
            "output": f"Self-realization added: '{realization}' (importance: {importance}/10)"
        }
        
    def reinforce_trait(self, realization):
        """Reinforce an existing trait by matching its description"""
        # Look for similar traits
        best_match = None
        best_similarity = 0.4  # Similarity threshold
        
        for trait in self.personality_traits:
            # Simple fuzzy matching - compare lowercase versions of strings
            similarity = self._similarity(trait["trait"].lower(), realization.lower())
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = trait
                
        # If we found a match, reinforce it
        if best_match:
            best_match["reinforcement_count"] += 1
            best_match["importance"] = min(10, best_match["importance"] + 0.5)  # Increase importance but cap at 10
            best_match["timestamp"] = datetime.now().isoformat()  # Update timestamp
            
            logger.info(f"Reinforced trait: '{best_match['trait']}' (new importance: {best_match['importance']})")
            return {
                "success": True,
                "output": f"Reinforced existing self-realization: '{best_match['trait']}' (importance now: {best_match['importance']}/10)"
            }
            
        # If no match, add as new trait
        return self.add_trait(realization)
        
    def get_traits(self, count=None):
        """Get traits sorted by importance (most important first)"""
        sorted_traits = sorted(self.personality_traits, key=lambda x: x["importance"], reverse=True)
        
        if count and 0 < count < len(sorted_traits):
            return sorted_traits[:count]
            
        return sorted_traits
        
    def get_traits_formatted(self, count=None):
        """Get traits formatted as a readable string"""
        traits = self.get_traits(count)
        
        if not traits:
            return "No personality traits or self-realizations recorded yet."
            
        lines = ["My personality traits and self-realizations:"]
        
        for i, trait in enumerate(traits):
            # Calculate how long ago this trait was added/reinforced
            trait_time = datetime.fromisoformat(trait["timestamp"])
            duration = datetime.now() - trait_time
            
            if duration.days > 0:
                time_ago = f"{duration.days} days ago"
            elif duration.seconds >= 3600:
                time_ago = f"{duration.seconds // 3600} hours ago"
            elif duration.seconds >= 60:
                time_ago = f"{duration.seconds // 60} minutes ago"
            else:
                time_ago = f"{duration.seconds} seconds ago"
                
            # Format the line with trait details
            line = f"{i+1}. {trait['trait']} (importance: {trait['importance']:.1f}/10, " \
                  f"reinforced {trait['reinforcement_count']} times, last updated {time_ago})"
            lines.append(line)
            
        return "\n".join(lines)
        
    def _similarity(self, str1, str2):
        """Calculate a simple similarity score between two strings"""
        # Simple string matching metric - proportion of words that match
        if not str1 or not str2:
            return 0
            
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0
            
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))
