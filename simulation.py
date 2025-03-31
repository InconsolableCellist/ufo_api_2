from model import *

def main():
    # Initialize components
    llm = LLMInterface()
    
    # Set path for persistent memory
    memory_path = "agent_memory.pkl"
    
    agent = Agent(llm, memory_path)
    simulation = SimulationController(agent)
    
    # Add some initial tasks
    agent.mind.motivation_center.tasks.append(Task("Meditate on existence"))
    agent.mind.motivation_center.tasks.append(Task("Explore emotional state"))
    
    # Add a stimuli source (could be user input, environment, etc.)
    class TestStimuli:
        def get_stimuli(self):
            # Random stimuli for testing
            if random.random() > 0.7:
                return {"external_event": random.choice(["good news", "bad news", "neutral event"])}
            return {}
    
    simulation.add_stimuli_source(TestStimuli())
    
    # Run simulation
    for _ in range(20):  # 20 time steps
        step_result = simulation.run_step()
        
        # Print results
        print(f"\nTime: {step_result['time']}")
        print(f"Physical state: {step_result['agent_state']['physical']}")
        print(f"Emotions: {step_result['agent_state']['mental']['emotional_state']}")
        print(f"Thoughts: {step_result['agent_state']['mental']['conscious_thought']}")
        print(f"Action: {step_result['agent_state']['mental']['action']}")
        
        # Optional: Add delay for real-time observation
        import time
        time.sleep(1)

if __name__ == "__main__":
    main()