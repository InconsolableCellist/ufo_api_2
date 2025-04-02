from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uvicorn
from model import *
import time
import threading
import json
import os
from datetime import datetime

app = FastAPI(
    title="Agent Simulation API",
    description="API for controlling and monitoring an agent simulation with emotional and cognitive processes",
    version="1.0.0"
)

# Global variables to manage simulation state
simulation = None
agent = None
simulation_thread = None
simulation_running = False
simulation_results = []
stop_event = threading.Event()

class SimulationConfig(BaseModel):
    memory_path: Optional[str] = "agent_memory.pkl"
    initial_tasks: Optional[List[str]] = ["Meditate on existence", "Explore emotional state"]

class MemoryQuery(BaseModel):
    query: Optional[str] = None
    emotional_state: Optional[Dict[str, float]] = None
    count: Optional[int] = 3
    memory_type: str = "long"  # "short" or "long"

class SimulationStatus(BaseModel):
    running: bool
    step_count: int
    current_time: Optional[float] = None
    last_result: Optional[Dict[str, Any]] = None

class MemoryResponse(BaseModel):
    memories: List[str]
    query: Optional[str] = None
    memory_type: str

class LLMConfigUpdate(BaseModel):
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    model: Optional[str] = None
    system_message: Optional[str] = None

class UserMessage(BaseModel):
    message: str
    
class UserAnnouncement(BaseModel):
    announcement: str

def update_user_response(message):
    """Update the most recent user message with a response"""
    try:
        if not os.path.exists("user_messages.txt"):
            # TODO: Create the file if it doesn't exist with proper structure
            # TODO: Add initialization of conversation history
            return False
            
        with open("user_messages.txt", "r") as f:
            lines = f.readlines()
            
        # Find the most recent message that has 'Nothing Yet' as the response
        for i in range(len(lines) - 1, -1, -1):
            if "RESPONSE: Nothing Yet" in lines[i]:
                # Update the response
                lines[i] = f"RESPONSE: {message}\n"
                break
                
        # Write the updated content back
        with open("user_messages.txt", "w") as f:
            f.writelines(lines)
            
        return True
    except Exception as e:
        print(f"Error updating user response: {e}")
        return False

def run_simulation_loop():
    global simulation, simulation_running, simulation_results, stop_event
    
    while not stop_event.is_set() and simulation_running:
        try:
            step_result = simulation.run_step()
            simulation_results.append(step_result)
            time.sleep(1)  # Delay between steps
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            simulation_running = False
            break

@app.post("/start", response_model=SimulationStatus)
async def start_simulation(config: SimulationConfig, background_tasks: BackgroundTasks):
    global simulation, agent, simulation_thread, simulation_running, simulation_results, stop_event
    
    if simulation_running:
        raise HTTPException(status_code=400, detail="Simulation is already running")
    
    # Initialize components
    llm = LLMInterface()
    agent = Agent(llm, config.memory_path)
    simulation = SimulationController(agent)
    
    # Add initial tasks
    for task in config.initial_tasks:
        agent.mind.motivation_center.tasks.append(Task(task))
    
    # Add a stimuli source
    class TestStimuli:
        def get_stimuli(self):
            if random.random() > 0.7:
                return {"external_event": random.choice(["good news", "bad news", "neutral event"])}
            return {}
    
    simulation.add_stimuli_source(TestStimuli())
    
    # Reset state
    simulation_results = []
    stop_event.clear()
    simulation_running = True
    
    # Start simulation in background
    background_tasks.add_task(run_simulation_loop)
    
    return SimulationStatus(
        running=simulation_running,
        step_count=len(simulation_results),
        current_time=simulation.current_time if simulation else None,
        last_result=None
    )

@app.post("/stop", response_model=SimulationStatus)
async def stop_simulation():
    global simulation_running, stop_event
    
    if not simulation_running:
        raise HTTPException(status_code=400, detail="Simulation is not running")
    
    stop_event.set()
    simulation_running = False
    
    return SimulationStatus(
        running=simulation_running,
        step_count=len(simulation_results),
        current_time=simulation.current_time if simulation else None,
        last_result=simulation_results[-1] if simulation_results else None
    )

@app.get("/status", response_model=SimulationStatus)
async def get_status():
    global simulation, simulation_running, simulation_results
    
    return SimulationStatus(
        running=simulation_running,
        step_count=len(simulation_results),
        current_time=simulation.current_time if simulation else None,
        last_result=simulation_results[-1] if simulation_results else None
    )

@app.post("/next", response_model=Dict[str, Any])
async def next_step():
    global simulation, simulation_running, simulation_results
    
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if simulation_running:
        raise HTTPException(status_code=400, detail="Cannot manually step while simulation is running automatically")
    
    step_result = simulation.run_step()
    simulation_results.append(step_result)
    
    return step_result

@app.post("/send_message")
async def send_message(message: UserMessage):
    """Send a message to the agent"""
    if not message.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    # TODO: Improve user message handling:
    # - Add proper conversation history tracking
    # - Implement message ID system for tracking conversations
    # - Add metadata for messages (timestamp, read status, etc.)
    # - Implement proper message storage (database instead of text file)
    # - Add emotion analysis of incoming messages to influence agent state
    
    # Update the most recent agent message with this user response
    success = update_user_response(message.message)
    
    if not success:
        return {"status": "warning", "message": "Updated user message but couldn't find the most recent agent request"}
    
    return {"status": "success", "message": "Message sent to agent"}

@app.post("/send_announcement")
async def send_announcement(announcement: UserAnnouncement):
    """Send a special announcement that will appear as a mysterious voice in the agent's mind"""
    if not announcement.announcement.strip():
        raise HTTPException(status_code=400, detail="Announcement cannot be empty")
        
    # Import the module to modify its global variable
    import model
    
    # Set the global announcement
    model.USER_ANNOUNCEMENT = announcement.announcement
    
    return {"status": "success", "message": "Announcement set. It will appear in the next message to the LLM."}

@app.get("/summary")
async def get_summary():
    global agent, simulation_results
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    # Create a summary of the agent's current state
    emotional_state = agent.mind.emotion_center.get_state()
    physical_state = agent.physical_state
    recent_thoughts = list(agent.mind.memory.short_term)
    current_task = agent.mind.motivation_center.get_current_task()
    
    return {
        "emotional_state": emotional_state,
        "physical_state": physical_state,
        "recent_thoughts": recent_thoughts,
        "current_task": current_task.description if current_task else None,
        "step_count": len(simulation_results),
        "simulation_time": simulation.current_time if simulation else None
    }

@app.post("/memory", response_model=MemoryResponse)
async def query_memory(query_params: MemoryQuery):
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if query_params.memory_type == "short":
        memories = list(agent.mind.memory.short_term)
    else:  # long-term memory
        emotional_state = query_params.emotional_state or agent.mind.emotion_center.get_state()
        memories = agent.mind.memory.recall(
            emotional_state=emotional_state,
            query=query_params.query,
            n=query_params.count
        )
    
    return MemoryResponse(
        memories=memories,
        query=query_params.query,
        memory_type=query_params.memory_type
    )

@app.post("/emotion/adjust")
async def adjust_emotion(
    emotion_name: str, 
    change: float = Query(..., ge=-1.0, le=1.0)
):
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if emotion_name not in agent.mind.emotion_center.emotions:
        raise HTTPException(status_code=400, detail=f"Unknown emotion: {emotion_name}")
    
    emotion = agent.mind.emotion_center.emotions[emotion_name]
    emotion.intensity += change
    emotion.intensity = max(0, min(1, emotion.intensity))
    
    return {
        "emotion": emotion_name,
        "previous_intensity": emotion.intensity - change,
        "new_intensity": emotion.intensity,
        "change": change
    }

@app.post("/config/llm")
async def update_llm_configuration(config_update: LLMConfigUpdate):
    from model import update_llm_config, LLM_CONFIG
    
    # Filter out None values
    updates = {k: v for k, v in config_update.dict().items() if v is not None}
    
    if not updates:
        return {"message": "No updates provided", "current_config": LLM_CONFIG}
    
    update_llm_config(updates)
    return {"message": "Configuration updated", "current_config": LLM_CONFIG}

# TODO: Add function to record agent messages to users
# TODO: Add function to retrieve conversation history with pagination
# TODO: Add function to clear conversation history

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8081, reload=False) 