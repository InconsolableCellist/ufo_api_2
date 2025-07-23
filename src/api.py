from fastapi import FastAPI, Query, HTTPException, BackgroundTasks, UploadFile
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import uvicorn
from config import logger, update_llm_config, LLM_CONFIG, MEMORY_PATH, EMOTION_PATH, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, STATE_DIR
import time
import threading
import os
import random
from datetime import datetime

# Initialize FastAPI app first
app = FastAPI(
    title="Agent Simulation API",
    description="API for controlling and monitoring an agent simulation with emotional and cognitive processes",
    version="1.0.0"
)

# Initialize global variables
simulation = None
agent = None
simulation_thread = None
simulation_running = False
stop_event = threading.Event()
processing_manager = None

# Now import other modules 
from services.sqlite_db import db_init, get_messages_with_summary
# Import agent-related modules after initializing app
from agent import Agent
from agent.motivation.task import Task
from simulation import SimulationController
from managers.thought_summary_manager import ThoughtSummaryManager
from services.llm_interface import LLMInterface
# Import these functions only where they're used to avoid circular dependencies
# from services.sqlite_db import insert_request, update_with_response, sync_with_summary_manager

class SimulationConfig(BaseModel):
    memory_path: Optional[str] = os.path.join(STATE_DIR, "agent_memory.pkl")
    emotion_path: Optional[str] = os.path.join(STATE_DIR, "emotion_state.pkl")
    initial_tasks: Optional[List[str]] = ["Meditate on existence", "Explore emotional state"]
    TELEGRAM_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None

class MemoryQuery(BaseModel):
    query: Optional[str] = None
    emotional_state: Optional[Dict[str, float]] = None
    count: Optional[int] = 3
    memory_type: str = "long"  # "short" or "long"

class MemoryTypeQuery(BaseModel):
    thought_type: str  # "normal_thought", "ego_thought", "stimuli_interpretation", "external_info"
    count: int = 5

class MemoryTimeQuery(BaseModel):
    start_time: str  # ISO format datetime string
    end_time: str  # ISO format datetime string
    count: int = 10

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

class MemoryStatsResponse(BaseModel):
    total_memories: int
    by_type: Dict[str, int]
    thinking_cycles: int
    most_recalled: List[List]  # [memory_content, recall_count]

class ThoughtSummaryResponse(BaseModel):
    summaries: List[Dict[str, Any]]
    total_entries: int
    summarized_entries: int

class ThoughtSummaryStatus(BaseModel):
    active: bool
    api_available: bool
    queue_size: int
    total_entries: int
    summarized_entries: int

class JournalResponse(BaseModel):
    content: str
    last_updated: Optional[datetime] = None

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
    global simulation, simulation_running, stop_event
    
    while not stop_event.is_set() and simulation_running:
        try:
            simulation.run_step()
            time.sleep(1)  # Delay between steps
        except Exception as e:
            print(f"Error in simulation loop: {e}")
            simulation_running = False
            break

@app.post("/start", response_model=SimulationStatus)
async def start_simulation(config: SimulationConfig, background_tasks: BackgroundTasks):
    global simulation, agent, simulation_thread, simulation_running, stop_event
    
    if simulation_running:
        raise HTTPException(status_code=400, detail="Simulation is already running")
    
    llm = LLMInterface()
    agent = Agent(
        llm=llm, 
        memory_path=config.memory_path,
        emotion_path=config.emotion_path,
        telegram_token=config.TELEGRAM_TOKEN,
        telegram_chat_id=config.TELEGRAM_CHAT_ID
    )
    simulation = SimulationController(agent)
    
    # Make sure thought summary manager is initialized
    if not hasattr(agent, 'thought_summary_manager') or agent.thought_summary_manager is None:
        agent.thought_summary_manager = ThoughtSummaryManager()
        logger.info("Initialized ThoughtSummaryManager")
    
    # Reset state
    stop_event.clear()
    simulation_running = True
    
    # Start simulation in background
    background_tasks.add_task(run_simulation_loop)
    
    return SimulationStatus(
        running=simulation_running,
        step_count=simulation.cycle_count,
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
        step_count=simulation.cycle_count,
        current_time=simulation.current_time if simulation else None,
        last_result=None
    )

@app.get("/status", response_model=SimulationStatus)
async def get_status():
    global simulation, simulation_running
    
    return SimulationStatus(
        running=simulation_running,
        step_count=simulation.cycle_count if simulation else 0,
        current_time=simulation.current_time if simulation else None,
        last_result=None
    )

@app.post("/next", response_model=Dict[str, Any])
async def next_step():
    global simulation, simulation_running
    
    if not simulation:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if simulation_running:
        raise HTTPException(status_code=400, detail="Cannot manually step while simulation is running automatically")
    
    return simulation.run_step()

@app.get("/summary")
async def get_summary():
    global agent, simulation
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    # Create a summary of the agent's current state
    emotional_state = agent.mind.emotion_center.get_state()
    physical_state = agent.physical_state
    recent_thoughts = list(agent.mind.memory.short_term)
    current_task = agent.mind.motivation_center.get_current_task()
    
    # Get goals data for backward compatibility
    goals = {}
    if hasattr(agent, 'llm') and hasattr(agent.llm, 'tool_registry'):
        goals = agent.llm.tool_registry.get_goals()
    
    return {
        "emotional_state": emotional_state,
        "physical_state": physical_state,
        "recent_thoughts": recent_thoughts,
        "current_task": current_task.description if current_task else None,
        "step_count": simulation.cycle_count if simulation else 0,
        "simulation_time": simulation.current_time if simulation else None,
        "short_term_goals": goals.get("short_term", []),
        "long_term_goal": goals.get("long_term")
    }

@app.post("/memory", response_model=MemoryResponse)
async def query_memory(query_params: MemoryQuery):
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if query_params.memory_type == "short":
        memories = list(agent.mind.memory.short_term)
    else:  # long-term memory
        memories = agent.mind.memory.recall(
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
    # Filter out None values
    updates = {k: v for k, v in config_update.dict().items() if v is not None}
    
    if not updates:
        return {"message": "No updates provided", "current_config": LLM_CONFIG}
    
    update_llm_config(updates)
    return {"message": "Configuration updated", "current_config": LLM_CONFIG}

@app.post("/stimuli")
async def add_stimuli(stimuli: dict):
    global processing_manager
    
    if not processing_manager:
        raise HTTPException(status_code=400, detail="Simulation not started")
        
    processing_manager.add_stimuli(stimuli)
    return {"status": "stimuli added"}

@app.post("/visual_stimuli")
async def add_visual_stimuli(file: UploadFile):
    global processing_manager
    
    if not processing_manager:
        raise HTTPException(status_code=400, detail="Simulation not started")
    
    # Process the uploaded image
    contents = await file.read()
    visual_data = process_image(contents)
    
    # Add to stimuli queue
    processing_manager.add_stimuli({
        "visual_input": visual_data
    })
    
    return {"status": "visual stimuli added"}

@app.post("/memory/by-type", response_model=MemoryResponse)
async def query_memory_by_type(query_params: MemoryTypeQuery):
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    # Ensure thought type exists
    valid_types = ["normal_thought", "ego_thought", "stimuli_interpretation", "external_info"]
    if query_params.thought_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid thought type. Must be one of: {', '.join(valid_types)}"
        )
    
    # Recall memories by type
    memories = agent.mind.memory.recall_by_type(
        thought_type=query_params.thought_type,
        n=query_params.count
    )
    
    return MemoryResponse(
        memories=memories,
        memory_type=query_params.thought_type
    )

@app.post("/memory/by-time", response_model=MemoryResponse)
async def query_memory_by_time(query_params: MemoryTimeQuery):
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    try:
        # Parse times
        start_time = datetime.fromisoformat(query_params.start_time)
        end_time = datetime.fromisoformat(query_params.end_time)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
        )
    
    # Recall memories by time range
    memories = agent.mind.memory.recall_by_time_range(
        start_time=start_time,
        end_time=end_time,
        n=query_params.count
    )
    
    return MemoryResponse(
        memories=memories,
        memory_type="time_range"
    )

@app.get("/memory/stats", response_model=MemoryStatsResponse)
async def get_memory_stats():
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    # Get memory statistics
    stats = agent.mind.memory.get_memory_stats()
    
    return MemoryStatsResponse(
        total_memories=stats["total_memories"],
        by_type=stats["by_type"],
        thinking_cycles=stats["thinking_cycles"],
        most_recalled=stats["most_recalled"]
    )

@app.get("/thought-summaries", response_model=ThoughtSummaryResponse)
async def get_thought_summaries(limit: int = 10, offset: int = 0):
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if not hasattr(agent, 'thought_summary_manager'):
        raise HTTPException(status_code=500, detail="Thought summary manager not initialized")
    
    summaries = agent.thought_summary_manager.get_summaries(limit, offset)
    status = agent.thought_summary_manager.get_summary_status()
    
    return ThoughtSummaryResponse(
        summaries=summaries,
        total_entries=status["total_entries"],
        summarized_entries=status["summarized_entries"]
    )

@app.post("/thought-summaries/start", response_model=ThoughtSummaryStatus)
async def start_thought_summarization():
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if not hasattr(agent, 'thought_summary_manager'):
        raise HTTPException(status_code=500, detail="Thought summary manager not initialized")
    
    success = agent.thought_summary_manager.start_summarization()
    status = agent.thought_summary_manager.get_summary_status()
    
    return ThoughtSummaryStatus(**status)

@app.post("/thought-summaries/stop", response_model=ThoughtSummaryStatus)
async def stop_thought_summarization():
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if not hasattr(agent, 'thought_summary_manager'):
        raise HTTPException(status_code=500, detail="Thought summary manager not initialized")
    
    success = agent.thought_summary_manager.stop_summarization()
    status = agent.thought_summary_manager.get_summary_status()
    
    return ThoughtSummaryStatus(**status)

@app.get("/thought-summaries/status", response_model=ThoughtSummaryStatus)
async def get_thought_summarization_status():
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if not hasattr(agent, 'thought_summary_manager'):
        raise HTTPException(status_code=500, detail="Thought summary manager not initialized")
    
    status = agent.thought_summary_manager.get_summary_status()
    
    return ThoughtSummaryStatus(**status)

@app.post("/thought-summaries/force-summarize")
async def force_summarize_all_thoughts():
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if not hasattr(agent, 'thought_summary_manager'):
        raise HTTPException(status_code=500, detail="Thought summary manager not initialized")
    
    # Force summarization of all pending thoughts
    result = agent.thought_summary_manager.force_summarize_all()
    
    # Get updated status
    status = agent.thought_summary_manager.get_summary_status()
    
    return {
        "forced_summarization_result": result,
        "current_status": status
    }

@app.get("/journal")
async def get_journal():
    global agent
    
    if not agent:
        # Return empty data instead of error
        return {
            "content": "# Agent Journal\n\nNo simulation is currently running.",
            "last_update": datetime.now().isoformat()
        }
    
    if not hasattr(agent, 'journal'):
        # Return empty data instead of error
        return {
            "content": "# Agent Journal\n\nJournal not found.",
            "last_update": datetime.now().isoformat()
        }
    
    # Get the journal path directly from the agent's journal object
    journal_path = agent.journal.file_path
    
    if not os.path.exists(journal_path):
        # Return empty data instead of error
        return {
            "content": "# Agent Journal\n\nJournal file not found.",
            "last_update": datetime.now().isoformat()
        }
    
    try:
        with open(journal_path, 'r') as f:
            content = f.read()
        
        # Get last modified time
        last_update = datetime.fromtimestamp(os.path.getmtime(journal_path))
        
        return {
            "content": content,
            "last_update": last_update.isoformat() if last_update else None
        }
    except Exception as e:
        # Log the error but return empty data
        logger.error(f"Error reading journal: {str(e)}")
        return {
            "content": "# Agent Journal\n\nError reading journal.",
            "last_update": datetime.now().isoformat()
        }

@app.get("/goals")
async def get_goals():
    global agent
    
    if not agent:
        # Return empty data instead of error
        return {
            "short_term_goals": [],
            "long_term_goal": None
        }
    
    if not hasattr(agent, 'llm') or not hasattr(agent.llm, 'tool_registry'):
        # Return empty data instead of error
        return {
            "short_term_goals": [],
            "long_term_goal": None
        }
    
    # Get goals data from the tool registry
    try:
        goals = agent.llm.tool_registry.get_goals()
        
        return {
            "short_term_goals": goals.get("short_term_details", []),
            "long_term_goal": goals.get("long_term_details", None)
        }
    except Exception as e:
        # Log the error but return empty data
        logger.error(f"Error retrieving goals: {str(e)}")
        return {
            "short_term_goals": [],
            "long_term_goal": None
        }

@app.post("/telegram/check-messages")
async def check_telegram_messages():
    """Check for new Telegram messages and return the count of unread messages."""
    global agent
    
    if not agent:
        raise HTTPException(status_code=400, detail="Simulation has not been started")
    
    if not hasattr(agent, 'telegram_bot') or not agent.telegram_bot:
        raise HTTPException(status_code=400, detail="Telegram bot not configured")
    
    try:
        # Check for new messages
        new_messages = agent.telegram_bot.check_new_messages()
        
        # Get all unread messages
        unread_messages = agent.telegram_bot.get_unread_messages()
        
        return {
            "success": True,
            "new_messages_count": len(new_messages),
            "unread_messages_count": len(unread_messages),
            "has_unread_messages": len(unread_messages) > 0
        }
    except Exception as e:
        logger.error(f"Error checking Telegram messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error checking messages: {str(e)}")

def process_image(image_data):
    """Process raw image data and extract features"""
    # In a real implementation, you would use a CV library or ML model
    # This is a placeholder
    return {
        "image_type": "webcam",
        "timestamp": datetime.now().isoformat(),
        "raw_data": "<binary data placeholder>",
        "features": {
            "brightness": 0.7,
            "dominant_colors": ["blue", "green"],
            "detected_objects": ["person", "desk"],
            "movement_detected": True
        }
    }

@app.get("/memory/all")
async def get_all_memories():
    global agent
    
    if not agent:
        # Return empty data instead of error
        return {
            "short_term": [],
            "long_term": []
        }
    
    try:
        # Get both long-term and short-term memories
        short_term = list(agent.mind.memory.short_term)
        
        # Get all long-term memories, not just through recall
        long_term = []
        if hasattr(agent.mind.memory, 'long_term'):
            long_term = agent.mind.memory.long_term
        
        # Format the memory data similar to agent_viewer.py
        memory_data = {
            "short_term": short_term,
            "long_term": long_term
        }
        
        return memory_data
        
    except Exception as e:
        # Log the error but return empty data
        logger.error(f"Error retrieving memories: {str(e)}")
        return {
            "short_term": [],
            "long_term": []
        }

@app.get("/tools/history")
async def get_tool_history():
    global agent
    
    if not agent:
        # Return empty data instead of error
        return {
            "tool_history": []
        }
    
    if not hasattr(agent, 'llm') or not hasattr(agent.llm, 'tool_registry'):
        # Return empty data instead of error
        return {
            "tool_history": []
        }
    
    try:
        # Get tool history from the tool registry
        tool_history = agent.llm.tool_registry.tool_history
        
        return {
            "tool_history": tool_history
        }
        
    except Exception as e:
        # Log the error but return empty data
        logger.error(f"Error retrieving tool history: {str(e)}")
        return {
            "tool_history": []
        }

@app.post("/thought-summaries/sync-db")
async def sync_db_with_summary_manager():
    """Sync unsummarized database messages with the thought summary manager."""
    global agent
    
    if not agent or not hasattr(agent, 'thought_summary_manager'):
        raise HTTPException(status_code=400, detail="ThoughtSummaryManager not initialized")
    
    try:
        # Import the function only when needed
        from services.sqlite_db import sync_with_summary_manager
        sync_count = sync_with_summary_manager(agent.thought_summary_manager)
        return {
            "success": True,
            "message": f"Synced {sync_count} messages with thought summary manager",
            "count": sync_count
        }
    except Exception as e:
        logger.error(f"Error syncing database with thought summary manager: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/thought-summaries/db")
async def get_summarized_thoughts(limit: int = 20, offset: int = 0):
    """Get messages that have summaries from the database."""
    try:
        messages = get_messages_with_summary(limit, offset)
        return {
            "success": True,
            "messages": messages,
            "count": len(messages),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error getting messages with summaries: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# TODO: Add function to record agent messages to users
# TODO: Add function to retrieve conversation history with pagination
# TODO: Add function to clear conversation history

if __name__ == "__main__":
    db_init()
    # Create a temporary ThoughtSummaryManager just to sync existing messages
    try:
        temp_thought_manager = ThoughtSummaryManager()
        # Import the function only when needed
        from services.sqlite_db import sync_with_summary_manager
        sync_count = sync_with_summary_manager(temp_thought_manager)
        logger.info(f"Synced {sync_count} messages with thought summary manager")
    except Exception as e:
        logger.error(f"Error syncing messages at startup: {e}")
    
    uvicorn.run("api:app", host="0.0.0.0", port=8081, reload=False) 