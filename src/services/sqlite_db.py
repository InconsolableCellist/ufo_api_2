import sqlite3
import json
from datetime import datetime
from config import DB_FILE
import logging

def db_init():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS messages (request_id TEXT PRIMARY KEY, request TEXT, response TEXT, summary_thoughts TEXT, timestamp TIMESTAMP)")
    conn.commit()
    conn.close()

def insert_request(request_id, request):
    """Insert a new request into the database with an empty response."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Convert dict to JSON string if needed
    if isinstance(request, dict):
        request = json.dumps(request)
    
    cursor.execute("INSERT INTO messages (request_id, request, response, summary_thoughts, timestamp) VALUES (?, ?, ?, ?, ?)", 
                 (request_id, request, "", "", datetime.now()))
    conn.commit()
    conn.close()

def update_with_response(request_id, response):
    """Update an existing request with its response."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Convert dict to JSON string if needed
    if isinstance(response, dict):
        response = json.dumps(response)
    
    cursor.execute("UPDATE messages SET response = ?, timestamp = ? WHERE request_id = ?", 
                 (response, datetime.now(), request_id))
    conn.commit()
    conn.close()

def update_with_summary(request_id, summary):
    """Update an existing message with its summary.
    
    Args:
        request_id (str): The request ID to update
        summary (str or dict): The summary to store
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if request_id exists
        cursor.execute("SELECT COUNT(*) FROM messages WHERE request_id = ?", (request_id,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            logging.warning(f"Request ID {request_id} not found in database")
            conn.close()
            return False
        
        # Convert dict to JSON string if needed
        if isinstance(summary, dict):
            summary = json.dumps(summary)
        
        cursor.execute("UPDATE messages SET summary_thoughts = ? WHERE request_id = ?", 
                     (summary, request_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error updating summary for request_id {request_id}: {e}")
        try:
            if 'conn' in locals() and conn:
                conn.close()
        except Exception:
            pass
        return False

def get_messages():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages")
    messages = cursor.fetchall()
    conn.close()
    return messages

def db_get_messages_by_request_id(request_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM messages WHERE request_id = ?", (request_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

def get_messages_without_summary():
    """Get all messages that don't have a summary yet."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT request_id, response FROM messages WHERE response != '' AND (summary_thoughts = '' OR summary_thoughts IS NULL)")
    messages = cursor.fetchall()
    conn.close()
    return messages

def sync_with_summary_manager(thought_summary_manager):
    """Sync unsummarized messages with the thought summary manager.
    
    Args:
        thought_summary_manager: The ThoughtSummaryManager instance
        
    Returns:
        int: Number of messages queued for summarization
    """
    if not thought_summary_manager:
        return 0
        
    unsummarized = get_messages_without_summary()
    count = 0
    
    for request_id, response in unsummarized:
        if response and len(response) > 50:
            thought_summary_manager.add_thought(response, thought_type="normal_thought", request_id=request_id)
            count += 1
    
    return count

def get_messages_with_summary(limit=20, offset=0):
    """Get messages that have summaries.
    
    Args:
        limit (int): Maximum number of messages to return
        offset (int): Number of messages to skip
        
    Returns:
        list: List of messages with summaries
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT request_id, request, response, summary_thoughts, timestamp 
        FROM messages 
        WHERE summary_thoughts IS NOT NULL AND summary_thoughts != '' 
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))
    messages = cursor.fetchall()
    conn.close()
    
    # Format the results as dictionaries
    result = []
    for msg in messages:
        request_id, request, response, summary, timestamp = msg
        
        # Try to parse JSON strings
        try:
            if request and request.startswith('{'):
                request = json.loads(request)
        except:
            pass
            
        result.append({
            "request_id": request_id,
            "request": request,
            "response": response,
            "summary": summary,
            "timestamp": timestamp
        })
        
    return result

