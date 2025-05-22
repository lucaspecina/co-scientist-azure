"""
Manages research sessions for DeepResearch, allowing to save and restore conversation state.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

class SessionManager:
    def __init__(self, sessions_dir: str = "research_sessions"):
        """Initialize the session manager.
        
        Args:
            sessions_dir: Directory where session files will be stored
        """
        self.sessions_dir = sessions_dir
        self._ensure_sessions_directory()
        self.current_session_id = None
        self.current_session = None
    
    def _ensure_sessions_directory(self):
        """Create the sessions directory if it doesn't exist."""
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir)
    
    def create_session(self, initial_query: str) -> str:
        """Create a new research session.
        
        Args:
            initial_query: The first query that started this research session
            
        Returns:
            session_id: Unique identifier for the session
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session = {
            "session_id": session_id,
            "created_at": timestamp,
            "last_updated": timestamp,
            "initial_query": initial_query,
            "queries": [{
                "query_id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "query": initial_query,
                "context": [],
                "used_tools": [],
                "final_answer": None
            }],
            "metadata": {
                "total_queries": 1,
                "status": "active"
            }
        }
        
        self.current_session_id = session_id
        self.current_session = session
        self._save_session(session)
        
        return session_id
    
    def add_query_to_session(self, query: str, context: List[Dict], used_tools: List[str], final_answer: Optional[str] = None) -> str:
        """Add a new query to the current session.
        
        Args:
            query: The new query
            context: The conversation context for this query
            used_tools: List of tools used in processing this query
            final_answer: The final answer provided for this query
            
        Returns:
            query_id: Unique identifier for the query
        """
        if not self.current_session:
            raise ValueError("No active session")
        
        query_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        query_data = {
            "query_id": query_id,
            "timestamp": timestamp,
            "query": query,
            "context": context,
            "used_tools": used_tools,
            "final_answer": final_answer
        }
        
        self.current_session["queries"].append(query_data)
        self.current_session["last_updated"] = timestamp
        self.current_session["metadata"]["total_queries"] += 1
        
        self._save_session(self.current_session)
        
        return query_id
    
    def load_session(self, session_id: str) -> Dict:
        """Load a specific research session.
        
        Args:
            session_id: The ID of the session to load
            
        Returns:
            The session data
        """
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        if not os.path.exists(session_file):
            raise ValueError(f"Session {session_id} not found")
            
        with open(session_file, 'r', encoding='utf-8') as f:
            session = json.load(f)
            
        self.current_session_id = session_id
        self.current_session = session
        return session
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict:
        """Get a summary of the specified session or current session.
        
        Args:
            session_id: Optional session ID. If None, uses current session
            
        Returns:
            Summary of the session
        """
        session = self.current_session
        if session_id:
            session = self.load_session(session_id)
            
        if not session:
            raise ValueError("No session available")
            
        return {
            "session_id": session["session_id"],
            "created_at": session["created_at"],
            "last_updated": session["last_updated"],
            "initial_query": session["initial_query"],
            "total_queries": session["metadata"]["total_queries"],
            "status": session["metadata"]["status"]
        }
    
    def list_sessions(self) -> List[Dict]:
        """List all available research sessions.
        
        Returns:
            List of session summaries
        """
        sessions = []
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith(".json"):
                session_id = filename[:-5]  # Remove .json
                try:
                    summary = self.get_session_summary(session_id)
                    sessions.append(summary)
                except Exception as e:
                    print(f"Error loading session {session_id}: {e}")
        
        return sorted(sessions, key=lambda x: x["last_updated"], reverse=True)
    
    def _save_session(self, session: Dict):
        """Save the session to disk.
        
        Args:
            session: The session data to save
        """
        session_file = os.path.join(self.sessions_dir, f"{session['session_id']}.json")
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False) 