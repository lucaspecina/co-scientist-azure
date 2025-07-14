import streamlit as st
import json
import time
from datetime import datetime
from deepresearch_azure.react_agent import ReActAgent
from deepresearch_azure.search_tools import AskUserTool
import logging

# Configure page
st.set_page_config(
    page_title="Co-Scientist Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .session-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .tool-call {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-left: 4px solid #1976d2;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        color: #1565c0;
    }
    .observation {
        background-color: #e8f5e8;
        padding: 0.8rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        color: #2e7d32;
    }
    .reasoning {
        background-color: #fff8e1;
        padding: 0.8rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        color: #e65100;
    }
    .user-input-required {
        background-color: #fff2f0;
        padding: 1rem;
        border: 2px solid #ff4d4f;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .final-answer {
        background-color: #f6ffed;
        padding: 1.5rem;
        border: 2px solid #52c41a;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitReActAgent(ReActAgent):
    """Extended ReAct agent for Streamlit integration"""
    
    def __init__(self, verbose=False, session_id=None, skip_session_creation=False):
        super().__init__(verbose, session_id, skip_session_creation)
        self.execution_log = []
        self.waiting_for_user_input = False
        self.user_input_query = ""
        self.user_response = None
        self.log_container = None  # Will be set by Streamlit app
        
    def run(self, query):
        """Run the ReAct agent on a query - EXACTLY like original but with UI logging"""
        self.logger.info(f"Running agent with query: {query}")
        
        # Log the initial query
        self.execution_log.append({
            "type": "query",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "query": query
        })
        self._update_ui()
        
        if self.skip_session_creation:
            raise ValueError("This agent instance was created for listing sessions only")
        
        # Initialize system prompt at the start
        from deepresearch_azure.prompts import REACT_PROMPT
        import deepresearch_azure.config as config
        
        system_prompt = REACT_PROMPT.system_prompt.replace("{tools}", self.tools_description)
        
        # If no active session, create one and initialize context
        if not self.session_manager.current_session:
            self.session_manager.create_session(query)
            self.context = []
            self.used_tools = set()  # Reset tools for new session
        
        # Format the initial message with instructions
        initial_message = f"""
{query}

IMPORTANT INSTRUCTIONS:
You have to approach research like a human researcher collaborating with you:

1. You have to first reflect on your question to understand what you're asking and plan your approach.
2. You have main research tools:
   - search_rag: For searching internal documents and research papers
   - search_web: For searching public information on the internet
   - search_arxiv: For searching academic papers on Arxiv.org
   - read_paper: For downloading and reading academic papers
   - ask_user: Ask the user (supervisor) for feedback, clarification, or scope (don't use it unless you really need to)

3. For technical questions like "How can I quantify paraffin content in crude oil?", you have to check both internal resources and public information, asking clarifying questions when needed.

4. For factual questions like sports results, you have to primarily use web search and provide direct answers when available.

5. For company-specific questions like financial results, you have to prioritize internal documents while confirming with me if you need more context.

6. You have to think critically throughout the process - planning, analyzing, reconsidering approaches and ensuring you're addressing the needs effectively.

**ALWAYS CALL AN ACTION, don't forget about it.**

Remember: Your answers are checkpoints in an ongoing conversation. The user may provide feedback or ask follow-up questions.
"""
        
        # For existing sessions with context, add a separator
        if self.context:
            self.context.append({
                "role": "system",
                "content": "\n=== New Research Question ===\n"
            })
            self.used_tools = set()  # Reset tools for new query
        
        # Add the new query with instructions
        self.context.append({"role": "user", "content": initial_message})
        self.original_query = query
        
        iteration = 0
        final_answer = None
        while iteration < config.MAX_ITERATIONS:
            iteration += 1
            
            # Log iteration start
            self.execution_log.append({
                "type": "iteration_start",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "iteration": iteration
            })
            self._update_ui()
            
            try:
                # Generate the next action
                self.logger.info("Generating model response")
                messages = [{"role": "system", "content": system_prompt}]  # Always include system prompt
                messages.extend(self.context)  # Add conversation context

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                
                assistant_message = response.choices[0].message.content
                
                # Log reasoning
                self.execution_log.append({
                    "type": "reasoning",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "content": assistant_message
                })
                self._update_ui()
                
                self.context.append({"role": "assistant", "content": assistant_message})

                # Parse and execute the action
                action = self._parse_action(assistant_message)
                if not action:
                    self.logger.warning("Failed to parse action, asking for clarification")
                    self.context.append({"role": "user", "content": "I couldn't understand your action. Please provide a valid action in the format: Action: {\"name\": \"tool_name\", \"arguments\": {\"query\": \"your query\"}}."})
                    continue
                
                # Execute the action
                self.logger.info(f"Executing action: {action.get('name')}")
                result = self._execute_action(action)
                
                # Handle special case for user input - STOP EXECUTION AND WAIT
                if result.get("result") == "WAITING_FOR_USER_INPUT":
                    return "WAITING_FOR_USER_INPUT"
                
                # Handle checkpoint (previously final answer)
                if result.get("is_checkpoint"):
                    self.logger.info("Checkpoint reached")
                    final_answer = result["result"]
                    # Save the query and its context to the session
                    self.session_manager.add_query_to_session(
                        query=query,
                        context=self.context,
                        used_tools=list(self.used_tools),
                        final_answer=final_answer
                    )
                    
                    # Add the checkpoint answer to the context
                    self.context.append({
                        "role": "assistant",
                        "content": f"Here's what I've found so far:\n\n{final_answer}\n\nWould you like me to explore any specific aspect further or do you have any questions about this?"
                    })
                    
                    return final_answer
                    
                # Format observation with "Observation:" prefix to match examples in prompts.py
                observation = f"Observation: {result['result']}"
                self.context.append({"role": "user", "content": observation})
                self.logger.info("Added observation to context")
                
            except Exception as e:
                self.logger.error(f"Error during iteration {iteration}: {e}")
                error_msg = f"Error: {str(e)}"
                # Save the failed query attempt but maintain context
                self.session_manager.add_query_to_session(
                    query=query,
                    context=self.context,
                    used_tools=list(self.used_tools),
                    final_answer=error_msg
                )
                return error_msg
        
        # If we reach the maximum number of iterations, save and return the last response
        self.logger.warning(f"Maximum iterations ({config.MAX_ITERATIONS}) reached without final answer")
        max_iter_msg = "Maximum iterations reached without a final answer."
        self.session_manager.add_query_to_session(
            query=query,
            context=self.context,
            used_tools=list(self.used_tools),
            final_answer=max_iter_msg
        )
        return max_iter_msg
    
    def _update_ui(self):
        """Update the Streamlit UI with current execution log"""
        if self.log_container is not None:
            with self.log_container:
                display_execution_log(self.execution_log, st.session_state.get('show_details', False))
        
    def _execute_action(self, action):
        """Override to handle ask_user tool specially for Streamlit"""
        name = action.get("name")
        arguments = action.get("arguments", {})
        
        # Log the action
        self.execution_log.append({
            "type": "action",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "action": name,
            "arguments": arguments
        })
        self._update_ui()
        
        # Handle ask_user specially for Streamlit
        if name == "ask_user":
            query = arguments.get("query", "")
            self.waiting_for_user_input = True
            self.user_input_query = query
            
            # Return a special marker that will be handled by the Streamlit app
            return {
                "result": "WAITING_FOR_USER_INPUT",
                "is_final": False,
                "user_query": query
            }
        
        # For all other tools, execute normally
        result = super()._execute_action(action)
        
        # Log the observation
        self.execution_log.append({
            "type": "observation", 
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "result": result.get("result", "")
        })
        self._update_ui()
        
        return result
    
    def set_user_response(self, user_response):
        """Set user response and continue execution"""
        self.waiting_for_user_input = False
        self.user_response = user_response
        
        # Log the user response
        self.execution_log.append({
            "type": "user_response",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "query": self.user_input_query,
            "response": user_response
        })
        self._update_ui()
        
        # Format and add the observation to context
        observation = f"Observation: User response to '{self.user_input_query}': {user_response}"
        self.context.append({"role": "user", "content": observation})
        
        return observation

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'execution_in_progress' not in st.session_state:
        st.session_state.execution_in_progress = False
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'show_details' not in st.session_state:
        st.session_state.show_details = False
    if 'paused_for_user' not in st.session_state:
        st.session_state.paused_for_user = False

def display_execution_log(log_entries, show_details=False):
    """Display the execution log in a formatted way"""
    if not log_entries:
        return
        
    for entry in log_entries:
        timestamp = entry.get("timestamp", "")
        
        if entry["type"] == "query":
            query = entry["query"]
            
            with st.container():
                st.markdown(f"""
                <div class="reasoning">
                    <strong>❓ Research Query [{timestamp}]:</strong><br>
                    {query}
                </div>
                """, unsafe_allow_html=True)
                
        elif entry["type"] == "iteration_start":
            iteration = entry["iteration"]
            
            with st.container():
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 0.5rem; border-radius: 0.25rem; margin: 0.5rem 0; text-align: center; color: #1565c0; border: 1px solid #1976d2;">
                    <strong>🔄 Iteration {iteration} [{timestamp}]</strong>
                </div>
                """, unsafe_allow_html=True)
                
        elif entry["type"] == "reasoning":
            content = entry["content"]
            
            with st.container():
                if show_details:
                    st.markdown(f"""
                    <div class="reasoning">
                        <strong>🧠 Agent Reasoning [{timestamp}]:</strong><br>
                        <pre>{content}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Show a condensed version
                    lines = content.split('\n')
                    preview = '\n'.join(lines[:3]) + ('...' if len(lines) > 3 else '')
                    
                    with st.expander(f"🧠 Agent Reasoning [{timestamp}]", expanded=False):
                        st.text(content)
                        
        elif entry["type"] == "action":
            action_name = entry["action"]
            arguments = entry.get("arguments", {})
            
            with st.container():
                st.markdown(f"""
                <div class="tool-call">
                    <strong>🔧 Tool Call [{timestamp}]:</strong> {action_name}<br>
                    <small>Arguments: {json.dumps(arguments, indent=2) if show_details else str(arguments)}</small>
                </div>
                """, unsafe_allow_html=True)
                
        elif entry["type"] == "observation":
            result = entry["result"]
            
            with st.container():
                if show_details:
                    st.markdown(f"""
                    <div class="observation">
                        <strong>👁️ Observation [{timestamp}]:</strong><br>
                        <pre>{str(result)}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Show condensed version for long results
                    result_str = str(result)
                    if len(result_str) > 300:
                        with st.expander(f"👁️ Observation [{timestamp}] - {len(result_str)} chars", expanded=False):
                            st.text(result_str)
                    else:
                        st.markdown(f"""
                        <div class="observation">
                            <strong>👁️ Observation [{timestamp}]:</strong><br>
                            {result_str}
                        </div>
                        """, unsafe_allow_html=True)
                
        elif entry["type"] == "user_response":
            query = entry["query"]
            response = entry["response"]
            
            with st.container():
                st.markdown(f"""
                <div class="observation">
                    <strong>💬 User Response [{timestamp}]:</strong><br>
                    <strong>Question:</strong> {query}<br>
                    <strong>Answer:</strong> {response}
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔬 Co-Scientist Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar for session management
    with st.sidebar:
        st.header("📁 Session Management")
        
        # List existing sessions
        if st.button("🔄 Refresh Sessions"):
            # Force refresh by creating a temp agent
            temp_agent = StreamlitReActAgent(skip_session_creation=True)
            sessions = temp_agent.list_available_sessions()
            st.session_state.available_sessions = sessions
        
        # Show available sessions
        if 'available_sessions' not in st.session_state:
            temp_agent = StreamlitReActAgent(skip_session_creation=True)
            st.session_state.available_sessions = temp_agent.list_available_sessions()
        
        if st.session_state.available_sessions:
            st.subheader("Previous Sessions")
            for session in st.session_state.available_sessions[:10]:  # Show last 10
                session_id = session["session_id"]
                created = session["created_at"][:19]
                initial_query = session["initial_query"][:50] + "..." if len(session["initial_query"]) > 50 else session["initial_query"]
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"{created}\n{initial_query}")
                with col2:
                    if st.button("Load", key=f"load_{session_id}"):
                        st.session_state.agent = StreamlitReActAgent(session_id=session_id)
                        st.session_state.current_session_id = session_id
                        st.success("Session loaded!")
                        st.rerun()
        
        # Current session info
        if st.session_state.agent and st.session_state.current_session_id:
            st.subheader("Current Session")
            session_summary = st.session_state.agent.get_current_session_summary()
            if session_summary:
                st.markdown(f"""
                <div class="session-info">
                    <strong>Session ID:</strong> {session_summary['session_id'][:8]}...<br>
                    <strong>Created:</strong> {session_summary['created_at'][:19]}<br>
                    <strong>Queries:</strong> {session_summary['total_queries']}<br>
                    <strong>Initial Query:</strong> {session_summary['initial_query'][:100]}...
                </div>
                """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Research Query")
        
        # Query input
        query = st.text_area(
            "Enter your research question:",
            height=100,
            placeholder="e.g., How can I quantify paraffin content in crude oil?"
        )
        
        # Execution controls
        col_run, col_new = st.columns(2)
        
        with col_run:
            run_disabled = not query or st.session_state.execution_in_progress
            if st.button("🔍 Run Research", disabled=run_disabled, type="primary"):
                st.session_state.execution_in_progress = True
                st.session_state.current_result = None
                st.session_state.current_query = query
                st.session_state.paused_for_user = False
                
                # Initialize agent if needed
                if not st.session_state.agent:
                    st.session_state.agent = StreamlitReActAgent(verbose=True)
                
                st.rerun()
        
        with col_new:
            if st.button("🆕 New Session"):
                st.session_state.agent = None
                st.session_state.current_session_id = None
                st.session_state.current_result = None
                st.session_state.execution_in_progress = False
                st.session_state.current_query = ""
                st.session_state.paused_for_user = False
                st.rerun()
    
    with col2:
        st.header("⚙️ Execution Settings")
        st.session_state.show_details = st.checkbox("Show detailed logs", value=st.session_state.show_details)
        auto_scroll = st.checkbox("Auto-scroll to bottom", value=True)
    
    # Handle execution
    if st.session_state.execution_in_progress:
        # Check if we're waiting for user input
        if st.session_state.agent and st.session_state.agent.waiting_for_user_input:
            st.markdown(f"""
            <div class="user-input-required">
                <h3>🤔 Agent needs your input:</h3>
                <p><strong>{st.session_state.agent.user_input_query}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            user_response = st.text_input("Your response:", key="user_input")
            
            col_submit, col_cancel = st.columns(2)
            with col_submit:
                if st.button("Submit Response", type="primary") and user_response.strip():
                    # Set user response and continue
                    st.session_state.agent.set_user_response(user_response.strip())
                    
                    # Continue execution - simply call run again, it will resume
                    try:
                        with st.spinner("🔄 Agent is continuing research..."):
                            result = st.session_state.agent.run(st.session_state.current_query)
                            
                            if result == "WAITING_FOR_USER_INPUT":
                                st.rerun()  # Wait for more user input
                            else:
                                st.session_state.current_result = result
                                st.session_state.execution_in_progress = False
                                st.success("Research completed!")
                                st.rerun()
                                
                    except Exception as e:
                        st.error(f"Error during continued execution: {str(e)}")
                        st.session_state.execution_in_progress = False
                        st.rerun()
            
            with col_cancel:
                if st.button("Cancel Execution"):
                    st.session_state.execution_in_progress = False
                    st.session_state.agent.waiting_for_user_input = False
                    st.rerun()
        
        else:
            # Start execution
            try:
                with st.spinner("🤖 Agent is thinking and researching..."):
                    # Set up the log container for real-time updates
                    if st.session_state.agent:
                        st.session_state.agent.log_container = st.container()
                    
                    result = st.session_state.agent.run(st.session_state.current_query)
                    
                    # Check if we hit a user input requirement
                    if result == "WAITING_FOR_USER_INPUT":
                        st.rerun()  # Refresh to show user input prompt
                    else:
                        st.session_state.current_result = result
                        st.session_state.execution_in_progress = False
                        st.success("Research completed!")
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error during execution: {str(e)}")
                st.session_state.execution_in_progress = False
                st.rerun()
    
    # Display execution log
    if st.session_state.agent and st.session_state.agent.execution_log:
        st.header("📋 Execution Log")
        
        # Create a container for the log that can be scrolled
        log_container = st.container()
        with log_container:
            display_execution_log(st.session_state.agent.execution_log, st.session_state.show_details)
        
        # Auto-scroll to bottom
        if auto_scroll:
            st.markdown("""
            <script>
                var element = document.querySelector('[data-testid="stVerticalBlock"]');
                element.scrollTop = element.scrollHeight;
            </script>
            """, unsafe_allow_html=True)
    
    # Display final result
    if st.session_state.current_result:
        st.markdown(f"""
        <div class="final-answer">
            <h2>🎯 Research Results</h2>
            <div>{st.session_state.current_result}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Follow-up input
        if not st.session_state.execution_in_progress:
            st.subheader("💭 Follow-up Question")
            follow_up = st.text_input("Ask a follow-up question based on these results:", key="follow_up")
            if st.button("🔄 Continue Research", disabled=not follow_up.strip()) and follow_up.strip():
                st.session_state.execution_in_progress = True
                st.session_state.current_result = None
                st.session_state.current_query = follow_up.strip()
                st.rerun()

if __name__ == "__main__":
    main() 