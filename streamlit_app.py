import streamlit as st
import os
from deepresearch_azure.react_agent import ReActAgent
from deepresearch_azure.session_manager import SessionManager
import json
import uuid
import time
import sys
import re

PASSWORD = "ytec"  # Change this to your desired password

# Password check
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

st.set_page_config(page_title="Co-Scientist Azure", layout="wide")

st.title("Co-Scientist Azure - Research Agent GUI")

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = ReActAgent(verbose=True, gui_mode=True)
if 'output_log' not in st.session_state:
    st.session_state.output_log = []
if 'pending_query' not in st.session_state:
    st.session_state.pending_query = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'sessions' not in st.session_state:
    st.session_state.sessions = st.session_state.agent.list_available_sessions()
if 'pending_action' not in st.session_state:
    st.session_state.pending_action = None

# Sidebar for sessions
with st.sidebar:
    st.header("Sessions")
    if st.button("Refresh Sessions"):
        st.session_state.sessions = st.session_state.agent.list_available_sessions()
    
    selected_session = st.selectbox(
        "Load Session",
        options=["New Session"] + [s['session_id'] for s in st.session_state.sessions]
    )
    
    if selected_session != "New Session" and selected_session != st.session_state.session_id:
        st.session_state.agent = ReActAgent(verbose=True, session_id=selected_session, gui_mode=True)
        st.session_state.session_id = selected_session
        st.session_state.output_log = []
        st.session_state.pending_query = None
        st.session_state.pending_action = None
        st.success(f"Loaded session {selected_session}")

# Main area
col1, col2 = st.columns([3,1])

with col1:
    st.header("Research Query")
    query = st.text_area("Enter your research query:", height=100)
    if st.button("Start Research") and query:
        st.session_state.agent.start_new_query(query)
        st.session_state.output_log = [f"**Query:** {query}"]
        st.session_state.pending_query = None
        st.session_state.session_id = st.session_state.agent.session_manager.current_session['session_id']
        st.info("Query initialized. Use 'Step' or 'Run Until Checkpoint or Input' to advance the research process.")
        st.rerun()

    # Display output log in chat-like format
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.output_log:
            lines = msg.split('\n')
            for line in lines:
                if line.startswith('Assistant: Thought:'):
                    st.markdown(f'**Thought:** {line.split("Thought:", 1)[1]}', unsafe_allow_html=True)
                elif line.startswith('Action:'):
                    st.markdown('**Action:**', unsafe_allow_html=True)
                    # Extract JSON and display as code
                    json_match = re.search(r'\{.*\}', msg, re.DOTALL)
                    if json_match:
                        st.code(json_match.group(0), language='json')
                elif line.startswith('[USING') or line.startswith('Observation:'):
                    with st.expander("Show Details"):
                        st.markdown(f'<div style="font-size:14px; white-space: pre-wrap; background-color: #d0d0d0; color: black; padding: 10px; border-radius: 5px;">{line}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="font-size:14px; white-space: pre-wrap;">{line}</div>', unsafe_allow_html=True)

    # Handle pending user input
    if st.session_state.pending_query:
        st.subheader("Agent Question")
        st.write(st.session_state.pending_query)
        user_response = st.text_input("Your response:")
        if st.button("Submit Response") and user_response:
            st.session_state.agent.provide_user_response(user_response)
            st.session_state.pending_query = None
            st.session_state.output_log.append(f"**User response:** {user_response}")
            st.rerun()

    # Step button - always at the bottom
    if st.button("Step"):
        if st.session_state.pending_action:
            with st.spinner("Executing action..."):
                exec_result = st.session_state.agent.execute_parsed_action(st.session_state.pending_action)
                st.session_state.output_log.append(exec_result['output'])
                st.session_state.pending_action = None
                if exec_result['type'] == 'ask_user':
                    st.session_state.pending_query = exec_result['query']
                elif exec_result['type'] == 'checkpoint':
                    st.session_state.output_log.append(f"**Result:** {exec_result['answer']}")
                elif exec_result['type'] == 'error':
                    st.error("Error executing action.")
            st.rerun()
        elif not st.session_state.pending_query:
            if st.session_state.agent.context:
                with st.spinner("Generating response..."):
                    gen_result = st.session_state.agent.generate_response()
                    st.session_state.output_log.append(gen_result['output'])
                    if gen_result['type'] == 'response':
                        st.session_state.pending_action = gen_result['action']
                    elif gen_result['type'] == 'error':
                        st.error("Error generating response.")
                st.rerun()
            else:
                st.warning("Start a query first.")
        else:
            st.info("Submit user response first to continue.")

with col2:
    st.header("Controls")
    
    if st.button("Run Until Checkpoint or Input"):
        while not st.session_state.pending_query and not st.session_state.pending_action:
            # Generate phase
            with st.spinner("Generating response..."):
                gen_result = st.session_state.agent.generate_response()
                print(gen_result['output'])
                st.session_state.output_log.append(gen_result['output'])
                if gen_result['type'] == 'error':
                    st.error("Error generating response.")
                    break
                st.session_state.pending_action = gen_result['action']
            st.rerun()
            
            # Execute phase (on next run, but since loop, it will continue after rerun)
            if st.session_state.pending_action:
                with st.spinner("Executing action..."):
                    exec_result = st.session_state.agent.execute_parsed_action(st.session_state.pending_action)
                    print(exec_result['output'])
                    st.session_state.output_log.append(exec_result['output'])
                    st.session_state.pending_action = None
                    if exec_result['type'] == 'ask_user':
                        st.session_state.pending_query = exec_result['query']
                        break
                    elif exec_result['type'] == 'checkpoint':
                        st.session_state.output_log.append(f"**Result:** {exec_result['answer']}")
                        break
                    elif exec_result['type'] == 'error':
                        st.error("Error executing action.")
                        break
                st.rerun()

    # Session info
    if st.session_state.session_id:
        summary = st.session_state.agent.get_current_session_summary()
        with st.expander("Session Summary"):
            st.json(summary)

    # Details toggle (expander for full logs)
    with st.expander("Detailed Logs"):
        st.text_area("Full Output Log", value='\n'.join(st.session_state.output_log), height=300) 