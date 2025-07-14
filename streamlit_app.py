import streamlit as st
import os
from deepresearch_azure.react_agent import ReActAgent
from deepresearch_azure.session_manager import SessionManager
import json
import uuid
import time
import sys
import re

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

with col2:
    st.header("Controls")
    if st.button("Step"):
        with st.spinner("Processing single step..."):
            if st.session_state.agent.context:  # Check if a query has been started
                step_result = st.session_state.agent.step()
                # Duplicate to terminal
                print(step_result['output'])
                st.session_state.output_log.append(step_result['output'])
                if step_result['type'] == 'ask_user':
                    st.session_state.pending_query = step_result['query']
                elif step_result['type'] == 'checkpoint':
                    st.session_state.output_log.append(f"**Result:** {step_result['answer']}")
                elif step_result['type'] == 'error':
                    st.error("An error occurred during processing.")
            else:
                st.warning("Please start a research query first.")
        st.rerun()  # Rerun to update the UI

    if st.button("Run Until Checkpoint or Input"):
        with st.spinner("Running until checkpoint or user input..."):
            while True:
                if st.session_state.pending_query:
                    break
                step_result = st.session_state.agent.step()
                # Duplicate to terminal
                print(step_result['output'])
                st.session_state.output_log.append(step_result['output'])
                if step_result['type'] == 'ask_user':
                    st.session_state.pending_query = step_result['query']
                    break
                elif step_result['type'] == 'checkpoint':
                    st.session_state.output_log.append(f"**Result:** {step_result['answer']}")
                    break
                elif step_result['type'] == 'error':
                    st.error("An error occurred during processing.")
                    break
            st.rerun()  # Rerun to update the UI

    # Session info
    if st.session_state.session_id:
        summary = st.session_state.agent.get_current_session_summary()
        with st.expander("Session Summary"):
            st.json(summary)

    # Details toggle (expander for full logs)
    with st.expander("Detailed Logs"):
        st.text_area("Full Output Log", value='\n'.join(st.session_state.output_log), height=300) 