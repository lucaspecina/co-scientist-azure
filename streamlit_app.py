import streamlit as st
import os
from deepresearch_azure.react_agent import ReActAgent
from deepresearch_azure.session_manager import SessionManager
import json
import uuid
import time

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
        st.session_state.output_log = [f"Query: {query}"]
        st.session_state.pending_query = None
        st.session_state.session_id = st.session_state.agent.session_manager.current_session['session_id']
        st.session_state.auto_run = True  # Flag to trigger auto run

    # Display output log in chat-like format
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.output_log:
            st.markdown(msg)

    # Handle pending user input
    if st.session_state.pending_query:
        st.subheader("Agent Question")
        st.write(st.session_state.pending_query)
        user_response = st.text_input("Your response:")
        if st.button("Submit Response") and user_response:
            st.session_state.agent.provide_user_response(user_response)
            st.session_state.pending_query = None
            st.session_state.output_log.append(f"User response: {user_response}")
            st.session_state.auto_run = True  # Resume auto-run after response
            st.rerun()  # Immediately rerun to trigger the auto-run loop

    if 'auto_run' in st.session_state and st.session_state.auto_run:
        st.session_state.auto_run = False
        progress_bar = st.progress(0)
        status_text = st.empty()
        i = 0
        max_steps = 20  # Arbitrary max to prevent infinite loop
        while i < max_steps:
            if st.session_state.pending_query:
                break
            status_text.text(f"Processing step {i+1}...")
            step_result = st.session_state.agent.step()
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
            progress_bar.progress((i + 1) / max_steps)
            i += 1
            # To give a sense of progress, but Streamlit won't update until end
        progress_bar.empty()
        status_text.empty()
        st.rerun()  # Rerun to update the UI

with col2:
    st.header("Controls")
    if st.button("Step"):
        if st.session_state.agent.context:  # Check if a query has been started
            step_result = st.session_state.agent.step()
            st.session_state.output_log.append(step_result['output'])
            if step_result['type'] == 'ask_user':
                st.session_state.pending_query = step_result['query']
            elif step_result['type'] == 'checkpoint':
                st.session_state.output_log.append(f"**Result:** {step_result['answer']}")
            elif step_result['type'] == 'error':
                st.error("An error occurred during processing.")
        else:
            st.warning("Please start a research query first.")

    if st.button("Run Until Checkpoint or Input"):
        while True:
            if st.session_state.pending_query:
                break
            step_result = st.session_state.agent.step()
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

    # Session info
    if st.session_state.session_id:
        summary = st.session_state.agent.get_current_session_summary()
        with st.expander("Session Summary"):
            st.json(summary)

    # Details toggle (expander for full logs)
    with st.expander("Detailed Logs"):
        st.text_area("Full Output Log", value='\n'.join(st.session_state.output_log), height=300) 