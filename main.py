"""
DeepResearch with ReAct main script.
Implements a ReAct-based agent for research queries using RAG and web search.
"""

import argparse
import logging
from deepresearch_azure.react_agent import ReActAgent

# Configure logging
def setup_logging(verbose=False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('deepresearch')

def display_sessions():
    """Display available sessions"""
    # Create a temporary agent just to list sessions
    temp_agent = ReActAgent(verbose=False, skip_session_creation=True)
    sessions = temp_agent.list_available_sessions()
    if not sessions:
        print("\nNo previous sessions found.")
        return None
    
    print("\nAvailable sessions:")
    print("-" * 80)
    print(f"{'ID':<36} | {'Created':<19} | {'Queries':<7} | {'Initial Query'}")
    print("-" * 80)
    
    for session in sessions:
        created = session["created_at"][:19]  # Truncate microseconds
        print(f"{session['session_id']:<36} | {created:<19} | {session['total_queries']:<7} | {session['initial_query'][:50]}")
    
    return sessions

def main():
    """Main function for running the DeepResearch agent interactively."""
    parser = argparse.ArgumentParser(description="DeepResearch ReAct Agent (interactive)")
    parser.add_argument("--query", type=str, help="Initial research query to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--session", type=str, help="Session ID to load")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)

    # Welcome banner
    print("\n" + "="*80)
    print("DeepResearch ReAct Agent".center(80))
    print("="*80)

    # Determine which session to use
    session_id = args.session
    if not session_id:
        sessions = display_sessions()
        if sessions:
            load_session = input("\nWould you like to load a previous session? (y/N): ").lower().strip()
            if load_session == 'y':
                session_id = input("Enter the session ID to load: ").strip()

    # Initialize the agent with the chosen session
    agent = ReActAgent(verbose=args.verbose, session_id=session_id)
    
    # If we loaded a session, show its details
    if session_id:
        current_session = agent.get_current_session_summary()
        print(f"\nLoaded session from: {current_session['created_at']}")
        print(f"Initial query: {current_session['initial_query']}")
        print(f"Total queries: {current_session['total_queries']}")

    # Read initial query from file if not provided via argument
    initial_query_from_file = None
    if not args.query:
        try:
            with open("initial_prompt.txt", "r", encoding="utf-8") as f:
                initial_query_from_file = f.read().strip()
            if initial_query_from_file:
                print(f"Loaded initial query from initial_prompt.txt: {initial_query_from_file}")
            else:
                print("initial_prompt.txt is empty. Please provide a query.")
        except FileNotFoundError:
            print("initial_prompt.txt not found. Please provide a query.")
        except Exception as e:
            print(f"Error reading initial_prompt.txt: {e}")

    # Start interactive loop
    next_query = args.query if args.query else initial_query_from_file
    while True:
        # Prompt for query if none
        if not next_query:
            print("\nAvailable commands:")
            print("- Enter your research query")
            print("- 'sessions' to list available sessions")
            print("- 'load <session_id>' to load a specific session")
            print("- 'summary' to see current session summary")
            print("- 'exit' to quit")
            next_query = input("\nEnter command or query: ").strip()

        # Handle commands
        if next_query.lower() == 'sessions':
            display_sessions()
            next_query = None
            continue
        elif next_query.lower().startswith('load '):
            session_id = next_query[5:].strip()
            try:
                agent = ReActAgent(verbose=args.verbose, session_id=session_id)
                current_session = agent.get_current_session_summary()
                print(f"\nLoaded session from: {current_session['created_at']}")
                print(f"Initial query: {current_session['initial_query']}")
                print(f"Total queries: {current_session['total_queries']}")
            except Exception as e:
                print(f"Error loading session: {e}")
            next_query = None
            continue
        elif next_query.lower() == 'summary':
            current_session = agent.get_current_session_summary()
            if current_session:
                print("\nCurrent Session Summary:")
                print("-" * 40)
                print(f"Session ID: {current_session['session_id']}")
                print(f"Created: {current_session['created_at']}")
                print(f"Last Updated: {current_session['last_updated']}")
                print(f"Total Queries: {current_session['total_queries']}")
                print(f"Initial Query: {current_session['initial_query']}")
            else:
                print("\nNo active session")
            next_query = None
            continue
        elif not next_query or next_query.lower() in ("exit", "quit"):
            print("\nThank you for using DeepResearch. Goodbye!")
            break

        # Display the query
        print("\n" + "-"*80)
        print(f"Query: {next_query}")
        print("-"*80)

        # Run the agent
        try:
            result = agent.run(next_query)

            # Show the result
            print("\n" + "="*80)
            print("RESULT".center(80))
            print("="*80)
            print(result)
            print("="*80)

            # Analysis summary
            print("\n" + "-"*80)
            print("ANALYSIS SUMMARY".center(80))
            print("-"*80)
            if agent.used_tools:
                if 'search_rag' in agent.used_tools:
                    print("✓ Performed internal documentation search.")
                else:
                    print("✗ No internal docs search performed.")
                if 'search_web' in agent.used_tools:
                    print("✓ Performed web search.")
                else:
                    print("✗ No web search performed.")
                if 'ask_user' in agent.used_tools:
                    print("✓ Asked clarifying questions to the user.")
                else:
                    print("✗ No clarifying questions were asked.")
            else:
                print("No tools were used in this session.")
            print("-"*80)

            # Show current session info
            current_session = agent.get_current_session_summary()
            if current_session:
                print(f"\nSession ID: {current_session['session_id']}")
                print(f"Total queries in this session: {current_session['total_queries']}")

            # Suggestions to continue the conversation
            print("\nSuggestions:")
            print("- Ask for more methodology details")
            print("- Challenge or refine the conclusion")
            print("- Ask a follow-up question on this topic")
            print("- Start a new topic")
            print("- Type 'sessions' to view or load other sessions")

        except Exception as e:
            print(f"\nError executing query: {e}")

        # Next action prompt
        next_query = input("\nWhat would you like to do next? ").strip()

if __name__ == "__main__":
    main() 