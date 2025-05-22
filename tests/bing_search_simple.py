from dotenv import load_dotenv
import os
import time
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import BingGroundingTool

# Load environment variables
load_dotenv()

# Get connection info from environment variables
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")  # Get the model name from env

if not MODEL_DEPLOYMENT_NAME:
    print("WARNING: MODEL_DEPLOYMENT_NAME not found in environment variables.")
    print("Using default model name from your Azure OpenAI deployment.")
    MODEL_DEPLOYMENT_NAME = "gpt-4"  # fallback to a common name

def simple_bing_search(query="What is the weather in Seattle?"):
    """
    Most basic, simple Bing search possible.
    Just does one search and displays the results.
    """
    print(f"Performing simple Bing search for: '{query}'")
    print(f"Using model: {MODEL_DEPLOYMENT_NAME}")
    
    # Step 1: Create the client and get Bing connection
    client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=PROJECT_CONNECTION_STRING,
    )
    
    bing_conn = client.connections.get(connection_name=BING_CONNECTION_NAME)
    print(f"Found Bing connection: {bing_conn.name}")
    print(f"Connection ID: {bing_conn.id}")
    print(f"Connection type: {bing_conn.connection_type}")
    
    # Display all available properties of the connection
    print("\nConnection properties:")
    for prop in dir(bing_conn):
        if not prop.startswith('_') and prop not in ['connection_type', 'id', 'name']:
            try:
                value = getattr(bing_conn, prop)
                if not callable(value):
                    print(f"  {prop}: {value}")
            except:
                pass
    
    # Step 2: Create a Bing tool
    bing_tool = BingGroundingTool(connection_id=bing_conn.id)
    print("\nBing tool created")
    
    # Step 3: Create a simple agent with the Bing tool
    print("\nCreating agent...")
    agent = client.agents.create_agent(
        model=MODEL_DEPLOYMENT_NAME,  # Use the model name from environment
        name="simple_bing_test_agent",
        instructions=f"Search for information about: {query}. Show your search process and cite your sources.",
        tools=bing_tool.definitions,
        headers={"x-ms-enable-preview": "true"}
    )
    print(f"Created agent: {agent.id}")
    
    # Step 4: Create a thread and message
    print("\nCreating thread and message...")
    thread = client.agents.create_thread()
    print(f"Thread created: {thread.id}")
    
    message = client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Find information about: {query}. Please show me what sources you're using."
    )
    print(f"Message created: {message.id}")
    
    # Step 5: Create and run the search
    print("\nStarting search...")
    run = client.agents.create_run(thread_id=thread.id, agent_id=agent.id)
    print(f"Run created: {run.id}")
    
    # Step 6: Wait for completion
    print("\nWaiting for completion...")
    while True:
        status = client.agents.get_run(thread_id=thread.id, run_id=run.id)
        print(f"Status: {status.status}")
        
        if status.status == "completed" or status.status == "failed":
            break
            
        time.sleep(2)
    
    # Step 7: Show results if successful or error details if failed
    if status.status == "completed":
        print("\n=== SEARCH RESULTS ===")
        
        # Get the response messages
        messages = client.agents.list_messages(thread_id=thread.id)
        
        # Print assistant messages with their citations
        for msg in messages["data"]:
            if msg['role'] == 'assistant' and msg["content"]:
                for content in msg["content"]:
                    if content.get("text"):
                        # Print the answer
                        print("\nAnswer:")
                        print("-" * 50)
                        print(content['text']['value'])
                        print("-" * 50)
                        
                        # Display sources if any
                        if content['text'].get('annotations'):
                            print("\nSources Used:")
                            for annotation in content['text']['annotations']:
                                if annotation['type'] == 'url_citation':
                                    print(f"• {annotation['url_citation']['title']}")
                                    print(f"  {annotation['url_citation']['url']}")
        
        # Show what was searched for
        print("\nSearch Query:")
        run_steps = client.agents.list_run_steps(thread_id=thread.id, run_id=run.id)
        for step in run_steps["data"]:
            if step.get("step_details") and step["step_details"].get("tool_calls"):
                for tool_call in step["step_details"]["tool_calls"]:
                    if tool_call.get("type") in ["bing_search", "bing_grounding"]:
                        if tool_call.get("type") == "bing_search":
                            print(f"• {tool_call['bing_search'].get('query', 'N/A')}")
                        else:  # bing_grounding
                            url = tool_call['bing_grounding'].get('requesturl', 'N/A')
                            query = url.split('q=')[-1]  # Extract the query part
                            print(f"• {query}")
        
        print("\n=== SEARCH COMPLETE ===")
    else:
        print("\n=== SEARCH FAILED ===")
        print(f"Status: {status.status}")
        if hasattr(status, 'last_error'):
            print(f"Error: {status.last_error}")
    
    # Step 8: Clean up
    try:
        print("\nCleaning up...")
        client.agents.delete_agent(agent.id)
        print("Agent deleted successfully")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        
    print("Test complete.")

if __name__ == "__main__":
    try:
        # What do you want to search for?
        # simple_bing_search("Current Bitcoin price")
        # simple_bing_search("Who is the Head of AI at Y-TEC? You can find it on LinkedIn. Tell me the link to the LinkedIn profile.")
        simple_bing_search("Como salio el partido entre Real Madrid y Arsenal HOY?")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {str(e)}") 