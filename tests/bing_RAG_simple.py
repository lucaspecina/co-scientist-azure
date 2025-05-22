from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import BingGroundingTool
from dotenv import load_dotenv
import os
import sys
from openai import AzureOpenAI
import asyncio

load_dotenv()

# Initialize environment variables
API_KEY = os.getenv("api_key")
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
BING_MODEL_DEPLOYMENT_NAME = os.getenv("BING_MODEL_DEPLOYMENT_NAME")
AGENT_MODEL_DEPLOYMENT_NAME = os.getenv("AGENT_MODEL_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

###############################################################################
#                              Azure Search Client
###############################################################################
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import os 

service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["AZURE_SEARCH_API_KEY"]

search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))

###############################################################################
#                              Azure OpenAI Client
###############################################################################
# # Verify that the model deployment exists before proceeding
# print(f"Verifying models deployment:")
# for model in [BING_MODEL_DEPLOYMENT_NAME, AGENT_MODEL_DEPLOYMENT_NAME]:
#     print(f"Verifying model deployment: {model}")
#     try:
#         # Create a direct OpenAI client to check the model
#         client = AzureOpenAI(
#             api_key=API_KEY,
#             api_version=MODEL_API_VERSION,
#             azure_endpoint=AZURE_ENDPOINT
#         )
    
#         # Try a simple completion to verify the model works
#         response = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": "Hello, what's your name?"}],
#             max_tokens=30
#         )
#         print(f"Model verification successful: {model}")
#         print(f"Model response: {response.choices[0].message.content}")
#     except Exception as e:
#         print(f"ERROR: Model deployment verification failed: {str(e)}")
#         print("Please check your Azure OpenAI deployment and update your .env file.")
#         sys.exit(1)

###############################################################################
#                              OpenAI Clients
###############################################################################

bing_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=BING_MODEL_DEPLOYMENT_NAME,
    model=BING_MODEL_DEPLOYMENT_NAME,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
    model_info={
        "context_length": 128000,
        "is_chat_model": True,
        "supports_functions": True,
        "name": BING_MODEL_DEPLOYMENT_NAME,
        "vision": False,
        "function_calling": "auto",
        "json_output": True,
        "family": "gpt-4",
        "structured_output": True
    }
)

agent_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=AGENT_MODEL_DEPLOYMENT_NAME,
    model="gpt-4o-2024-11-20",
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
)

openai_client = AzureOpenAI(
    api_key=API_KEY,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

###############################################################################
#                              AI Project Client
###############################################################################
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=PROJECT_CONNECTION_STRING,
)

# Retrieve the Bing connection
bing_connection = project_client.connections.get(connection_name=BING_CONNECTION_NAME)
conn_id = bing_connection.id

###############################################################################
#                               RAG SEARCH TOOLS
###############################################################################

def get_embedding(text, client):
    """Generate embedding for the given text using Azure OpenAI"""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_DEPLOYMENT,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def vector_search(query: str, client: SearchClient, openai_client: AzureOpenAI, top_k: int = 15):
    """Perform vector search using Azure Cognitive Search"""
    print("\nGenerating embedding for search...")
    
    query_vector = get_embedding(query, openai_client)
    if not query_vector:
        print("Failed to generate embedding")
        return None
    
    try:
        vector_query = {
            "vector_queries": [{
                "vector": query_vector,
                "k": top_k,
                "fields": "contentVector",
                "kind": "vector"
            }],
            "select": ["content", "title", "category", "url", "source", "chunk_id"],
            "top": top_k
        }
        
        print(f"\nSearching with expanded query (top_k={top_k})...")
        results = client.search(
            search_text=None,
            **vector_query
        )
        
        # Debug: Print number of results
        results_list = list(results)
        print(f"Found {len(results_list)} results from vector search")
        
        # Debug: Print first result if available
        if results_list:
            first_result = results_list[0]
            print("\nFirst result preview:")
            print(f"Title: {first_result.get('title', 'No title')}")
            content = first_result.get('content', 'No content')
            print(f"Content preview: {content[:200]}...")
        
        return results_list  # Return the list directly
        
    except Exception as e:
        print(f"Error during vector search: {str(e)}")
        return None

def extract_relevant_content(results):
    """Extract clean, relevant content from search results"""
    if not results:
        print("No results to extract content from")
        return []
    
    relevant_passages = []
    seen_contents = set()
    
    try:
        if not isinstance(results, list):
            results_list = list(results)
        else:
            results_list = results
            
        print(f"\nProcessing {len(results_list)} results for content extraction")
        
        # High-value content indicators
        core_concepts = [
            "rl generalizes", "generalization in rl", "reinforcement learning generalization",
            "model generalization", "out-of-distribution", "zero-shot", "transfer learning",
            "generalization performance", "generalization ability"
        ]
        
        # Sections to skip entirely
        skip_sections = [
            "references", "bibliography", "acknowledgments", "appendix",
            "table of contents", "list of figures", "list of tables"
        ]
        
        # Patterns indicating metadata or non-content
        metadata_patterns = [
            r"arXiv:\d{4}\.\d{5}",  # arXiv IDs
            r"^\s*\d+(\.\d+)*\s*$",  # Standalone numbers
            r"^figure \d+",  # Figure references
            r"^table \d+",   # Table references
            r"^\[.*?\]$",    # Citation brackets
            r"<.*?>",        # HTML tags
            r"http[s]?://\S+"  # URLs
        ]
        
        import re
        
        def is_metadata(line):
            return any(re.search(pattern, line.lower()) for pattern in metadata_patterns)
        
        def extract_paragraph(content):
            """Extract meaningful paragraphs from content."""
            lines = content.split('\n')
            paragraphs = []
            current_para = []
            
            for line in lines:
                line = line.strip()
                if not line:  # Empty line indicates paragraph break
                    if current_para:
                        para = ' '.join(current_para)
                        if len(para.split()) > 10:  # Only keep substantial paragraphs
                            paragraphs.append(para)
                        current_para = []
                elif not is_metadata(line) and not any(section in line.lower() for section in skip_sections):
                    current_para.append(line)
            
            # Don't forget the last paragraph
            if current_para:
                para = ' '.join(current_para)
                if len(para.split()) > 10:
                    paragraphs.append(para)
            
            return paragraphs
        
        for i, result in enumerate(results_list):
            content = result.get('content', '').strip()
            title = result.get('title', '').replace('%20', ' ')
            
            print(f"\nProcessing result {i+1}:")
            print(f"Title: {title}")
            
            # Skip if content is too short
            if not content or len(content) < 50:  # Increased minimum length
                print("Skipped: Content too short")
                continue
            
            # Extract meaningful paragraphs
            paragraphs = extract_paragraph(content)
            
            # Score each paragraph for relevance
            for para in paragraphs:
                # Skip if we've seen this content
                if para in seen_contents:
                    continue
                
                # Check for core concepts (higher weight)
                core_concept_matches = sum(concept.lower() in para.lower() for concept in core_concepts)
                if core_concept_matches > 0:
                    seen_contents.add(para)
                    relevant_passages.append({
                        'title': title,
                        'content': para,
                        'relevance_score': core_concept_matches
                    })
                    print(f"Added paragraph with {core_concept_matches} core concept matches")
        
        # Sort passages by relevance score and take top 5
        relevant_passages.sort(key=lambda x: x['relevance_score'], reverse=True)
        relevant_passages = relevant_passages[:5]
        
        print(f"\nFound {len(relevant_passages)} relevant passages")
            
    except Exception as e:
        print(f"Error extracting content: {str(e)}")
        
    return relevant_passages

def rag_search_tool(query: str) -> str:
    """
    A dedicated RAG search tool that performs vector search and returns relevant content
    """
    print(f"[rag_search_tool] Searching for: {query}")
    
    # Get search results
    results = vector_search(query, search_client, openai_client)
    
    # Extract relevant content
    relevant_passages = extract_relevant_content(results)
    
    if not relevant_passages:
        return "No relevant information found in the research papers."
    
    # Format the response
    response = "Here's what I found in the research papers:\n\n"
    
    for i, passage in enumerate(relevant_passages, 1):
        response += f"Source {i}: {passage['title']}\n"
        response += f"Content: {passage['content']}\n\n"
        
    return response

###############################################################################
#                               BING SEARCH TOOL
###############################################################################

def bing_search_tool(query: str) -> str:
    """
    A dedicated Bing search tool that performs web search and returns relevant content
    """
    print(f"[bing_search_tool] Searching web for: {query}")
    
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model=BING_MODEL_DEPLOYMENT_NAME,
        name="bing_search_agent",
        instructions=(
            f"Search the web for information about: {query}\n"
            "IMPORTANT: You must cite all sources used.\n"
            "For each piece of information, include the source website and provide direct URLs when available.\n"
            "Focus on finding accurate and relevant information from reliable sources."
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Search for information about: {query}. Please provide comprehensive results with sources."
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)
    
    # Clean up
    project_client.agents.delete_agent(agent.id)

    value = messages["data"][0]["content"][0]["text"]["value"]
    citations = []
    if "annotations" in messages["data"][0]["content"][0]["text"]:
        for annotation in messages["data"][0]["content"][0]["text"]["annotations"]:
            if "url_citation" in annotation and "url" in annotation["url_citation"]:
                citations.append(annotation["url_citation"]["url"])
    
    citation_text = "\n\nSources:\n" + "\n".join([f"- {url}" for url in citations]) if citations else "\n\nNo citations available"
    
    return value + citation_text

###############################################################################
#                              AGENT FUNCTIONS
###############################################################################

def rag_search_agent(query: str) -> str:
    """Agent function for RAG search, calls rag_search_tool."""
    return rag_search_tool(query)

def bing_search_agent(query: str = "") -> str:
    """Agent function for Bing search, calls bing_search_tool."""
    if not query:
        return "Error: No query provided"
    return bing_search_tool(query)

###############################################################################
#                         ASSISTANT AGENT DEFINITIONS
###############################################################################

rag_search_assistant = AssistantAgent(
    name="rag_search_agent",
    model_client=agent_model_client,
    tools=[rag_search_agent],
    system_message=(
        "You are the Research Papers Search Agent.\n"
        "\nYOUR TASK:"
        "\n- Search through research papers and academic content"
        "\n- Find relevant information from papers and documents"
        "\n- Extract and summarize key findings"
        "\n\nFORMAT YOUR RESPONSE:"
        "\n1. Key Findings:"
        "\n   • [Finding] (Source: [Paper Title])"
        "\n2. Relevant Excerpts:"
        "\n   • From [Paper Title]: [Quote or summary]"
        "\n3. Research Context:"
        "\n   • [Additional context or connections between findings]"
        "\n\nALWAYS cite the source papers."
        "\nBe precise and academic in your language."
        "\nDo not make claims without evidence from the papers."
        "\nUse ONLY the rag_search_agent tool for searching papers."  # Added explicit tool instruction
        "\nDO NOT try to use web search tools - that's handled by another agent."  # Added restriction
        "\n\nWhen you feel you have found sufficient relevant information, end your message with: '[RESEARCH_COMPLETE]'"
    )
)

bing_search_assistant = AssistantAgent(
    name="bing_search_agent",
    model_client=agent_model_client,
    tools=[bing_search_agent],
    system_message=(
        "You are the Web Search Agent.\n"
        "\nYOUR TASK:"
        "\n- Search the web for current and relevant information"
        "\n- Find reliable sources and recent content"
        "\n- Synthesize information from multiple sources"
        "\n\nFORMAT YOUR RESPONSE:"
        "\n1. Key Information:"
        "\n   • [Information] (Source: [URL])"
        "\n2. Recent Developments:"
        "\n   • [Development] (Source: [URL])"
        "\n3. Additional Context:"
        "\n   • [Context from multiple sources]"
        "\n\nALWAYS cite your sources with URLs."
        "\nFocus on accuracy and reliability."
        "\nMake it clear when information is from different time periods."
        "\nUse ONLY the bing_search_agent tool for web searches."  # Added explicit tool instruction
        "\nDO NOT try to use research paper search tools - that's handled by another agent."  # Added restriction
        "\n\nWhen you feel you have found sufficient relevant information, end your message with: '[WEB_COMPLETE]'"
    )
)

synthesis_assistant = AssistantAgent(
    name="synthesis_agent",
    model_client=agent_model_client,
    system_message=(
        "You are the Synthesis Agent.\n"
        "\nYOUR TASK:"
        "\n- Review and synthesize information from both research papers and web sources"
        "\n- Create a comprehensive, well-structured answer to the original query"
        "\n- Highlight where academic research and current real-world information align or differ"
        "\n\nFORMAT YOUR RESPONSE:"
        "\n=== COMPREHENSIVE ANSWER ==="
        "\n[Provide a clear, direct answer to the original query]"
        "\n\n=== RESEARCH FINDINGS ==="
        "\n[Summarize key findings from academic papers]"
        "\n\n=== CURRENT CONTEXT ==="
        "\n[Summarize relevant real-world information]"
        "\n\n=== SYNTHESIS ==="
        "\n[Explain how the academic research and current information relate]"
        "\n\n=== SOURCES ==="
        "\nAcademic Sources:"
        "\n[List academic sources]"
        "\n\nWeb Sources:"
        "\n[List web sources with URLs]"
        "\n\nDO NOT perform any searches - your role is to synthesize information from the other agents."  # Added restriction
        "\n\nEnd with '[SYNTHESIS_COMPLETE]' when you've provided a comprehensive answer."
    )
)

###############################################################################
#                         TERMINATION & TEAM CONFIGURATION
###############################################################################
# Stop once synthesis is complete or if max messages reached
text_termination = TextMentionTermination("[SYNTHESIS_COMPLETE]")
max_message_termination = MaxMessageTermination(5)
termination = text_termination | max_message_termination

# Initialize the team with all three agents
search_team = RoundRobinGroupChat(
    [
        rag_search_assistant,
        bing_search_assistant,
        synthesis_assistant,
    ],
    termination_condition=termination,
)

###############################################################################
#                                   MAIN
###############################################################################
def main():
    query = "What does it mean that RL generalizes?"  # Example query combining RAG and real-time info
    print("\n" + "="*80)
    print(f"Starting Analysis for query: {query}")
    print("="*80 + "\n")
    
    async def run_analysis():
        await Console(
            search_team.run_stream(
                task=f"Analyze this topic using both research papers and real-time information. The RAG and Bing agents should first gather information. Once both indicate completion, the synthesis agent should provide a comprehensive answer synthesizing all sources: {query}"
            )
        )
    
    asyncio.run(run_analysis())

if __name__ == "__main__":
    main()
