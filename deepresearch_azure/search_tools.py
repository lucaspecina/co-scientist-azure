"""
Search tools for the DeepResearch ReAct agent.
Includes RAG search and Bing search implementations.
"""

import deepresearch_azure.config as config
import logging
from deepresearch_azure.content_utils import extract_relevant_content, format_context_for_react
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import BingGroundingTool
from openai import AzureOpenAI

# Setup logging
logger = logging.getLogger('deepresearch.tools')

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(
    api_key=config.AZURE_API_KEY,
    api_version=config.AZURE_API_VERSION,
    azure_endpoint=config.AZURE_ENDPOINT
)

# Initialize Azure Cognitive Search client
search_client = SearchClient(
    endpoint=config.AZURE_SEARCH_ENDPOINT,
    index_name=config.AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
)

# Initialize Azure AI Project client for Bing
try:
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
        conn_str=config.PROJECT_CONNECTION_STRING,
    )
    bing_connection = project_client.connections.get(connection_name=config.BING_CONNECTION_NAME)
    bing_connection_id = bing_connection.id
    logger.info(f"Bing search connection initialized: {config.BING_CONNECTION_NAME}")
except Exception as e:
    logger.warning(f"Failed to initialize Bing search: {e}")
    bing_connection_id = None

class SearchTool:
    """Base class for search tools"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f'deepresearch.tools.{name}')
    
    def execute(self, query):
        """Execute the search tool with a query"""
        raise NotImplementedError("Subclasses must implement execute()")
    
    def format_result(self, query, result):
        """Format search results for the ReAct agent"""
        if not result:
            self.logger.warning(f"No results found for query: {query}")
            return f"No results found for query: {query}"
        
        self.logger.info(f"Extracting relevant content from search results")
        relevant_passages = extract_relevant_content(result)
        self.logger.info(f"Found {len(relevant_passages)} relevant passages")
        
        # Print more details about the results for user visibility
        return format_context_for_react(query, relevant_passages)

class RAGSearchTool(SearchTool):
    """RAG search tool using Azure Cognitive Search"""
    
    def __init__(self):
        super().__init__(
            name="search_rag",
            description="Search through research papers and documents in the knowledge base"
        )
    
    def get_embedding(self, text):
        """Generate embedding for the given text using Azure OpenAI"""
        try:
            self.logger.info(f"Generating embedding with model: {config.EMBEDDING_DEPLOYMENT}")
            response = openai_client.embeddings.create(
                model=config.EMBEDDING_DEPLOYMENT,
                input=text
            )
            self.logger.info("Embedding generated successfully")
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def execute(self, query, top_k=15):
        """Perform vector search using Azure Cognitive Search"""
        self.logger.info(f"Executing RAG search for: {query}")
        print(f"\n[RAG Search] Searching research papers for: {query}")
        
        # Start with a more detailed search query for research papers
        expanded_query = f"""
        {query}
        Information from research papers on this topic
        Scientific evidence and studies about this
        """
        
        self.logger.info(f"Generating embedding for expanded query")
        query_vector = self.get_embedding(expanded_query)
        if not query_vector:
            self.logger.error("Failed to generate embedding")
            return None
        
        try:
            self.logger.info(f"Executing vector search with top_k={top_k}")
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
            
            results = search_client.search(
                search_text=None,
                **vector_query
            )
            
            # Convert to list and log info
            results_list = list(results)
            self.logger.info(f"Received {len(results_list)} results from RAG search")
            
            # Display info about results to make them visible
            if results_list and len(results_list) > 0:
                print(f"\n[RAG RESULTS] Found {len(results_list)} relevant documents")
                print("-" * 40)
                
                # Display the top 3 results with titles and snippets
                for i, result in enumerate(results_list[:3], 1):
                    title = result.get('title', 'No title').replace('%20', ' ')
                    content = result.get('content', 'No content')
                    
                    # Format and display a snippet
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    clean_snippet = snippet.replace('\n', ' ')
                    print(f"{i}. {title}")
                    print(f"   Snippet: {clean_snippet}")
                    print()
                
                print("-" * 40)
            
            return results_list
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            return None

class BingSearchTool(SearchTool):
    """Bing search tool using Azure AI Projects"""
    
    def __init__(self):
        super().__init__(
            name="search_web",
            description="Search the web for information using Bing"
        )
    
    def execute(self, query):
        """Perform web search using Bing"""
        self.logger.info(f"Executing Bing search for: {query}")
        print(f"\n[Bing Search] Searching web for: {query}")
        
        if not bing_connection_id:
            self.logger.error("Bing search is not available. Check your configuration.")
            return ["Bing search is not available. Check your configuration."]
        
        try:
            bing = BingGroundingTool(connection_id=bing_connection_id)
            
            self.logger.info(f"Creating Bing search agent with model: {config.BING_MODEL_DEPLOYMENT}")
            agent = project_client.agents.create_agent(
                model=config.BING_MODEL_DEPLOYMENT,
                name="bing_search_agent",
                instructions=f"Search the web for information about: {query}. Provide relevant results with sources.",
                tools=bing.definitions,
                headers={"x-ms-enable-preview": "true"}
            )

            self.logger.info("Creating thread and message for Bing search")
            thread = project_client.agents.create_thread()
            message = project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=f"Search for information about: {query}. Please provide comprehensive results with sources."
            )
            
            self.logger.info("Processing Bing search request")
            run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
            messages = project_client.agents.list_messages(thread_id=thread.id)
            
            # Clean up
            self.logger.info("Cleaning up Bing search agent")
            project_client.agents.delete_agent(agent.id)

            response_message = messages["data"][0]["content"][0]["text"]["value"] if messages["data"] else "No results found"
            
            # Extract citations if available
            citations = []
            if "annotations" in messages["data"][0]["content"][0]["text"]:
                for annotation in messages["data"][0]["content"][0]["text"]["annotations"]:
                    if "url_citation" in annotation and "url" in annotation["url_citation"]:
                        url = annotation["url_citation"]["url"]
                        citations.append(url)
                        self.logger.info(f"Found citation: {url}")
            
            # Display the Bing search results more prominently
            print("\n[BING RESULTS] Web information found:")
            print("-" * 40)
            
            # Print a snippet of the response (first 500 chars) to show what was found
            response_snippet = response_message[:500] + "..." if len(response_message) > 500 else response_message
            print(response_snippet)
            print()
            
            # Show the sources
            if citations:
                print("Sources:")
                for i, url in enumerate(citations[:3], 1):  # Show the first 3 sources
                    print(f"{i}. {url}")
                
                if len(citations) > 3:
                    print(f"... and {len(citations) - 3} more sources")
            
            print("-" * 40)
            
            # Add citations to response
            if citations:
                response_message += "\n\nSources:\n" + "\n".join([f"- {url}" for url in citations])
                print(f"[Bing] Found information from {len(citations)} web sources")
            
            self.logger.info(f"Bing search completed with {len(citations)} citations")
            return [{"title": "Bing Search Results", "content": response_message}]
            
        except Exception as e:
            self.logger.error(f"Error during Bing search: {str(e)}")
            return None

class AskUserTool(SearchTool):
    """Tool to ask the user for feedback or clarification"""
    def __init__(self):
        super().__init__(
            name="ask_user",
            description="Ask the user for feedback or clarification"
        )

    def execute(self, query):
        # Prompt the user and return their input
        print(f"\n[ASK USER] {query}")
        answer = input("> ")
        return answer

    def format_result(self, query, result):
        # Return the raw user input as the formatted observation
        return result

# Available tools
RAG_TOOL = RAGSearchTool()
BING_TOOL = BingSearchTool()
ASK_USER_TOOL = AskUserTool()

# Import and instantiate ArxivSearchTool
from deepresearch_azure.arxiv_tool import ArxivSearchTool
from deepresearch_azure.paper_reader import PaperReaderTool

ARXIV_TOOL = ArxivSearchTool()
PAPER_READER_TOOL = PaperReaderTool()

def get_all_tools():
    """Return all available search tools"""
    logger.info("Getting all search tools")
    return [RAG_TOOL, BING_TOOL, ASK_USER_TOOL, ARXIV_TOOL, PAPER_READER_TOOL]  # Add PAPER_READER_TOOL to the list 