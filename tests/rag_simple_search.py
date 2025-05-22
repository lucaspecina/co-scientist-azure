# Import libraries
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.search.documents.models import VectorizableTextQuery
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
key = os.environ["AZURE_SEARCH_API_KEY"]

search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
search_client.get_document_count()

results = search_client.search(search_text="RL generalizes")


# Initialize environment variables
API_KEY = os.getenv("api_key")
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
# MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
BING_MODEL_DEPLOYMENT_NAME = os.getenv("BING_MODEL_DEPLOYMENT_NAME")
AGENT_MODEL_DEPLOYMENT_NAME = os.getenv("AGENT_MODEL_DEPLOYMENT_NAME")


agent_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=AGENT_MODEL_DEPLOYMENT_NAME,
    model=AGENT_MODEL_DEPLOYMENT_NAME,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
)


# Provide instructions to the model
GROUNDED_PROMPT="""
You are an AI assistant that helps users learn from the information found in the source material.
Answer the query using only the sources provided below.
Use bullets if the answer has multiple points.
If the answer is longer than 3 sentences, provide a summary.
Answer ONLY with the facts listed in the list of sources below. Cite your source when you answer the question
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.
Query: {query}
Sources:\n{sources}

"""

def check_index_schema():
    """Check the schema of the search index to identify vector fields"""
    try:
        # Get index client
        from azure.search.documents.indexes import SearchIndexClient
        index_client = SearchIndexClient(service_endpoint, AzureKeyCredential(key))
        
        # Get the index
        index = index_client.get_index(index_name)
        
        # print("\nIndex fields:")
        # for field in index.fields:
        #     print(f"\nField name: {field.name}")
        #     print(f"Type: {field.type}")
        #     if hasattr(field, 'vector_search_dimensions'):
        #         print(f"Vector dimensions: {field.vector_search_dimensions}")
        #     if hasattr(field, 'vector_search_configuration'):
        #         print(f"Vector config: {field.vector_search_configuration}")
            
        return index.fields
    except Exception as e:
        print(f"Error checking index schema: {str(e)}")
        return None

def main():
    # First check the index schema
    print("Checking index schema...")
    fields = check_index_schema()
    
    # Use basic search which is working well
    query = "What does it mean that RL generalizes?"
    print(f"\nSearching for: {query}")
    search_results = search_client.search(
        search_text=query,
        select=["content", "title", "category", "url", "source"],
        top=5
    )

    # Display the search results in a more organized way
    print("\nSearch Results:")
    print("=" * 80)
    
    seen_contents = set()  # To avoid duplicate content
    for result in search_results:
        content = result.get('content', '').strip()
        # Skip if we've seen this content or if it's just noise
        if content in seen_contents or not content or content.startswith("Model output") or content.startswith("[ACTION]"):
            continue
        seen_contents.add(content)
        
        print("\nDocument:")
        print("-" * 40)
        if result.get('title'):
            print(f"Title: {result['title'].replace('%20', ' ')}")
        if result.get('url'):
            print(f"Source: {result['url']}")
        print("\nRelevant Content:")
        print(content[:500] + "..." if len(content) > 500 else content)
        print("-" * 80)

if __name__ == "__main__":
    main()
