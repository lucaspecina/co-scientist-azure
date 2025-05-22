import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI settings
AZURE_API_KEY = os.getenv("api_key")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_VERSION = os.getenv("MODEL_API_VERSION")
AGENT_MODEL_DEPLOYMENT = os.getenv("AGENT_MODEL_DEPLOYMENT_NAME")
BING_MODEL_DEPLOYMENT = os.getenv("BING_MODEL_DEPLOYMENT_NAME")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

# Azure Cognitive Search settings
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")

# Azure AI Project settings for Bing
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")

# ReAct agent settings
MAX_ITERATIONS = 50  # Maximum number of reasoning iterations
MAX_TOKENS = 5000   # Maximum tokens in response
TEMPERATURE = 0     # Temperature for LLM (0 = deterministic) 