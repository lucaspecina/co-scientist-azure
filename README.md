A research assistant powered by Azure OpenAI, Azure Cognitive Search, and the ReAct framework.

## Requirements

- Python 3.8+
- Azure OpenAI access
- Azure Cognitive Search with vector index
- Azure AI Project with Bing connection

## Setup

1. Clone the repository
2. Create a `.env` file with your Azure credentials (see `.env.example`)
3. Install dependencies:
   ```
   pip install openai azure-search-documents azure-ai-projects python-dotenv
   ```

## Usage

Run the main script with a query:

```
python main.py
```

## Environment Variables

Required environment variables in your `.env` file:

```
# Azure OpenAI
api_key=your_azure_openai_key
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
MODEL_API_VERSION=2023-05-15
AGENT_MODEL_DEPLOYMENT_NAME=your_gpt_deployment
BING_MODEL_DEPLOYMENT_NAME=your_bing_deployment
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Azure Cognitive Search
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_INDEX_NAME=your_index_name
AZURE_SEARCH_API_KEY=your_search_api_key

# Azure AI Project for Bing
PROJECT_CONNECTION_STRING=your_project_connection_string
BING_CONNECTION_NAME=your_bing_connection_name
```