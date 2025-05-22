# DeepResearch ReAct

A research assistant powered by Azure OpenAI, Azure Cognitive Search, and the ReAct framework.

## Overview

DeepResearch ReAct combines two powerful search capabilities:
1. RAG (Retrieval-Augmented Generation) for searching through a vector database of research papers
2. Bing web search for accessing real-time information from the web

The system uses the ReAct (Reasoning + Acting) framework to make the agent's reasoning process transparent and allow it to choose the most appropriate search method for each query.

## Features

- **ReAct Framework**: The agent uses a reasoning-action-observation cycle that makes its thought process transparent
- **RAG Search**: Searches through a vector database of research papers using Azure Cognitive Search
- **Bing Web Search**: Accesses real-time information from the web using Azure AI
- **Dynamic Tool Selection**: The agent can choose which search method to use based on the query
- **Flexible Architecture**: Easy to extend with additional tools

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
python main.py --query "What does it mean that RL generalizes?"
```

Or import and use the agent in your code:

```python
from react_agent import ReActAgent

agent = ReActAgent()
result = agent.run("What does it mean that RL generalizes?")
print(result)
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

## Architecture

- `main.py`: Entry point for the application
- `react_agent.py`: Implementation of the ReAct agent
- `search_tools.py`: Search tool implementations (RAG and Bing)
- `content_utils.py`: Utilities for processing search results
- `config.py`: Configuration and environment variables
- `prompts.py`: Prompt templates for the ReAct agent

## Extending

To add new tools to the agent:

1. Create a new tool class in `search_tools.py` that inherits from `SearchTool`
2. Implement the `execute` and `format_result` methods
3. Add the tool to the `get_all_tools` function

## Tests:

- test_bing_search_simple.py: lo mas basico para buscar en bing con Azure
- test_bing_multi_stocks.py: implementacion multi-agente que buscan todos en bing y despues hay una decision final
- test_rag_simple_search.py: buscador de un RAG de Azure Search con logica simple semantica
- test_rag_vector_search.py: buscador de un RAG de Azure Search con busqueda de vectores
- test_bing_RAG_simple.py: implementacion multi-agente donde un agente busca en bing, otro en RAG y despues hay otro agente que sumariza todo