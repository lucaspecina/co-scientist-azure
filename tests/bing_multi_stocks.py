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
# MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
BING_MODEL_DEPLOYMENT_NAME = os.getenv("BING_MODEL_DEPLOYMENT_NAME")
AGENT_MODEL_DEPLOYMENT_NAME = os.getenv("AGENT_MODEL_DEPLOYMENT_NAME")

# Verify that the model deployment exists before proceeding
print(f"Verifying models deployment:")
for model in [BING_MODEL_DEPLOYMENT_NAME, AGENT_MODEL_DEPLOYMENT_NAME]:
    print(f"Verifying model deployment: {model}")
    try:
        # Create a direct OpenAI client to check the model
        client = AzureOpenAI(
            api_key=API_KEY,
            api_version=MODEL_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
    
        # Try a simple completion to verify the model works
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, what's your name?"}],
            max_tokens=30
        )
        print(f"Model verification successful: {model}")
        print(f"Model response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"ERROR: Model deployment verification failed: {str(e)}")
        print("Please check your Azure OpenAI deployment and update your .env file.")
        sys.exit(1)

###############################################################################
#                              Azure OpenAI Client
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
    model=AGENT_MODEL_DEPLOYMENT_NAME,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY,
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
#                               BING QUERY TOOLS
###############################################################################
def stock_price_trends_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on real-time stock prices,
    changes over the last few months for 'stock_name'.
    """
    print(f"[stock_price_trends_tool] Fetching stock price trends for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model=BING_MODEL_DEPLOYMENT_NAME,
        name="stock_price_trends_tool_agent",
        instructions=(
            f"Focus on retrieving real-time stock prices, changes over the last few months, "
            f"and summarize market trends for {stock_name}. "
            "IMPORTANT: You must cite all sources used. For each piece of information, "
            "include the source website and provide direct URLs when available."
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Please get stock price trends data for {stock_name}. Show me your sources."
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)
    
    # Clean up
    project_client.agents.delete_agent(agent.id)

    value = messages["data"][0]["content"][0]["text"]["value"]
    # Extract all citation URLs from annotations
    citations = []
    if "annotations" in messages["data"][0]["content"][0]["text"]:
        for annotation in messages["data"][0]["content"][0]["text"]["annotations"]:
            if "url_citation" in annotation and "url" in annotation["url_citation"]:
                citations.append(annotation["url_citation"]["url"])
    
    # Join all citations or use a default message if none found
    citation = ", ".join(citations) if citations else "No citations available"

    # print(f"[stock_price_trends_tool] Bing result: {value}")
    # print(f"[stock_price_trends_tool] Citation: {citation}")
    
    return value + f"\n\nCitation: {citation}"


def news_analysis_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on the latest news for 'stock_name'.
    """
    print(f"[news_analysis_tool] Fetching news for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model=BING_MODEL_DEPLOYMENT_NAME,
        name="news_analysis_tool_agent",
        instructions=(
            f"Focus on the latest news highlights for the stock {stock_name}. "
            "IMPORTANT: For each news item or piece of information:"
            "\n- Cite the source publication/website"
            "\n- Include the publication date"
            "\n- Provide direct URLs to articles"
            "\n- Mention if it's from a premium/subscription source"
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"Retrieve the latest news articles about {stock_name}. Include all sources and URLs."
    )
    run = project_client.agents.create_and_process_run(thread_id=thread.id, agent_id=agent.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)

    value = messages["data"][0]["content"][0]["text"]["value"]
    citations = []
    if "annotations" in messages["data"][0]["content"][0]["text"]:
        for annotation in messages["data"][0]["content"][0]["text"]["annotations"]:
            if "url_citation" in annotation and "url" in annotation["url_citation"]:
                citations.append(annotation["url_citation"]["url"])
    
    citation = ", ".join(citations) if citations else "No citations available"

    # print(f"[news_analysis_tool] Bing result: {value}")
    # print(f"[news_analysis_tool] Citation: {citation}")
    
    # Clean up
    project_client.agents.delete_agent(agent.id)
    
    return value + f"\n\nCitation: {citation}"


def market_sentiment_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on overall market sentiment
    for 'stock_name'.
    """
    print(f"[market_sentiment_tool] Fetching sentiment for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model=BING_MODEL_DEPLOYMENT_NAME,
        name="market_sentiment_tool_agent",
        instructions=(
            f"Focus on analyzing general market sentiment regarding {stock_name}. "
            "IMPORTANT: For all information provided:"
            "\n- Cite each source with name and URL"
            "\n- Include dates for all sentiment indicators"
            "\n- Note if sentiment is from retail investors, institutions, or analysts"
            "\n- Provide direct links to sentiment analysis or market reports"
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=(
            f"Gather market sentiment, user opinions, and overall feeling about {stock_name}. "
            "Include all sources and URLs."
        )
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
    
    citation = ", ".join(citations) if citations else "No citations available"

    # print(f"[market_sentiment_tool] Bing result: {value}")
    # print(f"[market_sentiment_tool] Citation: {citation}")

    return value + f"\n\nCitation: {citation}"


def analyst_reports_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on analyst reports
    for 'stock_name'.
    """
    print(f"[analyst_reports_tool] Fetching analyst reports for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model=BING_MODEL_DEPLOYMENT_NAME,
        name="analyst_reports_tool_agent",
        instructions=(
            f"Focus on finding recent analyst reports and professional analyses about {stock_name}. "
            "IMPORTANT: For each analyst report or opinion:"
            "\n- Name the analyst and their firm"
            "\n- Include the publication date"
            "\n- Provide direct URLs to sources"
            "\n- Note if it's from a premium/subscription service"
            "\n- Cite any price targets or ratings changes"
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=(
            f"Find recent analyst reports and professional opinions on {stock_name}. "
            "Include all sources, dates, and URLs."
        )
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
    
    citation = ", ".join(citations) if citations else "No citations available"

    # print(f"[analyst_reports_tool] Bing result: {value}")
    # print(f"[analyst_reports_tool] Citation: {citation}")

    return value + f"\n\nCitation: {citation}"


def expert_opinions_tool(stock_name: str) -> str:
    """
    A dedicated Bing call focusing on expert or industry leaders' opinions
    for 'stock_name'.
    """
    print(f"[expert_opinions_tool] Fetching expert opinions for {stock_name}...")
    bing = BingGroundingTool(connection_id=conn_id)
    agent = project_client.agents.create_agent(
        model=BING_MODEL_DEPLOYMENT_NAME,
        name="expert_opinions_tool_agent",
        instructions=(
            f"Focus on finding expert and industry leader opinions about {stock_name}. "
            "IMPORTANT: For each expert opinion:"
            "\n- Include the expert's name and credentials"
            "\n- Specify their role/position"
            "\n- Note the date of their statement"
            "\n- Provide direct URLs to sources"
            "\n- Indicate if it's from an interview, report, or social media"
        ),
        tools=bing.definitions,
        headers={"x-ms-enable-preview": "true"}
    )

    thread = project_client.agents.create_thread()
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=(
            f"Find expert opinions and quotes about {stock_name}. "
            "Include full source attribution with names, dates, and URLs."
        )
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
    
    citation = ", ".join(citations) if citations else "No citations available"

    # print(f"[expert_opinions_tool] Bing result: {value}")
    # print(f"[expert_opinions_tool] Citation: {citation}")

    return value + f"\n\nCitation: {citation}"


###############################################################################
#                              AGENT FUNCTIONS
###############################################################################
#
# These "agent functions" are how each assistant actually calls the above tools.
# The difference is that each AssistantAgent below will have 'tools=[...]'
# pointing to these Python functions. Then the agent can call them
# (directly or via the round-robin workflow).
#
###############################################################################

# -- Trend Data
def stock_price_trends_agent(stock_name: str) -> str:
    """Agent function for 'stock trends', calls stock_price_trends_tool."""
    return stock_price_trends_tool(stock_name)

# -- News
def news_analysis_agent(stock_name: str) -> str:
    """Agent function for 'latest news', calls news_analysis_tool."""
    return news_analysis_tool(stock_name)

# -- Market Sentiment
def market_sentiment_agent(stock_name: str) -> str:
    """Agent function for 'market sentiment', calls market_sentiment_tool."""
    return market_sentiment_tool(stock_name)

# -- Analyst Reports
def analyst_reports_agent(stock_name: str) -> str:
    """Agent function for 'analyst reports', calls analyst_reports_tool."""
    return analyst_reports_tool(stock_name)

# -- Expert Opinions
def expert_opinions_agent(stock_name: str) -> str:
    """Agent function for 'expert opinions', calls expert_opinions_tool."""
    return expert_opinions_tool(stock_name)


###############################################################################
#                         ASSISTANT AGENT DEFINITIONS
###############################################################################
#
# In RoundRobinGroupChat, each of these agents is called in turn. The system_message
# clarifies each agent's role, and the 'tools=[...]' argument lists the Python
# functions that agent can call.
#
###############################################################################
stock_trends_agent_assistant = AssistantAgent(
    name="stock_trends_agent",
    model_client=agent_model_client,
    tools=[stock_price_trends_agent],
    system_message=(
        "You are the Stock Price Trends Agent.\n"
        "\nYOUR TASK:"
        "\n- Fetch and analyze current stock prices"
        "\n- Show price changes over recent months"
        "\n- Highlight significant price movements"
        "\n\nFORMAT YOUR RESPONSE:"
        "\n1. Current Price: [Value] (Source: [URL])"
        "\n2. Recent Changes: [Details] (Source: [URL])"
        "\n3. Key Trends: [Bullet points]"
        "\n4. Sources Used:"
        "\n   • [Source Name] - [URL]"
        "\n   • [Source Name] - [URL]"
        "\n\nDo NOT provide investment advice."
        "\nALWAYS cite your sources with URLs."
    )
)

news_agent_assistant = AssistantAgent(
    name="news_agent",
    model_client=agent_model_client,
    tools=[news_analysis_agent],
    system_message=(
        "You are the News Analysis Agent.\n"
        "\nYOUR TASK:"
        "\n- Find and summarize latest news"
        "\n- Focus on market-moving events"
        "\n- Highlight company developments"
        "\n\nFORMAT YOUR RESPONSE:"
        "\n1. Latest Headlines"
        "\n   • [Headline] (Source: [Publication], Date: [Date], URL: [Link])"
        "\n2. Major Developments"
        "\n   • [Development] (Source: [URL])"
        "\n3. Potential Impact"
        "\n\nSources Used:"
        "\n• [Publication Name] - [URL]"
        "\n• [Publication Name] - [URL]"
        "\n\nKeep summaries concise and factual."
        "\nALWAYS cite sources with URLs and dates."
    )
)

sentiment_agent_assistant = AssistantAgent(
    name="sentiment_agent",
    model_client=agent_model_client,
    tools=[
        market_sentiment_agent,
        analyst_reports_agent,
        expert_opinions_agent
    ],
    system_message=(
        "You are the Market Sentiment Agent.\n"
        "\nYOUR TASK:"
        "\n- Analyze overall market sentiment"
        "\n- Review analyst reports"
        "\n- Gather expert opinions"
        "\n\nFORMAT YOUR RESPONSE:"
        "\n1. Market Sentiment: [Positive/Negative/Mixed]"
        "\n   Sources:"
        "\n   • [Source] - [URL]"
        "\n2. Analyst Consensus: [Summary]"
        "\n   Sources:"
        "\n   • [Analyst/Firm] - [URL]"
        "\n3. Expert Views: [Key points]"
        "\n   Sources:"
        "\n   • [Expert Name/Title] - [URL]"
        "\n\nFocus on facts, not speculation."
        "\nALWAYS cite sources with URLs."
    )
)

decision_agent_assistant = AssistantAgent(
    name="decision_agent",
    model_client=agent_model_client,
    system_message=(
        "You are the Decision Agent.\n"
        "\nYOUR TASK:"
        "\nSynthesize all information and make a final decision."
        "\n\nFORMAT YOUR RESPONSE:"
        "\n=== ANALYSIS SUMMARY ==="
        "\n1. Stock Performance: [Summary]"
        "\n   Key Sources: [URLs]"
        "\n2. News Overview: [Key Points]"
        "\n   Key Sources: [URLs]"
        "\n3. Market Sentiment: [Overview]"
        "\n   Key Sources: [URLs]"
        "\n\n=== FINAL DECISION ==="
        "\n• Decision: [INVEST or DO NOT INVEST]"
        "\n• Current Price: [Latest Price] (Source: [URL])"
        "\n• Key Factors: [Bullet Points]"
        "\n\nAll Sources Used:"
        "\n[List all unique sources with URLs]"
        "\n\nEnd with 'Decision Made'"
    )
)

###############################################################################
#                        TERMINATION & TEAM CONFIGURATION
###############################################################################
# Stop once "Decision Made" is in the response, or if 15 messages have passed
text_termination = TextMentionTermination("Decision Made")
max_message_termination = MaxMessageTermination(15)
termination = text_termination | max_message_termination

# Round-robin chat among the four agents
investment_team = RoundRobinGroupChat(
    [
        stock_trends_agent_assistant,
        news_agent_assistant,
        sentiment_agent_assistant,
        decision_agent_assistant,
    ],
    termination_condition=termination,
)

###############################################################################
#                                   MAIN
###############################################################################
def main():
    stock_name = "Tesla"
    print("\n" + "="*80)
    print(f"Starting Analysis for {stock_name}")
    print("="*80 + "\n")
    
    async def run_analysis():
        await Console(
            investment_team.run_stream(
                task=f"Analyze stock trends, news, and sentiment for {stock_name}, plus analyst reports and expert opinions, and then decide whether to invest."
            )
        )
    
    asyncio.run(run_analysis())

if __name__ == "__main__":
    main()
