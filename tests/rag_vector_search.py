from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import json

load_dotenv()

# Azure Search settings
service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
search_key = os.environ["AZURE_SEARCH_API_KEY"]

# Azure OpenAI settings
openai_key = os.getenv("api_key")
openai_endpoint = os.getenv("AZURE_ENDPOINT")
openai_version = os.getenv("MODEL_API_VERSION")
embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
gpt_deployment = os.getenv("AGENT_MODEL_DEPLOYMENT_NAME")  # Get GPT deployment from environment

if not gpt_deployment:
    raise ValueError("AZURE_GPT_DEPLOYMENT environment variable is not set. Please set it to your GPT model deployment name.")

def get_embedding(text, client):
    """Generate embedding for the given text using Azure OpenAI"""
    try:
        response = client.embeddings.create(
            model=embedding_deployment,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def vector_search(query: str, client: SearchClient, openai_client: AzureOpenAI, top_k: int = 15):
    """Perform vector search using Azure Cognitive Search"""
    print("\nGenerating embedding for search...")
    
    # Expand the query to be more specific about RL generalization
    expanded_query = f"""
    {query}
    What is the difference between how RL and SFT generalize?
    How does reinforcement learning generalize compared to supervised fine-tuning?
    What are the key findings about RL generalization and memorization?
    """
    
    query_vector = get_embedding(expanded_query, openai_client)
    if not query_vector:
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
        return results
    except Exception as e:
        print(f"Error during vector search: {str(e)}")
        return None

def extract_relevant_content(results):
    """Extract clean, relevant content from search results"""
    if not results:
        return []
    
    relevant_passages = []
    seen_contents = set()
    
    try:
        results_list = list(results)
        
        # Keywords to identify relevant content
        relevant_keywords = [
            "rl generalizes", "generalization", "memorization",
            "supervised fine-tuning", "sft", "reinforcement learning",
            "generalize", "memorize", "performance on unseen"
        ]
        
        # Keywords to filter out irrelevant content
        irrelevant_keywords = [
            "time & hardware", "f1 = 2", "success rate", "memory mechanism",
            "long-context", "conversation", "game", "code generation"
        ]
        
        for result in results_list:
            content = result.get('content', '').strip()
            title = result.get('title', '').replace('%20', ' ')
            
            # Skip if content is too short or empty
            if not content or len(content) < 50:
                continue
                
            # Skip if content is mostly irrelevant
            if any(keyword in content.lower() for keyword in irrelevant_keywords):
                continue
                
            # Check if content is relevant
            is_relevant = any(keyword in content.lower() for keyword in relevant_keywords)
            if not is_relevant:
                continue
                
            # Clean up the content
            content = '\n'.join(line for line in content.split('\n') 
                              if not line.strip().startswith(('<!--', 'PageNumber', 'PageBreak', 'PageHeader')))
            
            content = content.strip()
            if not content or content in seen_contents:
                continue
                
            seen_contents.add(content)
            relevant_passages.append({
                'title': title,
                'content': content
            })
            
    except Exception as e:
        print(f"Error extracting content: {str(e)}")
        
    return relevant_passages

def generate_answer(query: str, relevant_passages: list, openai_client: AzureOpenAI):
    """Generate a comprehensive answer using Azure OpenAI"""
    if not relevant_passages:
        return "No relevant information found to answer the question."
        
    context = "\n\n".join([
        f"From '{passage['title']}':\n{passage['content'][:1000]}"
        for passage in relevant_passages
    ])
    
    try:
        prompt = f"""Based ONLY on the following research paper excerpts, provide an answer to this question: "{query}"

Context from research papers:
{context}

IMPORTANT INSTRUCTIONS:
1. ONLY use information that is explicitly stated in the provided excerpts
2. DO NOT add any information from your general knowledge
3. If the provided excerpts don't contain enough information to fully answer the question, explicitly state what information is missing
4. Quote relevant parts of the text to support your answer
5. If you find the information insufficient, say so

Your answer:"""

        response = openai_client.chat.completions.create(
            model=gpt_deployment,
            messages=[
                {"role": "system", "content": "You are a research assistant that ONLY uses the provided information to answer questions. Never add information from your own knowledge. If the provided information is insufficient, say so explicitly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"Error: {str(e)}"

def main():
    try:
        # Initialize clients
        search_client = SearchClient(
            endpoint=service_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_key)
        )
        
        openai_client = AzureOpenAI(
            api_key=openai_key,
            api_version=openai_version,
            azure_endpoint=openai_endpoint
        )
        
        # Single query to process
        query = "What does it mean that RL generalizes?"
        print(f"\nQuery: {query}")
        
        # Get search results
        results = vector_search(query, search_client, openai_client)
        
        # Extract relevant content
        relevant_passages = extract_relevant_content(results)
        
        if not relevant_passages:
            print("\nNo relevant content found in the index.")
            return
            
        print(f"\nFound {len(relevant_passages)} relevant passages in the index:")
        print("=" * 80)
        
        # Print index information
        try:
            index_info = search_client.get_index_statistics()
            print(f"\nIndex Statistics:")
            print(f"Document Count: {index_info.document_count}")
            print(f"Storage Size: {index_info.storage_size} bytes")
            print("=" * 80)
        except Exception as e:
            print(f"Could not retrieve index statistics: {str(e)}")
        
        for i, passage in enumerate(relevant_passages, 1):
            print(f"\nPassage {i}:")
            print(f"Title: {passage['title']}")
            if 'chunk_id' in passage:
                print(f"Chunk ID: {passage['chunk_id']}")
            print("-" * 40)
            print(passage['content'])
            print("-" * 80)
        
        # Generate and display answer
        print("\nGenerated Answer (based ONLY on the above content):")
        print("=" * 80)
        answer = generate_answer(query, relevant_passages, openai_client)
        print(answer)
        print("=" * 80)
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()