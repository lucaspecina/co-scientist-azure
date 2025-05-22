"""
Utilities for processing content from search results.
"""

def extract_relevant_content(results, max_passages=5):
    """Extract clean, relevant content from search results"""
    if not results:
        return []
    
    relevant_passages = []
    seen_contents = set()
    
    try:
        if not isinstance(results, list):
            results_list = list(results)
        else:
            results_list = results
        
        for result in results_list:
            # Handle RAG search results format
            if isinstance(result, dict):
                content = result.get('content', '').strip()
                title = result.get('title', '').replace('%20', ' ')
            else:
                # Handle Bing search results format
                content = getattr(result, 'content', '').strip()
                title = getattr(result, 'title', '').replace('%20', ' ')
                
            # Skip if content is too short or empty
            if not content or len(content) < 50:
                continue
                
            # Clean up the content (basic version)
            content = '\n'.join(line for line in content.split('\n') 
                              if line.strip() and not line.strip().startswith(('<!--', 'PageNumber', 'PageBreak', 'PageHeader')))
            
            content = content.strip()
            if not content or content in seen_contents:
                continue
                
            seen_contents.add(content)
            relevant_passages.append({
                'title': title,
                'content': content
            })
            
            # Limit number of passages
            if len(relevant_passages) >= max_passages:
                break
            
    except Exception as e:
        print(f"Error extracting content: {str(e)}")
        
    return relevant_passages

def format_context_for_react(query, relevant_passages):
    """Format search results into a context for the ReAct agent"""
    if not relevant_passages:
        return "No relevant information found."
        
    context = f"Search results for query: {query}\n\n"
    
    for i, passage in enumerate(relevant_passages, 1):
        context += f"Source {i}: {passage['title']}\n"
        context += f"Content: {passage['content'][:1000]}\n\n"  # Limit content length
        
    return context 