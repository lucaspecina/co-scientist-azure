"""
Semantic Scholar search tool for the DeepResearch ReAct agent using direct API calls.
"""

import requests
import logging
from deepresearch_azure.content_utils import extract_relevant_content, format_context_for_react
from deepresearch_azure.search_tools import SearchTool
import deepresearch_azure.config as config

# Setup logging
logger = logging.getLogger('deepresearch.tools.semantic_scholar')

class SemanticScholarSearchTool(SearchTool):
    """Semantic Scholar search tool using direct API calls"""
    
    def __init__(self):
        super().__init__(
            name="search_semantic_scholar",
            description="Search for academic papers and research using Semantic Scholar API"
        )
        self.api_url = config.SEMANTIC_SCHOLAR_API_URL
        self.api_key = config.SEMANTIC_SCHOLAR_API_KEY
    
    def execute(self, query, limit=10):
        """Perform academic paper search using Semantic Scholar API"""
        self.logger.info(f"Executing Semantic Scholar search for: {query}")
        print(f"\n[Semantic Scholar] Searching academic papers for: {query}")
        
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            # Construct the search query
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,url,year,authors,venue,citationCount,fieldsOfStudy"
            }
            
            response = requests.get(
                f"{self.api_url}/paper/search",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            
            results = response.json()
            papers = results.get("data", [])
            
            if not papers:
                self.logger.warning("No papers found in Semantic Scholar search")
                return None
            
            # Format the results
            formatted_results = []
            for paper in papers:
                # Extract author names
                authors = [author.get("name", "") for author in paper.get("authors", [])]
                authors_str = ", ".join(authors) if authors else "Unknown authors"
                
                # Create a formatted result
                formatted_paper = {
                    "title": paper.get("title", "No title"),
                    "authors": authors,
                    "year": paper.get("year", "Unknown"),
                    "venue": paper.get("venue", "Unknown venue"),
                    "citations": paper.get("citationCount", 0),
                    "fields": paper.get("fieldsOfStudy", ["Unknown"]),
                    "abstract": paper.get("abstract", "No abstract available"),
                    "url": paper.get("url", "No URL available")
                }
                formatted_results.append(formatted_paper)
            
            # Display info about results
            print(f"\n[SEMANTIC SCHOLAR RESULTS] Found {len(formatted_results)} relevant papers")
            print("-" * 40)
            
            # Display the top 3 results
            for i, paper in enumerate(formatted_results[:3], 1):
                print(f"{i}. {paper['title']}")
                print(f"   Authors: {', '.join(paper['authors'])}")
                print(f"   Year: {paper['year']}")
                print(f"   Venue: {paper['venue']}")
                print(f"   Citations: {paper['citations']}")
                print(f"   Fields: {', '.join(paper['fields'])}")
                print(f"   Abstract: {paper['abstract'][:200]}...")
                print()
            
            print("-" * 40)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error during Semantic Scholar search: {str(e)}")
            return None

    def format_result(self, query, result):
        """Format Semantic Scholar search results for the ReAct agent"""
        if not result:
            self.logger.warning(f"No Semantic Scholar results found for query: {query}")
            return f"No Semantic Scholar results found for query: {query}"
        
        formatted_string = f"Semantic Scholar search results for query: '{query}'\n\n"
        
        for i, paper in enumerate(result, 1):
            formatted_string += f"Paper {i}:\n"
            formatted_string += f"  Title: {paper['title']}\n"
            formatted_string += f"  Authors: {', '.join(paper['authors'])}\n"
            formatted_string += f"  Year: {paper['year']}\n"
            formatted_string += f"  Venue: {paper['venue']}\n"
            formatted_string += f"  Citations: {paper['citations']}\n"
            formatted_string += f"  Fields: {', '.join(paper['fields'])}\n"
            # Limit abstract length for brevity
            abstract_snippet = paper['abstract'][:500] + "..." if len(paper['abstract']) > 500 else paper['abstract']
            formatted_string += f"  Abstract: {abstract_snippet}\n"
            formatted_string += f"  URL: {paper['url']}\n\n"
        
        self.logger.info(f"Formatted {len(result)} Semantic Scholar papers into a structured string for context.")
        return formatted_string.strip()

def get_semantic_scholar_tool():
    """Return the Semantic Scholar search tool"""
    logger.info("Getting Semantic Scholar search tool")
    return SemanticScholarSearchTool() 