"""
Arxiv search tool for the DeepResearch ReAct agent using direct API calls.
"""

import requests
import xml.etree.ElementTree as ET
import logging
from deepresearch_azure.content_utils import extract_relevant_content, format_context_for_react
from deepresearch_azure.search_tools import SearchTool

# Setup logging
logger = logging.getLogger('deepresearch.tools.arxiv')

class ArxivSearchTool(SearchTool):
    """Arxiv search tool using direct API calls"""
    
    def __init__(self):
        super().__init__(
            name="search_arxiv",
            description="Search for research papers on Arxiv.org via API"
        )
        self.arxiv_api_url = "http://export.arxiv.org/api/query"

    def execute(self, query, max_results=5):
        """Perform search on Arxiv using its public API"""
        self.logger.info(f"Executing Arxiv API search for: {query}")
        print(f"\n[Arxiv API Search] Searching Arxiv for: {query}")
        
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance'
        }
        
        try:
            response = requests.get(self.arxiv_api_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Parse the XML response
            root = ET.fromstring(response.content)
            
            # XML namespace, often present in Arxiv API responses
            # The actual namespace might vary, common one is Atom
            ns = {'atom': 'http://www.w3.org/2005/Atom'} 
            
            results = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else "No title"
                
                authors = []
                for author_element in entry.findall('atom:author', ns):
                    name_element = author_element.find('atom:name', ns)
                    if name_element is not None:
                        authors.append(name_element.text.strip())
                
                summary = entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else "No summary"
                
                # Find PDF link (usually the one with title 'pdf')
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf' and link.get('href'):
                        pdf_url = link.get('href')
                        break
                # Fallback if no link with title='pdf' is found, take the first one that ends with .pdf
                if not pdf_url:
                    for link in entry.findall('atom:link', ns):
                        href = link.get('href')
                        if href and href.endswith('.pdf'):
                           pdf_url = href
                           break

                published_date_element = entry.find('atom:published', ns)
                published_date = published_date_element.text.split('T')[0] if published_date_element is not None else "No date"

                results.append({
                    'title': title,
                    'authors': authors,
                    'summary': summary,
                    'pdf_url': pdf_url,
                    'published': published_date
                })
            
            self.logger.info(f"Received {len(results)} results from Arxiv API search")
            
            if results:
                print(f"\n[ARXIV API RESULTS] Found {len(results)} relevant papers")
                print("-" * 40)
                for i, paper in enumerate(results[:3], 1): # Display top 3
                    print(f"{i}. {paper['title']}")
                    print(f"   Authors: {', '.join(paper['authors']) if paper['authors'] else 'N/A'}")
                    print(f"   Published: {paper['published']}")
                    print(f"   Summary: {paper['summary'][:150]}...")
                    print(f"   PDF: {paper['pdf_url'] if paper['pdf_url'] else 'N/A'}")
                    print()
                if len(results) > 3:
                    print(f"... and {len(results) - 3} more papers.")
                print("-" * 40)
            else:
                print("[ARXIV API RESULTS] No papers found.")
                
            return results
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error during Arxiv API request: {str(e)}")
            print(f"Error during Arxiv API request: {str(e)}")
            return None
        except ET.ParseError as e:
            self.logger.error(f"Error parsing Arxiv API XML response: {str(e)}")
            print(f"Error parsing Arxiv API XML response: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during Arxiv API search: {str(e)}")
            print(f"Unexpected error during Arxiv API search: {str(e)}")
            return None

    def format_result(self, query, result):
        """Format Arxiv search results for the ReAct agent with more structure."""
        if not result: # This covers None or empty list
            self.logger.warning(f"No Arxiv results found for query: {query}")
            return f"No Arxiv results found for query: {query}."

        # Construct a string that the LLM can parse more easily.
        # The LLM is good at understanding structured text in various formats.
        # We will format it clearly so it knows what each part is.
        
        formatted_string = f"Arxiv search results for query: '{query}'\n\n"
        
        for i, paper in enumerate(result, 1):
            formatted_string += f"Paper {i}:\n"
            formatted_string += f"  Title: {paper.get('title', 'N/A')}\n"
            authors_str = ', '.join(paper.get('authors', [])) if paper.get('authors') else 'N/A'
            formatted_string += f"  Authors: {authors_str}\n"
            formatted_string += f"  Published: {paper.get('published', 'N/A')}\n"
            # Limit summary length for brevity in the observation context
            summary_snippet = paper.get('summary', 'N/A')[:500] + "..." if len(paper.get('summary', 'N/A')) > 500 else paper.get('summary', 'N/A')
            formatted_string += f"  Summary: {summary_snippet}\n"
            formatted_string += f"  PDF URL: {paper.get('pdf_url', 'N/A')}\n\n"
            
        self.logger.info(f"Formatted {len(result)} Arxiv papers into a structured string for context.")
        return formatted_string.strip()

# Keep this function if it's used elsewhere, or remove if ArxivSearchTool is instantiated directly.
# For consistency with how other tools might be fetched, it's good to keep.
def get_arxiv_tool():
    """Return the Arxiv search tool"""
    logger.info("Getting Arxiv search tool (API version)")
    return ArxivSearchTool() 