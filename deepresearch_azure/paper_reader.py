"""
The PaperReaderTool is a specialized tool for downloading, processing, and analyzing academic papers. Here's how it works:

1. Initialization:
   - Creates a papers directory to store downloaded PDFs
   - Sets up Azure OpenAI client for text summarization
   - Inherits from SearchTool base class

2. PDF Download Process:
   - Handles Arxiv URLs by converting them to PDF format
   - Downloads PDFs with timeout protection
   - Verifies content type is actually PDF
   - Saves files with unique names based on URL

3. PDF Processing:
   - Uses PyMuPDF (fitz) to extract text from PDFs
   - Handles multi-page documents
   - Extracts text while preserving structure
   - Cleans and formats the extracted text

4. Content Analysis:
   - Uses Azure OpenAI to generate summaries
   - Identifies key sections (abstract, methodology, results)
   - Extracts main findings and conclusions
   - Provides structured analysis of the paper
"""


import os
import logging
import requests
from pathlib import Path
import fitz  # PyMuPDF
from openai import AzureOpenAI
import deepresearch_azure.config as config
from deepresearch_azure.search_tools import SearchTool

# Setup logging
logger = logging.getLogger('deepresearch.tools.paper_reader')

class PaperReaderTool(SearchTool):
    """Tool for downloading and reading PDF papers"""
    
    def __init__(self):
        super().__init__(
            name="read_paper",
            description="Download and read a paper from its PDF URL, then summarize its content focusing on the specific research question"
        )
        # Initialize OpenAI client for summarization
        self.client = AzureOpenAI(
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_ENDPOINT
        )
        self.model = config.AGENT_MODEL_DEPLOYMENT
        
        # Create papers directory if it doesn't exist
        self.papers_dir = Path(__file__).parent.parent / 'papers'
        self.papers_dir.mkdir(exist_ok=True)
        
    def download_pdf(self, url):
        """Download PDF from URL to the papers directory"""
        try:
            # Handle arxiv URLs by converting to PDF URL if needed
            if 'arxiv.org' in url and not url.endswith('.pdf'):
                if '/pdf/' not in url:
                    url = url.replace('/abs/', '/pdf/')
                if not url.endswith('.pdf'):
                    url = url + '.pdf'
            
            self.logger.info(f"Downloading PDF from URL: {url}")
            response = requests.get(url, timeout=30)  # Add timeout
            response.raise_for_status()
            
            # Verify content type is PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.endswith('.pdf'):
                self.logger.error(f"URL does not point to a PDF file: {content_type}")
                return None
            
            # Generate a filename from the URL
            filename = url.split('/')[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            # Save to papers directory
            pdf_path = self.papers_dir / filename
            pdf_path.write_bytes(response.content)
            
            return str(pdf_path)
        except requests.exceptions.Timeout:
            self.logger.error("Download timed out")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading PDF: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during download: {str(e)}")
            return None

    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            # Extract text from each page
            for page_num, page in enumerate(doc):
                self.logger.info(f"Processing page {page_num + 1}")
                text += page.get_text()
                
                # Add page separator for better structure
                text += "\n--- Page Break ---\n"
            
            doc.close()
            return text
        except fitz.FileDataError:
            self.logger.error("Invalid or corrupted PDF file")
            return None
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def summarize_paper(self, text, research_question):
        """Use Azure OpenAI to generate a focused summary of the paper based on the research question"""
        try:
            # Split text into chunks if too long
            max_chunk_size = 10000  # Adjust based on model's context window
            chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            summaries = []
            for i, chunk in enumerate(chunks):
                prompt = f"""Please analyze this {'part of the ' if len(chunks) > 1 else ''}research paper specifically in relation to the following research question:

{research_question}

Focus on:
1. Information directly relevant to answering the research question
2. Specific methodologies, results, or findings that address the question
3. Any limitations or caveats that affect the application of these findings
4. Practical implications or recommendations related to the question

Here's the paper text{f' (part {i+1}/{len(chunks)})' if len(chunks) > 1 else ''}:
{chunk}

Please provide a focused summary that specifically addresses the research question."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant that creates focused, relevant summaries of academic papers. Concentrate on information that directly addresses the given research question."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                
                summaries.append(response.choices[0].message.content)
            
            # If we have multiple summaries, combine them with focus on the research question
            if len(summaries) > 1:
                combined_prompt = f"""Combine these {len(summaries)} summaries into a single coherent analysis that addresses the research question:

{research_question}

Summaries to combine:
{chr(10).join(f'Part {i+1}:{chr(10)}{summary}' for i, summary in enumerate(summaries))}

Create a unified analysis that specifically answers the research question, highlighting the most relevant findings and implications."""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a research assistant that creates focused, relevant summaries of academic papers. Concentrate on information that directly addresses the given research question."},
                        {"role": "user", "content": combined_prompt}
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            else:
                return summaries[0]
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return None

    def execute(self, url, research_question):
        """Download, read and summarize a paper from its URL, focusing on the research question"""
        self.logger.info(f"Processing paper from URL: {url}")
        print(f"\n[PAPER READER] Downloading and processing paper from: {url}")
        print(f"Research Question: {research_question}")
        
        # Download the PDF
        pdf_path = self.download_pdf(url)
        if not pdf_path:
            return "Failed to download the paper. Please verify the URL is accessible and points to a PDF file."
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return "Failed to extract text from the paper. The PDF might be corrupted or protected."
        
        # Generate focused summary
        summary = self.summarize_paper(text, research_question)
        if not summary:
            return "Failed to generate paper summary. Please try again later."
        
        return summary

    def format_result(self, url, result):
        """Format the paper summary for the ReAct agent"""
        if not result or result.startswith("Failed to"):
            self.logger.warning(f"Failed to process paper from URL: {url}")
            return f"Failed to process paper from URL: {url}"
        
        formatted_string = f"Paper Analysis Results:\n\n"
        formatted_string += result
        
        self.logger.info("Successfully formatted paper analysis results")
        return formatted_string

def get_paper_reader_tool():
    """Return the paper reader tool"""
    logger.info("Getting paper reader tool")
    return PaperReaderTool() 