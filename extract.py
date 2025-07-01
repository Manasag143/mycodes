import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
import requests
import warnings
import hashlib
import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress SSL warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Llama URL
LLAMA_URL = "https://ue1-llm.crisil.local/llama3_3/70b/llm/"

def getFilehash(file_path: str):
    with open(file_path, 'rb') as f:
        return hashlib.sha3_256(f.read()).hexdigest()

class LLMGenerator:
    """Component to generate responses from hosted LLM model"""
    def __init__(self, url: str = LLAMA_URL):
        self.url = url
        self.generation_kwargs = {
            "max_new_tokens": 2048,
            "return_full_text": False,
            "temperature": 0.1
        }
        # Use a session for connection pooling
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification
    
    def run(self, prompt: str) -> str:
        """Send prompt to LLM and get response with timeout"""
        start_time = time.time()
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        try:
            # Add timeout parameter to prevent hanging
            response = self.session.post(self.url, json=body, timeout=30)
            
            if response.status_code != 200:
                print(f"API Error: Status {response.status_code}")
                return f"Error: LLM API error: Status {response.status_code}"
            
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                result = response_json[0].get('generated_text', '')
                print(f"LLM API call took {time.time() - start_time:.2f} seconds")
                return result
            else:
                print(f"Unexpected API response format: {response_json}")
                return "Error: Unexpected response format from LLM API"
                
        except requests.exceptions.Timeout:
            print(f"API call timed out after {time.time() - start_time:.2f} seconds")
            return "Error: Request to LLM API timed out"
        except Exception as e:
            print(f"API call failed after {time.time() - start_time:.2f} seconds: {e}")
            return f"Error: {str(e)}"

class PDFExtractor:
    """Class for extracting text from PDF files"""
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page of a PDF file"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                pages.append({
                    "page_num": page_num + 1,
                    "text": text
                })
                
            # Explicitly close the document to free memory
            doc.close()
            print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages
        except Exception as e:
            print(f"PDF extraction error after {time.time() - start_time:.2f} seconds: {e}")
            raise

def mergeDocs(pdf_path: str, split_pages: bool = False) -> List[Dict[str, Any]]:
    """Merge PDF documents into a single context"""
    extractor = PDFExtractor()
    pages = extractor.extract_text_from_pdf(pdf_path)
    
    if split_pages:
        return [{"context": page["text"], "page_num": page["page_num"]} for page in pages]
    else:
        # Merge all pages into single context
        all_text = "\n".join([page["text"] for page in pages])
        return [{"context": all_text}]

class LlamaQueryPipeline:
    def __init__(self, pdf_path: str, queries_csv_path: str, llm_url: str = LLAMA_URL):
        self.llm_generator = LLMGenerator(url=llm_url)
        self.docs = mergeDocs(pdf_path, split_pages=False)
        queries_df = pd.read_excel(queries_csv_path)
        self.queries = queries_df["prompt"].tolist()
    
    def query_llama(self) -> pd.DataFrame:
        """
        Queries the Llama API for a list of queries using the provided context.
        
        Returns:
            pd.DataFrame: DataFrame containing usage statistics and responses for each query.
        """
        sys_prompt = f"""You must answer the question strictly based on the below given context.
Context:
{self.docs[0]["context"]}\n\n"""
        
        prompt_template = """Question: {query}
Answer:"""
        
        # Initialize conversation history
        conversation_history = ""
        results = []
        
        for query in self.queries:
            start = time.perf_counter()
            try:
                # Build the full prompt with conversation history
                if conversation_history:
                    full_prompt = f"{sys_prompt}\n{conversation_history}\n{prompt_template.format(query=query)}"
                else:
                    full_prompt = f"{sys_prompt}\n{prompt_template.format(query=query)}"
                
                # Get response from Llama
                response_text = self.llm_generator.run(full_prompt)
                end = time.perf_counter()
                
                # Calculate token approximations (rough estimation)
                input_tokens = len(full_prompt.split())
                completion_tokens = len(response_text.split()) if response_text else 0
                
                usage = dict(
                    query=query,
                    response=response_text,
                    completion_tokens=completion_tokens,
                    input_tokens=input_tokens,
                    response_time=f"{end - start:.2f}"
                )
                
                # Update conversation history for next iteration
                conversation_history += f"\nQuestion: {query}\nAnswer: {response_text}\n"
                
            except Exception as e:
                end = time.perf_counter()
                usage = dict(
                    query=query,
                    response=f"Error: {str(e)}",
                    completion_tokens=None,
                    input_tokens=None,
                    response_time=f"{end - start:.2f}"
                )
            
            results.append(usage)
            
        return pd.DataFrame(results)

# Example usage queries (for testing)
queries = [
    "Extract Scope 1 emissions ?",
    "Extract Scope 2 emissions for all the given years?",
    "Extract the scope 3 emissions for all the given years.",
    "Extract the targets set for CO2 emission control ?"
]

# Example usage:
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = LlamaQueryPipeline(
        pdf_path="path/to/your/document.pdf",
        queries_csv_path="path/to/your/queries.xlsx"
    )
    
    # Run queries
    results_df = pipeline.query_llama()
    print(results_df)
    
    # Save results
    results_df.to_csv("llama_query_results.csv", index=False)
