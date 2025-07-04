import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
import requests
import warnings
import hashlib
import logging
import json
import ast
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Suppress SSL warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def getFilehash(file_path: str):
    """Generate SHA3-256 hash of a file"""
    with open(file_path, 'rb') as f:
        return hashlib.sha3_256(f.read()).hexdigest()

class HostedLLM(LLM):
    """Custom LLM class for hosted Llama model"""
    
    def __init__(self, endpoint: str, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint
    
    @property
    def _llm_type(self) -> str:
        return "Hosted LLM"
    
    def _call(self, prompt: str, stop=None, run_manager: CallbackManagerForLLMRun = None) -> str:
        """Make API call to hosted LLM"""
        try:
            prompt_template = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            
            payload = json.dumps({
                "provider": "tgi", 
                "deployment": "Llama 3.3 v1", 
                "spec_version": 1, 
                "input_text": prompt_template, 
                "params": {"temperature": 0.1}
            })
            
            headers = {
                'token': '0e53d2cf9f7', 
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                url=self.endpoint, 
                headers=headers, 
                data=payload, 
                verify=False
            )
            
            response_v = ast.literal_eval(response.text)
            resp_o = response_v['output']
            output = str(resp_o).replace(prompt_template, "")
            return output.strip()
            
        except Exception as e:
            return f"LLM Call Failed: {e}"

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
            logger.info(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
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
    """Main pipeline class for querying PDF content with Llama"""
    
    def __init__(self, pdf_path: str, queries_csv_path: str, llm_endpoint: str = "https://llmgateway.crisil.local/api/v1/llm"):
        """Initialize the pipeline"""
        self.llm = HostedLLM(endpoint=llm_endpoint)
        self.docs = mergeDocs(pdf_path, split_pages=False)
        
        # Load queries from Excel or CSV
        if queries_csv_path.endswith('.xlsx'):
            queries_df = pd.read_excel(queries_csv_path)
        else:
            queries_df = pd.read_csv(queries_csv_path)
            
        self.queries = queries_df["prompt"].tolist()
        self.pdf_path = pdf_path

    def query_llama(self, maintain_conversation: bool = True) -> pd.DataFrame:
        """Query the Llama API for a list of queries using the provided context"""
        sys_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{self.docs[0]["context"]}

"""
        
        prompt_template = """Question: {query}
Answer:"""
        
        # Initialize conversation history
        conversation_history = ""
        results = []
        
        for i, query in enumerate(self.queries, 1):
            start = time.perf_counter()
            
            try:
                # Build the full prompt with conversation history
                if maintain_conversation and conversation_history:
                    full_prompt = f"{sys_prompt}\n{conversation_history}\n{prompt_template.format(query=query)}"
                else:
                    full_prompt = f"{sys_prompt}\n{prompt_template.format(query=query)}"
                
                # Get response from Llama
                response_text = self.llm._call(full_prompt)
                end = time.perf_counter()
                
                # Calculate token approximations (rough estimation)
                input_tokens = len(full_prompt.split())
                completion_tokens = len(response_text.split()) if response_text else 0
                
                usage = {
                    "query_id": i,
                    "query": query,
                    "response": response_text,
                    "completion_tokens": completion_tokens,
                    "input_tokens": input_tokens,
                    "response_time": f"{end - start:.2f}"
                }
                
                # Update conversation history for next iteration
                if maintain_conversation:
                    conversation_history += f"\nQuestion: {query}\nAnswer: {response_text}\n"
                
            except Exception as e:
                end = time.perf_counter()
                
                usage = {
                    "query_id": i,
                    "query": query,
                    "response": f"Error: {str(e)}",
                    "completion_tokens": None,
                    "input_tokens": None,
                    "response_time": f"{end - start:.2f}"
                }
            
            results.append(usage)
        
        return pd.DataFrame(results)
    
    def save_results(self, results_df: pd.DataFrame, output_path: str = None):
        """Save results to CSV file"""
        if output_path is None:
            # Generate filename based on PDF name and timestamp
            pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"llama_query_results_{pdf_name}_{timestamp}.csv"
        
        results_df.to_csv(output_path, index=False)
        return output_path

def main():
    """Example usage of the pipeline"""
    # Configuration
    pdf_path = r"C:\Users\DeshmukhK\Downloads\EWS\Earning call transcripts\Sterling Q2 FY23.pdf"
    queries_csv_path = r"C:\Users\DeshmukhK\Downloads\EWS_LLAMA_prompts.xlsx"
    
    # Initialize pipeline
    pipeline = LlamaQueryPipeline(
        pdf_path=pdf_path,
        queries_csv_path=queries_csv_path
    )
    
    # Run queries
    results_df = pipeline.query_llama(maintain_conversation=True)
    
    # Save results
    output_file = pipeline.save_results(results_df)
    
    return results_df

if __name__ == "__main__":
    results = main()
