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

# Suppress SSL warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def getFilehash(file_path: str):
    """Generate SHA3-256 hash of a file"""
    with open(file_path, 'rb') as f:
        return hashlib.sha3_256(f.read()).hexdigest()

class HostedLLM:
    """Custom LLM class for hosted Llama model"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def _call(self, prompt: str) -> str:
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
                url="https://llmgateway.crisil.local/api/v1/llm", 
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

# Suppress SSL warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def getFilehash(file_path: str):
    """Generate SHA3-256 hash of a file"""
    with open(file_path, 'rb') as f:
        return hashlib.sha3_256(f.read()).hexdigest()

class HostedLLM:
    """Custom LLM class for hosted Llama model"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
    
    def _call(self, prompt: str) -> str:
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
                url="https://llmgateway.crisil.local/api/v1/llm", 
                headers=headers, 
                data=payload, 
                verify=False,
                timeout=30
            )
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Text: {response.text[:500]}...")  # First 500 chars
            
            if response.status_code != 200:
                return f"LLM Call Failed: HTTP {response.status_code} - {response.text}"
            
            response_v = ast.literal_eval(response.text)
            resp_o = response_v['output']
            output = str(resp_o).replace(prompt_template, "")
            return output.strip()
            
        except requests.exceptions.RequestException as e:
            return f"LLM Call Failed - Network Error: {e}"
        except json.JSONDecodeError as e:
            return f"LLM Call Failed - JSON Error: {e}"
        except KeyError as e:
            return f"LLM Call Failed - Missing Key: {e}"
        except Exception as e:
            return f"LLM Call Failed - General Error: {e}"

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
    
    def __init__(self, pdf_path: str, queries_csv_path: str, llm_endpoint: str = "https://llmgateway.crisil.local/api/v1/llm", previous_results_path: str = None):
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
        
        # Load previous results if provided
        self.previous_results = None
        if previous_results_path and os.path.exists(previous_results_path):
            self.previous_results = pd.read_csv(previous_results_path)
    
    def query_llama_with_chaining(self, new_queries_csv_path: str, iteration_number: int = 2) -> pd.DataFrame:
        """Query the Llama API using previous results for chaining"""
        if self.previous_results is None:
            raise ValueError("No previous results loaded. Please provide previous_results_path in __init__")
        
        # Load new queries
        if new_queries_csv_path.endswith('.xlsx'):
            new_queries_df = pd.read_excel(new_queries_csv_path)
        else:
            new_queries_df = pd.read_csv(new_queries_csv_path)
        
        new_queries = new_queries_df["prompt"].tolist()
        
        sys_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{self.docs[0]["context"]}

"""
        
        results = []
        
        # Process each new query with corresponding previous response
        for i, new_query in enumerate(new_queries):
            start = time.perf_counter()
            
            try:
                # Get the corresponding previous response (if available)
                if i < len(self.previous_results):
                    # Check if it's 3rd iteration (chained results) or 2nd iteration (original results)
                    if 'chained_response' in self.previous_results.columns:
                        previous_response = self.previous_results.iloc[i]['chained_response']
                    else:
                        previous_response = self.previous_results.iloc[i]['response']
                    
                    # Create chained prompt: new query + previous response
                    chained_prompt = f"""Previous Analysis: {previous_response}

Based on the above analysis and the original context, please answer: {new_query}

Answer:"""
                else:
                    # If no previous response available, use regular prompt
                    chained_prompt = f"""Question: {new_query}
Answer:"""
                
                full_prompt = f"{sys_prompt}\n{chained_prompt}"
                
                # Get response from Llama
                response_text = self.llm._call(full_prompt)
                end = time.perf_counter()
                
                # Calculate token approximations
                input_tokens = len(full_prompt.split())
                completion_tokens = len(response_text.split()) if response_text else 0
                
                usage = {
                    "iteration": iteration_number,
                    "query_id": i + 1,
                    "original_query": new_queries_df.iloc[i]["prompt"] if i < len(new_queries_df) else new_query,
                    "previous_response": previous_response if i < len(self.previous_results) else "",
                    "new_query": new_query,
                    "chained_response": response_text,
                    "completion_tokens": completion_tokens,
                    "input_tokens": input_tokens,
                    "response_time": f"{end - start:.2f}"
                }
                
            except Exception as e:
                end = time.perf_counter()
                
                usage = {
                    "iteration": iteration_number,
                    "query_id": i + 1,
                    "original_query": new_queries_df.iloc[i]["prompt"] if i < len(new_queries_df) else new_query,
                    "previous_response": previous_response if i < len(self.previous_results) else "",
                    "new_query": new_query,
                    "chained_response": f"Error: {str(e)}",
                    "completion_tokens": None,
                    "input_tokens": None,
                    "response_time": f"{end - start:.2f}"
                }
            
            results.append(usage)
        
        return pd.DataFrame(results)

    def query_llama(self, maintain_conversation: bool = True, enable_chaining: bool = False) -> pd.DataFrame:
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
        previous_output = ""  # For prompt chaining
        
        for i, query in enumerate(self.queries, 1):
            start = time.perf_counter()
            
            try:
                # Handle prompt chaining
                if enable_chaining and i > 1 and previous_output:
                    # For chained prompts, append previous output to current query
                    chained_query = f"{query}\n\nPrevious context: {previous_output}"
                else:
                    chained_query = query
                
                # Build the full prompt with conversation history
                if maintain_conversation and conversation_history:
                    full_prompt = f"{sys_prompt}\n{conversation_history}\n{prompt_template.format(query=chained_query)}"
                else:
                    full_prompt = f"{sys_prompt}\n{prompt_template.format(query=chained_query)}"
                
                # Get response from Llama
                response_text = self.llm._call(full_prompt)
                end = time.perf_counter()
                
                # Calculate token approximations (rough estimation)
                input_tokens = len(full_prompt.split())
                completion_tokens = len(response_text.split()) if response_text else 0
                
                usage = {
                    "query_id": i,
                    "query": query,
                    "chained_query": chained_query if enable_chaining else query,
                    "response": response_text,
                    "completion_tokens": completion_tokens,
                    "input_tokens": input_tokens,
                    "response_time": f"{end - start:.2f}"
                }
                
                # Update conversation history for next iteration
                if maintain_conversation:
                    conversation_history += f"\nQuestion: {chained_query}\nAnswer: {response_text}\n"
                
                # Store output for next chaining iteration
                if enable_chaining:
                    previous_output = response_text
                
            except Exception as e:
                end = time.perf_counter()
                
                usage = {
                    "query_id": i,
                    "query": query,
                    "chained_query": query,
                    "response": f"Error: {str(e)}",
                    "completion_tokens": None,
                    "input_tokens": None,
                    "response_time": f"{end - start:.2f}"
                }
                
                # Reset previous output on error
                if enable_chaining:
                    previous_output = ""
            
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
    
    # For first iteration (if not already done)
    pipeline = LlamaQueryPipeline(
        pdf_path=pdf_path,
        queries_csv_path=queries_csv_path
    )
    
    # Uncomment below for first iteration
    # results_df = pipeline.query_llama(maintain_conversation=True, enable_chaining=False)
    # output_file = pipeline.save_results(results_df)
    
    return None

def main_chained():
    """Example usage for chained queries using previous results"""
    # Configuration
    pdf_path = r"C:\Users\DeshmukhK\Downloads\EWS\Earning call transcripts\Sterling Q2 FY23.pdf"
    previous_results_path = r"llama_query_results.csv"  # Your existing results file
    new_queries_csv_path = r"C:\Users\DeshmukhK\Downloads\EWS_LLAMA_prompts_iteration2.xlsx"  # New prompts for chaining
    
    # Initialize pipeline with previous results
    pipeline = LlamaQueryPipeline(
        pdf_path=pdf_path,
        queries_csv_path="",  # Not needed for chaining
        previous_results_path=previous_results_path
    )
    
    # Run chained queries (2nd iteration)
    chained_results_df = pipeline.query_llama_with_chaining(new_queries_csv_path, iteration_number=2)
    
    # Save chained results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    chained_output_file = f"llama_chained_results_iteration2_{timestamp}.csv"
    chained_results_df.to_csv(chained_output_file, index=False)
    
    return chained_results_df

def main_third_iteration():
    """Example usage for 3rd iteration using 2nd iteration results"""
    # Configuration
    pdf_path = r"C:\Users\DeshmukhK\Downloads\EWS\Earning call transcripts\Sterling Q2 FY23.pdf"
    previous_results_path = r"llama_chained_results_iteration2_20250704_143022.csv"  # Your 2nd iteration results
    
    # Create Excel file with your 2nd and 3rd prompts
    third_iteration_prompts = [
        "Remove the duplicates from the above context. Also if the Original Quote and Keyword identifies is same remove them.",
        """I have a list of identified red flags related to a company's financial health and operations. I need help categorizing these red flags into the following categories:
1. Balance Sheet Issues: Red flags related to the company's assets, liabilities, equity, and overall financial position.
2. P&L (Income Statement) Issues: Red flags related to the company's revenues, expenses, profits, and overall financial performance.
3. Management Issues: Red flags related to the company's leadership, governance, and decision-making processes.
4. Regulatory Issues: Red flags related to the company's compliance with laws, regulations, and industry standards.
5. Industry and Market Issues: Red flags related to the company's position within its industry, market trends, and competitive landscape.
6. Operational Issues: Red flags related to the company's internal processes, systems, and infrastructure.
7. Financial Reporting and Transparency Issues: Red flags related to the company's financial reporting, disclosure, and transparency practices.
8. Strategic Issues: Red flags related to the company's overall strategy, vision, and direction.
Please review the below red flags and assign each one to the most relevant category. If a red flag could fit into multiple categories, please assign it to the one that seems most closely related, do not leave any flag unclassified or fit it into multiple categories. While classifying, classify it in a such a way that the flags come under the categories. 
Provide a comprehensive and detailed summary of each category of red flags, ensuring that all relevant information is included and accurately represented. When generating the summary, adhere to the following guidelines:
1. Retain all information: Ensure that no details are omitted or lost during the summarization process.
2. Maintain a neutral tone: Present the summary in a factual and objective manner, avoiding any emotional or biased language.
3. Focus on factual content: Base the summary solely on the information associated with each red flag, without introducing external opinions or assumptions.
4. Include all red flags: Incorporate every red flag within the category into the summary, without exception.
5. Balance detail and concision: Provide a summary that is both thorough and concise, avoiding unnecessary elaboration while still conveying all essential information.
6. Incorporate quantitative data: Wherever possible, include quantitative data and statistics to support the summary and provide additional context.
7. Category-specific content: Ensure that the summary is generated based solely on the content present within each category, without drawing from external information or other categories.
8. Summary should be factual, Avoid any subjective interpretations or opinions.
By following these guidelines, provide a summary that accurately and comprehensively represents each category of red flags, including all relevant details and quantitative data present from each red flag identified."""
    ]
    
    # Create temporary Excel file for 3rd iteration
    third_df = pd.DataFrame({"prompt": third_iteration_prompts})
    third_queries_path = "EWS_LLAMA_prompts_iteration3.xlsx"
    third_df.to_excel(third_queries_path, index=False)
    
    # Initialize pipeline with 2nd iteration results
    pipeline = LlamaQueryPipeline(
        pdf_path=pdf_path,
        queries_csv_path="",  # Not needed for chaining
        previous_results_path=previous_results_path
    )
    
    # Run 3rd iteration
    third_results_df = pipeline.query_llama_with_chaining(third_queries_path, iteration_number=3)
    
    # Save 3rd iteration results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    third_output_file = f"llama_chained_results_iteration3_{timestamp}.csv"
    third_results_df.to_csv(third_output_file, index=False)
    
    return third_results_df

if __name__ == "__main__":
    results = main()
