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

def clean_response(text: str) -> str:
    """Clean the response text by removing unwanted patterns"""
    # Remove any prompt repetition or unwanted prefixes
    # Add your specific cleaning logic here
    return text

class HostedLLM_Perplexity(LLM):
    """Custom LLM class for hosted Perplexity model"""
    
    def __init__(self, endpoint: str, **kwargs):
        super().__init__(**kwargs)
        self._endpoint = endpoint
    
    def _llm_type(self) -> str:
        return "Hosted LLM"
    
    def _call(self, prompt: str, stop=None, run_manager: CallbackManagerForLLMRun=None) -> str:
        """Make API call to hosted Perplexity LLM"""
        try:
            data = {
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.1
                }
            }
            
            response = requests.post(self._endpoint, json=data, verify=False, timeout=30)
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Text: {response.text[:500]}...")  # First 500 chars
            
            if response.status_code != 200:
                return f"LLM Call Failed: HTTP {response.status_code} - {response.text}"
            
            response_v = ast.literal_eval(response.text)
            print(response_v)
            text = response_v[0]["generated_text"]
            output = clean_response(text)
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
    """Main pipeline class for querying PDF content with Perplexity"""
    
    def __init__(self, pdf_path: str, queries_csv_path: str = None, llm_endpoint: str = "https://as1-lower-llm.crisil.local/perplexity/llama/70b/llm/", previous_results_path: str = None):
        """Initialize the pipeline with Perplexity LLM"""
        self.llm = HostedLLM_Perplexity(endpoint=llm_endpoint)
        self.docs = mergeDocs(pdf_path, split_pages=False)
        
        # Load queries from Excel or CSV (only if provided)
        if queries_csv_path:
            if queries_csv_path.endswith('.xlsx'):
                queries_df = pd.read_excel(queries_csv_path)
            else:
                queries_df = pd.read_csv(queries_csv_path)
            self.queries = queries_df["prompt"].tolist()
        else:
            self.queries = []
            
        self.pdf_path = pdf_path
        
        # Load previous results if provided
        self.previous_results = None
        if previous_results_path and os.path.exists(previous_results_path):
            self.previous_results = pd.read_csv(previous_results_path)

    def query_llama_with_chaining(self, new_queries_csv_path: str, iteration_number: int = 2) -> pd.DataFrame:
        """Query the Perplexity API using previous results for chaining"""
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
                
                # Get response from Perplexity
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
        """Query the Perplexity API for a list of queries using the provided context"""
        sys_prompt = f"""You are a financial analyst expert specializing in identifying red flags from earnings call transcripts and financial documents.

COMPLETE DOCUMENT TO ANALYZE:
{self.docs[0]["context"]}

Your task is to analyze the ENTIRE document above and identify ALL potential red flags with their complete conversational context.

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- Number each red flag sequentially (1, 2, 3, etc.)
- Start each entry with: "The potential red flag you observed - [brief description]"
- Follow with "Complete Conversation Context:" and include:
  * The question/topic that led to the concerning response
  * The full response from management
  * Any follow-up questions and responses related to this red flag
  * Speaker names and their roles (if mentioned)
- Include page references where available: (Page X)
- Provide broader context (2-3 exchanges before and after if relevant)
- Ensure comprehensive analysis of the entire document
- Capture the flow of conversation to understand the full context

ENHANCED EXAMPLE FORMAT:
1. The potential red flag you observed - Debt reduction lower than expected with evasive management response
Complete Conversation Context:
"Analyst: Given the strong cash flow this quarter, what are your plans for debt reduction? Are you on track to meet your deleveraging targets?

Vikrant Kashyap: Have you -- are you able to reduce any debt in quarter one?

Ramesh Kalyanaraman: So India, we are not reduced, but the cash balance has been increased. We have maintained our debt levels but improved our liquidity position. The focus has been on operational efficiency rather than debt reduction at this time.

Follow-up Analyst: Can you provide more specifics on the debt reduction timeline?

Ramesh Kalyanaraman: We will evaluate this on a quarter-by-quarter basis depending on market conditions and investment opportunities." (Page 9)

2. The potential red flag you observed - Margin pressure with vague explanations
Complete Conversation Context:
"[Include full conversational flow around margin discussions]" (Page X)

Continue this exact format for ALL red flags identified throughout the document. Ensure you capture the complete conversational context, not just isolated quotes.

"""
        
        prompt_template = """Question: {query}

Analyze the complete document and provide ALL red flags with their complete conversational context in the exact numbered format specified above. Be thorough and comprehensive - cover the entire document and include full conversation flows around each red flag.

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
                
                # Get response from Perplexity
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
            output_path = f"perplexity_query_results_{pdf_name}_{timestamp}.csv"
        
        results_df.to_csv(output_path, index=False)
        return output_path

def run_four_iteration_pipeline():
    """
    Complete 4-iteration pipeline for financial red flag analysis using Perplexity:
    1. Initial red flag identification with conversational context
    2. Deduplication and cleanup
    3. Categorization of red flags
    4. Detailed summary generation
    """
    
    # Configuration
    pdf_path = r"Kalyan June 24.pdf"
    queries_csv_path = r"EWS_prompts_v2.xlsx"
    
    print("Starting complete 4-iteration pipeline with Perplexity...")
    
    # ITERATION 1: Initial red flag identification with conversational context
    print("Running 1st iteration - Initial Analysis with Conversational Context...")
    pipeline_1st = LlamaQueryPipeline(
        pdf_path=pdf_path,
        queries_csv_path=queries_csv_path,
        llm_endpoint="https://as1-lower-llm.crisil.local/perplexity/llama/70b/llm/"
    )
    
    # Run 1st iteration
    first_results_df = pipeline_1st.query_llama(maintain_conversation=True, enable_chaining=False)
    
    first_output_file = pipeline_1st.save_results(first_results_df, "perplexity_query_results_iteration1.csv")
    print(f"1st iteration completed. Results saved to: {first_output_file}")
    
    # Get first response for chaining
    first_response = first_results_df.iloc[0]['response']
    
    # ITERATION 2: Deduplication and cleanup
    print("Running 2nd iteration - Deduplication...")
    second_prompt = """Remove the duplicates from the above context. Also if the Complete Conversation Context and red flag description are the same, remove them. 
    
Ensure that:
1. No duplicate red flags are present
2. Similar or redundant conversational contexts are consolidated
3. All unique red flags with their complete conversational context are retained
4. The output maintains the structure, clarity, and conversational flow of the original analysis
5. Preserve the complete conversational context for each unique red flag"""
    
    second_full_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{pipeline_1st.docs[0]["context"]}

Previous Analysis: {first_response}

Based on the above analysis and the original context, please answer: {second_prompt}

Answer:"""
    
    second_response = pipeline_1st.llm._call(second_full_prompt)
    
    # ITERATION 3: Categorization of red flags
    print("Running 3rd iteration - Categorization...")
    third_prompt = """Based on the deduplicated red flags with their complete conversational context identified in the previous analysis, I need help categorizing these red flags into the following 8 categories:

1. Balance Sheet Issues: Red flags related to the company's assets, liabilities, equity, and overall financial position.
2. P&L (Income Statement) Issues: Red flags related to the company's revenues, expenses, profits, and overall financial performance.
3. Management Issues: Red flags related to the company's leadership, governance, and decision-making processes.
4. Regulatory Issues: Red flags related to the company's compliance with laws, regulations, and industry standards.
5. Industry and Market Issues: Red flags related to the company's position within its industry, market trends, and competitive landscape.
6. Operational Issues: Red flags related to the company's internal processes, systems, and infrastructure.
7. Financial Reporting and Transparency Issues: Red flags related to the company's financial reporting, disclosure, and transparency practices.
8. Strategic Issues: Red flags related to the company's overall strategy, vision, and direction.

**Task**: Take each red flag WITH ITS COMPLETE CONVERSATIONAL CONTEXT from the deduplicated list in the previous analysis and assign it to the most appropriate category above.

**Rules for categorization**:
- If a red flag could fit into multiple categories, assign it to the one that seems most closely related
- Do not leave any flag unclassified
- Do not assign any flag to multiple categories
- Ensure clear classification that groups similar issues together
- MAINTAIN the complete conversational context for each red flag in the categorization

**Output Format**:
### 1. Balance Sheet Issues
- [Red flag 1 with complete conversational context from previous analysis]
- [Red flag 2 with complete conversational context from previous analysis]

### 2. P&L (Income Statement) Issues
- [Red flag 1 with complete conversational context from previous analysis]
- [Red flag 2 with complete conversational context from previous analysis]

Continue this format for all 8 categories, ensuring every red flag with its complete conversational context from the previous analysis is categorized."""
    
    third_full_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{pipeline_1st.docs[0]["context"]}

Previous Analysis: {second_response}

Based on the above analysis and the original context, please answer: {third_prompt}

Answer:"""
    
    third_response = pipeline_1st.llm._call(third_full_prompt)
    
    # ITERATION 4: Detailed summary generation
    print("Running 4th iteration - Summary Generation...")
    fourth_prompt = """Based on the categorized red flags with their complete conversational context from the previous analysis, provide a comprehensive and detailed summary of each category of red flags. Follow these guidelines:

1. **Retain all conversational information**: Ensure that no conversational details, questions, or responses are omitted during the summarization process
2. **Maintain a neutral tone**: Present the summary in a factual and objective manner, avoiding any emotional or biased language
3. **Focus on factual content**: Base the summary solely on the conversational information associated with each red flag
4. **Include all red flags**: Incorporate every red flag within the category into the summary, without exception
5. **Balance detail and concision**: Provide a summary that captures the conversational flow while being thorough and concise
6. **Incorporate quantitative data**: Include all quantitative data and statistics mentioned in the conversations
7. **Category-specific content**: Ensure that the summary captures the conversational context present within each category
8. **Summary should be factual**: Avoid any subjective interpretations while preserving the conversational context

Format the output exactly like this example:
### 1. Balance Sheet Issues
[Detailed paragraph summary of all balance sheet related red flags incorporating the conversational context, questions asked, management responses, and follow-up discussions with specific data points and factual information]

### 2. P&L (Income Statement) Issues  
[Detailed paragraph summary of all P&L related red flags incorporating the conversational context, questions asked, management responses, and follow-up discussions with specific data points and factual information]

Continue this format for all 8 categories. Each summary should be a comprehensive paragraph that captures all the relevant red flags, their conversational contexts, and their details within that category."""
    
    fourth_full_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{pipeline_1st.docs[0]["context"]}

Previous Analysis: {third_response}

Based on the above analysis and the original context, please answer: {fourth_prompt}

Answer:"""
    
    fourth_response = pipeline_1st.llm._call(fourth_full_prompt)
    
    # Save all results together
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    all_results = pd.DataFrame({
        "iteration": [1, 2, 3, 4],
        "stage": [
            "Initial Analysis with Conversational Context",
            "Deduplication", 
            "Categorization",
            "Summary Generation"
        ],
        "prompt": [
            first_results_df.iloc[0]['query'],  # Original query from 1st iteration
            second_prompt,
            third_prompt,
            fourth_prompt
        ],
        "response": [
            first_response,
            second_response,
            third_response,
            fourth_response
        ],
        "timestamp": [timestamp, timestamp, timestamp, timestamp]
    })
    
    # Save complete results
    complete_output_file = f"complete_perplexity_pipeline_results_{timestamp}.csv"
    all_results.to_csv(complete_output_file, index=False)
    
    print(f"Complete 4-iteration pipeline with Perplexity finished!")
    print(f"1st iteration: {first_output_file}")
    print(f"Complete results: {complete_output_file}")
    print(f"Final categorized summary with conversational context is in row 4 of {complete_output_file}")
    
    return all_results

def main():
    """Example usage of the pipeline with Perplexity"""
    # Configuration
    pdf_path = r"C:\Users\DeshmukhK\Downloads\EWS\Earning call transcripts\Sterling Q2 FY23.pdf"
    queries_csv_path = r"C:\Users\DeshmukhK\Downloads\EWS_LLAMA_prompts.xlsx"
    
    # For first iteration (if not already done)
    pipeline = LlamaQueryPipeline(
        pdf_path=pdf_path,
        queries_csv_path=queries_csv_path,
        llm_endpoint="https://as1-lower-llm.crisil.local/perplexity/llama/70b/llm/"
    )
    
    # Uncomment below for first iteration
    # results_df = pipeline.query_llama(maintain_conversation=True, enable_chaining=False)
    # output_file = pipeline.save_results(results_df)
    
    return None

def main_chained():
    """Example usage for chained queries using previous results with Perplexity"""
    # Configuration - Updated for relative paths
    pdf_path = r"Kalyan June 24.pdf"
    previous_results_path = r"perplexity_query_results.csv"  # Your existing results file
    new_queries_csv_path = r"EWS_prompts_v2_iteration2.xlsx"  # New prompts for chaining
    
    # Initialize pipeline with previous results
    pipeline = LlamaQueryPipeline(
        pdf_path=pdf_path,
        queries_csv_path=None,  # Not needed for chaining
        previous_results_path=previous_results_path,
        llm_endpoint="https://as1-lower-llm.crisil.local/perplexity/llama/70b/llm/"
    )
    
    # Run chained queries (2nd iteration)
    chained_results_df = pipeline.query_llama_with_chaining(new_queries_csv_path, iteration_number=2)
    
    # Save chained results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    chained_output_file = f"perplexity_chained_results_iteration2_{timestamp}.csv"
    chained_results_df.to_csv(chained_output_file, index=False)
    
    return chained_results_df

if __name__ == "__main__":
    # Run the improved 4-iteration pipeline with Perplexity
    all_results = run_four_iteration_pipeline()
