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

                llm = HostedLLM(endpoint="https://llmgateway.crisil.local/api/v1/llm")

                response_text = llm._call(full_prompt)

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
 
# Example usage:

if __name__ == "__main__":

    # Initialize pipeline

    pipeline = LlamaQueryPipeline(

        pdf_path=r"C:\Users\DeshmukhK\Downloads\EWS\Earning call transcripts\Sterling Q2 FY23.pdf",

        queries_csv_path=r"C:\Users\DeshmukhK\Downloads\EWS_LLAMA_prompts.xlsx"

    )

    # Run queries

    results_df = pipeline.query_llama()

    print(results_df)

    # Save results

    results_df.to_csv("llama_query_results_4.csv", index=False)
 
import json

import ast

from langchain.llms.base import LLM

from langchain.callbacks.manager import CallbackManagerForLLMRun

class HostedLLM(LLM):

    def __init__(self, endpoint:str, **kwargs):

        super().__init__(**kwargs)

    def _llm_type(self) -> str:

        return "Hosted LLM"

    def _call(self, prompt:str, stop=None, run_manager: CallbackManagerForLLMRun=None) -> str:

        try:

            prompt_template = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

            {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """

            payload = json.dumps({"provider": "tgi", "deployment": "Llama 3.3 v1", "spec_version": 1, "input_text": prompt_template, "params": { "temperature": 0.1}})

            headers = {'token': '0e53d2cf9f724a94a6ca0a9c880fdee7', 'Content-Type': 'application/json'}

            response = requests.request("POST", url="https://llmgateway.crisil.local/api/v1/llm", headers=headers, data=payload, verify=False)

            response_v = ast.literal_eval(response.text)

            resp_o = response_v['output']

            output = str(resp_o).replace(prompt_template, "")

            return output.strip()

        except Exception as e:

            return f"LLM Call Failed: {e}"
 
3rd Prompt:
I have a list of identified red flags related to a company's financial health and operations. I need help categorizing these red flags into the following categories:
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
By following these guidelines, provide a summary that accurately and comprehensively represents each category of red flags, including all relevant details and quantitative data present from each red flag identified.
 
2nd Prompt:
Remove the duplicates from the above context. Also if the Original Quote and Keyword identifies is same remove them.

 
#2nd Iteration:

response_text = llm._call(2ndprompt+ output_of_previous_prompt)

#3rd Iteration:

response_text = llm._call(3rdprompt+ output_of_previous_prompt)

 
