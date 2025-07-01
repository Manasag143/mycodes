import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging
import hashlib
import httpx
from typing import List, Dict
import time
load_dotenv()
# from RAG import mergeDocs
import pandas as pd
import boto3
import time
import hashlib
import json
# deployment_id = os.getenv("embeddings")
deployment_id = "text-embedding-3-small"
print(deployment_id)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
httpx_client = httpx.Client(verify=False)

def getFilehash(file_path: str):
    with open(file_path, 'rb') as f:
        return hashlib.sha3_256(f.read()).hexdigest()
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1-mini")
 
client = AzureOpenAI(
    api_key="c3ecb5d7c1fb4244bfb5483ff308b2f1",
    api_version="2024-02-15-preview",
    azure_endpoint="https://crisil-gen-ai-uat.openai.azure.com/",
    http_client=httpx_client
)
queries = [
    "Extract Scope 1 emissions ?",
    "Extract Scope 2 emissions for all the given years?",
    "Extract the scope 3 emissions for all the given years.",
    "Extract the targets set for CO2 emission control ?"
    ""
]
 
 

class AzureOpenAIQueryPipeline:
    def __init__(self, pdf_path: str, queries_csv_path: str,model_name = "gpt-4o-mini"):
        self.model_name = model_name
        self.docs = mergeDocs(pdf_path, split_pages=False)
        queries_df = pd.read_excel(queries_csv_path)
        self.queries = queries_df["prompt"].tolist()
    def query_azure_openai(self, client, deployment: str) -> pd.DataFrame:
        """
        Queries the Azure OpenAI chat completion API for a list of queries using the provided context.
        Args:
            client: AzureOpenAI client instance.
            deployment (str): Deployment name or model identifier.
        Returns:
            pd.DataFrame: DataFrame containing usage statistics and responses for each query.
        """
        sys_prompt = f"""You must answer the question strictly based on the below given context.
        Context:
{self.docs[0]["context"]}\n\n"""
        prompt = """
        Question:{query}
        Answer:"""
 
        messages=[
                        {"role": "system", "content": sys_prompt}
                    ]
        results = []
        for query in self.queries:
            start = time.perf_counter()
            try:
                try_message = messages + [{"role": "user", "content": prompt.format(query=query)}]
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages= try_message,
                    temperature=0.1,
                    seed=42
                )
                end = time.perf_counter()
                completion_tokens = response.usage.completion_tokens
                input_tokens = response.usage.prompt_tokens
                usage = dict(
                    query=query,
                    response=response.choices[0].message.content,
                    completion_tokens=completion_tokens,
                    input_tokens=input_tokens,
                    response_time=f"{end - start:.2f}"
                )
                new_message = [{"role": "user", "content": prompt.format(query=query)},
                               {"role": "assistant", "content": response.choices[0].message.content} ]
                messages.extend(new_message)
            except Exception as e:
                end = time.perf_counter()
                usage = dict(
                    query=query,
                    response=f"Error: {str(e)}",
                    cached_tokens=None,
                    completion_tokens=None,
                    input_tokens=None,
                    response_time=f"{end - start:.2f}"
                )
            results.append(usage)
        return pd.DataFrame(results)








import os
import pymupdf
import glob
def mergeDocs(path: str, split_pages: bool = True):
    """
    Extracts text from all PDF files in the specified directory or from a single PDF file and organizes it into a structured format.
    Args:
        path (str): The directory path containing the PDF files to process, or a single PDF file path.
        split_pages (bool): If True, splits each PDF into individual pages; if False, extracts the entire PDF as a single context.
    Returns:
        List[Dict[str, Dict[str, str]]]: A list of dictionaries where each dictionary represents either a page
        or the entire PDF, depending on split_pages, with the following structure:
            - "context" (str): The extracted text content.
            - "metadata" (Dict[str, str]): Metadata about the context, including:
                - "file_name" (str): The name of the PDF file.
                - "pg_no" (int or str): The page number within the PDF file, or "all" if not splitting.
    """
    docs = []
    if os.path.isfile(path) and path.lower().endswith('.pdf'):
        files = [path]
    else:
        files = glob.glob(f'{path}/*.pdf')
    for file in files:
        doc = pymupdf.open(file)
        if split_pages:
            pages = [page.get_text() for page in doc]
            temp = [page for page in pages if page != '']
            temp = [{"context": page, "metadata": {"file_name": file, "pg_no": n}} for n, page in enumerate(temp)]
            docs.extend(temp)
        else:
            full_text = "".join([page.get_text() for page in doc])
            if full_text.strip():
                docs.append({"context": full_text, "metadata": {"file_name": file, "pg_no": "all"}})
    return docs















