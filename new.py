"""
Author: Harshal Pal
Team: Central AI
"""
import yaml, os, json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, Optional, List
from scripts.rag import indexing_pipeline, retrieval_pipeline
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import logging, requests
from constants import *
from models import RequestModel
from urllib3 import Retry
from requests.adapters import HTTPAdapter
from io import BytesIO
from Milvus import MilvusVector

logger = logging.getLogger('centralai_esg').setLevel(logging.INFO)

local_download_path = os.path.join(os.getcwd(), "S3_downloads")
if not os.path.exists(local_download_path):
    os.mkdir(local_download_path)

app = FastAPI()

def get_prompts():
    """ Retrieve data from the API """
    try:
        
        url = URL_ESG_PROMPTS

        payload = json.dumps({
            "modelName": "Gen_AI_KPI"
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload, verify=False)

        prompts_dict = json.loads(response.text)

    except Exception as error:
        print(error)
    return prompts_dict


# Accessing config-file
with open(r"phoenix-centralai-esg\config\config_dev.yaml", mode="r", encoding="utf-8") as file:
    config_file = file.read()
config = yaml.safe_load(config_file)

class Params(BaseModel):
    """
    Params - pydantic model to validate the structure of input JSON for post request call
    """

    filename : Union[str, str]
    filepath : Union[str, str]
    callbackurl : Union[str, str]
    reportid : Union[str, str]

class Model_params:
    '''
    contains initializations for LLM and embedding model
    '''
    def __init__(self):

        self.embedding = AzureOpenAIEmbeddings(deployment = deployment,
                                          azure_endpoint = azure_endpoint,
                                          openai_api_key = openai_api_key,
                                          chunk_size = 8,
                                          show_progress_bar=False,
                                          disallowed_special=(),
                                          openai_api_type = openai_api_type,
                                          )
        self.llm = AzureChatOpenAI(openai_api_key = openai_api_key,
                              model = model,
                              temperature = 0.0,
                              api_version = api_version,
                              azure_endpoint = azure_endpoint,
                              )

document_store = None




# @app.post("/rag/")
class CL_RAG:
    '''This class is meant for vectorDB creation using indeixng_pipeline and info-retrieval with response generation using retrieval pipeline'''

    def callback_response(self, callback_payload, params, headers=None):
        if not headers:
            headers = {"Content-Type": "application/json"}
        # print(callback_payload)
        request_session = requests.Session()

        retries = Retry(total=5,
                        status_forcelist=[429, 500, 502, 503, 504])

        request_session.mount('http://', HTTPAdapter(max_retries=retries))
        request_session.mount('https://', HTTPAdapter(max_retries=retries))

        try:
            r = request_session.post(params.callback_url, json=callback_payload, headers=headers, timeout=10)
        except ConnectionError as e:
            print(e.__class__.__name__)
            print("Failed to connect to callback url.")
        except Exception as e:
            print(e)
        else:
            if r.status_code == 200:
                print("Successfully send response to, ", params.callback_url)  # r.status_code==200
            else:
                print("CallbackURL status code: ", r.status_code)

    def rag(self, params: RequestModel):
        """
        End point to deploy entire RAG pipeline as single point of contact for rest call.

        :params: object for class Params
        :return: a dictionary with company names as key and each value is a sub-dictionary with kpi:response structure
        """

        models = Model_params() # object for LLM and embedding-model instantiation

        logging.info("Entered RAG Pipeline")
        system_message = config["prompts"]["SYS_MESSAGE"]

        response = {} # would store company-names and kpi:response
        pdf_key = params.filepath + params.filename


        local_filename = f"{local_download_path}/{params.filename}"
        print(params.filename, local_filename)
        # s3_client.download_file(s3_bucket, s3_file_name, local_filename)

        pdf_obj = s3_client.get_object(Bucket = s3_bucket, Key = pdf_key)
        pdf_bytes = pdf_obj['Body'].read()
        file = BytesIO(pdf_bytes)

        document_store = indexing_pipeline(file= file, filepath = params.filename,  embeddings=models.embedding)
        rows = get_prompts()

        answer_dict = {}

        for i in rows['resultset']:
            kpi, query = i["field_name"], i["mapping_in_source_file"]
            try:

                prompt = system_message + query

                output = retrieval_pipeline(system_message = system_message, query = prompt, document_store=document_store, llm = models.llm, embeddings = models.embedding)

                output['Source'] = pdf_key
                answer_dict[kpi] = output


            except Exception as e:
                return f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"

        output_json = answer_dict.json()

        # delete vector store
        MilvusVector().delete_vector_store(params.filename)

        # send the output json
        self.callback_response(output_json, params)

@app.post("/test")
async def test():
    '''just a test'''
    return {"Hello":"World"}
