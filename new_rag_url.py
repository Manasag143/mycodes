import warnings, difflib, requests
import os,sys, time, json

from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import pandas as pd
from tqdm import tqdm

from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from ESG_GIX_RAG_repo import Llama3_3
from ESG_evaluator_with_unit_allFY_sep24 import evaluate #...............support script for evaluation
from Unstructured_DB_creator import * #..................................support script for creating-and-persisting vectorstore
from JSON_parser import parser #.........................................support script to parse LLM responses into individual FY-responses
from typing import Any, Optional, Dict
from Llama3_3 import LLMGenerator # .....................................custom LLM (URL-request basis)

import pandas as pd
import openai, json


import warnings
warnings.filterwarnings("ignore") #......................................to supress warnings

# GPT 3.5 creds
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = "c3ecb5d7c1fb4244bfb5483ff308b2f1"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://crisil-gen-ai-uat.openai.azure.com/" # same as openai_api_base
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
os.environ['DEPLOYMENT_NAME'] = "gpt-35-turbo-16k"

# to read databases/vectorstores originally created using AzureOpenAI embeddings
embedding = AzureOpenAIEmbeddings(deployment = "text-embedding-3-small",
                                  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                                  openai_api_key=os.getenv("OPENAI_API_KEY"),
                                  chunk_size=8, show_progress_bar=False,
                                  disallowed_special=(),
                                  openai_api_type="azure")

pdf_folder = r"C:\Users\HarshalP\ESG\Data_oct17_30\Input" # custom folder where you place one or multiple pdfs
pdfs = os.listdir(pdf_folder)

db_folder = r"C:\Users\HarshalP\ESG\Data_oct17_30\DB_unstructured_MD" # folder where all vectorstores (pdf-wise) are persisted
print('> Creating vectorstore...\n')
db_creator(pdfs, db_folder) # creates databse/vectorstore

# NOTE: Ideally, db/vectorstore creation should be a one-time process for a PDF IFF db is persisted without change
# in any configuration

destination = r"C:\Users\HarshalP\ESG\Data_oct17_30\AWQ\Response" # folder where responses are saved in an excel sheet

dbs = os.listdir(db_folder)

# FETCHING USER-PROMPTS
with open(r'C:\Users\HarshalP\ESG\Data_sep2\Unstructured\Guided_kpi_prompt_2shot_433.json') as f: # json having our 'kpi : prompt' pairs
    kp = json.load(f)
kpi_list = list(kp.keys())
prompts = list(kp.values())

for db in dbs: # now we begin iterating through each PDF's vectorstore
    try:
        # Loading the DB
        db_loaded = Chroma(persist_directory= db_folder +"\\"+ db, embedding_function=embedding) # to load Chroma DB

        print('\n> CURRENT DB:',db)

        retriever = db_loaded.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # These empty-lists will be used to create output/response excel sheet
        q_list = []
        llm_output_list = []
        response_24_list = []
        response_23_list = []
        response_22_list = []
        c_list = []
        cr_list = []
        unit_list = []
        source_list = []
        page_list = []
        time_list = []

        print('\n> Generating responses...')
        for query in tqdm(prompts): # iterating through prompts from json
            if query != '\n' or "":

                question = query.split("####")[-2].split("####")[0].strip()  # one-liner question like "Extract CO2 level for all FYs."

                # Fetching top-5 contexts
                retrieved_docs = db_loaded.search(query= question, search_type="similarity",k=5)

                # NOTE: Our each prompt (or query) was in langchain's 'from_template' format specifically for GPT-4-o-mini. So it has all the components clubbed together.
                # Components it has is as follows:
                # 1. System message, included instructions to answer the query in JSON format
                # 2. 1st example, to implement 2-shot learning, includes a context, a question and example answer to that question
                # 3. 2nd example, to implement 2-shot learning, includes a context, a question and example answer to that question
                # 4. Review and guidences

                # Now since Llama3.3 has a different prompting-format, we had to separate out each component to fit them properly within the right special tokens.
                # This component-splitting is exactly what is done below

                system_message = query.split("Use the following example delimited by the tags <example> and </example> as reference:")[0].strip()
                example1_with_tag = "<example>\n"+query.split("<example>")[2].split("</example>")[0]+"</example>"
                example1_with_tag = example1_with_tag.replace("{{","{").replace("}}","}").strip()
                example2_with_tag = "<example>\n"+query.split("<example>")[2].split("</example>")[0]+"</example>"
                example2_with_tag = example2_with_tag.replace("{{","{").replace("}}","}").strip()
                review = "Review:-\n"+query.split("\nReview:-")[1].split("####")[0].strip()

                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{example1_with_tag.replace("'", '"')}

{example2_with_tag.replace("'", '"')}

{review}

####
{question}
####

&&&&
{retrieved_docs}
&&&&

Expected answer:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

                response, time_taken = Llama3_3.run(formatted_prompt) # URL-based llama3.3, returns a tuple --> (response, time)

                print("\nLLM Output:",response)
                print("Elapsed time:", time_taken)
                parsed_response = parser(response) # parses LLM-response according to the financial years
                print('> Parsed response:',parsed_response)

                #individual FY responses after parsing
                response_24, response_23, response_22, unit = parsed_response[0], parsed_response[1], parsed_response[2], parsed_response[3]

                time_list.append(time_taken)
                llm_output_list.append(response)

                ctx = [] # to gather all top-5 contexts
                for idx, ct in enumerate(retrieved_docs):
                    ctx.append(str(ct.page_content).replace('\n',' '))

                query = str(query).replace('"',"'")
                q_list.append(formatted_prompt)
                response_24_list.append(response_24)
                response_23_list.append(response_23)
                response_22_list.append(response_22)
                unit_list.append(unit)
                c_list.append('\n\n'.join(ctx))

        print('> Responses Generation Done.')

        # creates the output df/excel
        response_df = pd.DataFrame({
                           'KPI':kpi_list,
                            'Prompt': q_list,
                            "LLM Output": llm_output_list,
                            'Response FY 23-24' : response_24_list,
                            'Response FY 22-23': response_23_list,
                            'Response FY 21-22': response_22_list,
                            'Langchain context' : c_list,
                            'Unit': unit_list,
                            'Time elapsed': time_list,
                            })
        # saving the output excel to the destination
        response_df.to_excel(fr"{destination}\{db}.xlsx", index=False, engine='xlsxwriter')

        print('> Response excel saved!')
        print('> Evaluating generated responses...')
        evaluate(fr"{destination}\{db}.xlsx", evaluation_destination)

    except Exception as e:
        print(f"\n\nERROR AT RESPONSE GENERATION: {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
