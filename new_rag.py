import warnings, os,sys, time, json, json, pickle, logging
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from Milvus import MilvusVector
from langchain.docstore.document import Document
import fitz
warnings.filterwarnings("ignore") # to supress warnings

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)

def unit_splitter(text):
    """
    DESCRIPTION : This is function is used to separate out units of measurement from LLM's response
    Input : string (LLM response)
    Output : unit, non_unit (numerical part/entire response if no unit present in response)
    """
    try:
        text_len = len(text.split())

        if text_len == 2:
            if text.split()[0].isnumeric() == True and text.split()[1][0].isalpha() == True:
                non_unit = text.split()[0]
                unit = text.split()[1]
                return unit, non_unit
            elif text.split()[1].isnumeric() == True and text.split()[0][0].isalpha() == True:
                non_unit = text.split()[1]
                unit = text.split()[0]
                return unit, non_unit
            else:
                unit = '-'
                non_unit = text
                return unit, non_unit
        elif text_len > 2:
            unit = '-'
            non_unit = text
            for word in text.split():
                if word.isnumeric() == True:
                    non_unit = word
                    index = text.split().index(word)
                    unit = text.split()[index+1]
                    break
            return unit, non_unit

        else:
            unit = '-'
            non_unit = text
            return unit, non_unit
    except:
        return '-', text

def process_chunk(args):
    pdf_content, filename = args
    pdf_document = fitz.open("pdf", pdf_content)
    chunk_documents = []

    for page_number in range(0, pdf_content.page_count):
        page = pdf_document[page_number]
        # Extract text from the page using PyMuPDF
        page_text = page.get_text()
        page_text = f"[Page no. {page}] " + page_text # add pg number at beginning

        # Skip pages without text content
        if not page_text.strip():
            continue
        # Create a document object
        document = Document(
            page_content=page_text,
            metadata={"page": page_number, "file_name": filename},
        )
        chunk_documents.append(document)

    pdf_document.close()  # Close the PDF document

    return chunk_documents
def indexing_pipeline(file, filename, embeddings):

    """
    DESCRIPTION: This function generates and saves/persists the Chroma vector database

    :param filepath: Location of the PDF/input file
    :param embeddings: Embedding model
    :return: the path of vector database
    """
    try:

        docs = process_chunk(file, filename)

        logging.info('PDF Loading done. Performing Chunking.')

        # Define the Text Chunker
        semantic_chunker = SemanticChunker(embeddings = embeddings,
                                           breakpoint_threshold_type="percentile",
                                           )

        semantic_chunks = semantic_chunker.create_documents([d.page_content for d in docs])

        semantic_chunk_vectorstore = MilvusVector().create_vector_store(filename ,semantic_chunks, embeddings)

        logging.info('VectorDB created and returned.')

        return semantic_chunk_vectorstore

    except Exception as e:
        logging.info(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        return f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"



def retrieval_pipeline(system_message :str, query, document_store, llm, embeddings, top_k=5):
    """
    DESCRIPTION: This function is the backbone of RAG, takes in user query/prompt, llm, vector db location, embedding model and returns the query respone.

    :param system_message: system message from config-file
    :param query: user-query
    :param document_store: location of vector DB
    :param llm: llm
    :param embeddings: embedding model
    :param top_k: top contexts to retrieve
    :return: the response for each query
    """
    try:
        logging.info('Loading vector database')
        # Loading the vector database

        retriever = document_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        qa = RetrievalQA.from_chain_type(llm = llm,
                                         chain_type="stuff",
                                         retriever=retriever,
                                         return_source_documents=True,
                                         )

        logging.info('VectorDB loaded. Retrieving response for the query.')

        # docs = db_loaded.similarity_search(query)

        result = qa({"query": query})

        response = result['result']

        logging.info('Response obtained.')

        # Fetching/retrieving langchain-context
        context_list = [] # list to contain top-k strings separated by comma
        context_pages = [] # list to contain each context's page no

        for ct in result['source_documents']:
            page_no = ct.page_content[10:].split(']')[0]
            context_pages.append(page_no)
            context_list.append(str(ct.page_content).replace('\n',' '))

        contexts = '\n'.join(context_list) # joining all contents of context_list to form a string

        logging.info('Contexts retrieved.')

        unit, unitless_response = unit_splitter(response)

        dictionary = {}

        dictionary['Query'] = query
        dictionary['Response'] = unitless_response
        dictionary['Unit'] = unit
        dictionary['Context'] = contexts
        dictionary['Page'] = context_pages[0]

        return dictionary

    except Exception as e:
        logging.info(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        return f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"

