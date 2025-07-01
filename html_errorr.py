import logging
import time
import json
import os
import requests
from datetime import date
from colorama import Fore, Style, init
from typing import Dict, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUBE_DETAILS_DIR = os.path.join(BASE_DIR, "cube_details")
IMPORT_HISTORY_FILE = os.path.join(BASE_DIR, "import_history.json")
history_file = os.path.join(BASE_DIR, "conversation_history.json")
vector_db_path = os.path.join(BASE_DIR, "vector_db")
config_file = os.path.join(BASE_DIR, "config.json")

def safe_create_bm25_retriever(documents: List[Document], k: int = 10):
    """
    Safely create BM25Retriever with detailed error reporting
    """
    print(f"🔄 Attempting to create BM25Retriever with {len(documents)} documents, k={k}")
    
    # Step 1: Validate input documents
    try:
        if not documents:
            print("❌ ERROR: Empty documents list provided")
            return None
        
        print(f"✅ Step 1: Document validation passed - {len(documents)} documents")
        
        # Check document structure
        for i, doc in enumerate(documents[:3]):  # Check first 3 docs
            print(f"   📄 Document {i}: {doc.page_content[:100]}...")
            if not hasattr(doc, 'page_content'):
                print(f"❌ ERROR: Document {i} missing 'page_content' attribute")
                return None
            if not doc.page_content or not doc.page_content.strip():
                print(f"❌ ERROR: Document {i} has empty content")
                return None
        
    except Exception as e:
        print(f"❌ ERROR in document validation: {type(e).__name__}: {e}")
        return None

    # Step 2: Check BM25Retriever class availability
    try:
        print(f"✅ Step 2: BM25Retriever class available: {BM25Retriever}")
        print(f"   📋 Available methods: {[method for method in dir(BM25Retriever) if not method.startswith('_')]}")
        
        # Check if from_documents method exists
        if not hasattr(BM25Retriever, 'from_documents'):
            print("❌ ERROR: BM25Retriever.from_documents method not found!")
            print("   This suggests an API change or version incompatibility")
            return None
        else:
            print("✅ from_documents method is available")
            
    except NameError as e:
        print(f"❌ ERROR: BM25Retriever class not found: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR checking BM25Retriever class: {type(e).__name__}: {e}")
        return None

    # Step 3: Check rank_bm25 dependency
    try:
        import rank_bm25
        print(f"✅ Step 3: rank_bm25 package available: {rank_bm25.__version__ if hasattr(rank_bm25, '__version__') else 'version unknown'}")
    except ImportError as e:
        print(f"❌ ERROR: rank_bm25 package not found: {e}")
        print("   Install with: pip install rank_bm25")
        return None
    except Exception as e:
        print(f"❌ ERROR with rank_bm25: {type(e).__name__}: {e}")
        return None

    # Step 4: Attempt to create BM25Retriever
    try:
        print("🔄 Step 4: Creating BM25Retriever...")
        
        # THE CRITICAL LINE WITH DETAILED ERROR HANDLING
        bm25_retriever = BM25Retriever.from_documents(documents, k=k)
        
        print("✅ SUCCESS: BM25Retriever created successfully!")
        print(f"   📊 Retriever type: {type(bm25_retriever)}")
        print(f"   📊 Retriever k value: {bm25_retriever.k}")
        
        return bm25_retriever
        
    except AttributeError as e:
        print(f"❌ AttributeError creating BM25Retriever: {e}")
        print("   🔍 Possible causes:")
        print("   - BM25Retriever API has changed")
        print("   - Wrong version of langchain-community")
        print("   - Method signature has changed")
        print(f"   - Available methods: {[m for m in dir(BM25Retriever) if 'from' in m.lower()]}")
        return None
        
    except ImportError as e:
        print(f"❌ ImportError during BM25Retriever creation: {e}")
        print("   🔍 This suggests missing dependencies during runtime")
        print("   - Check if rank_bm25 is properly installed")
        print("   - Try: pip install --upgrade rank_bm25")
        return None
        
    except TypeError as e:
        print(f"❌ TypeError creating BM25Retriever: {e}")
        print("   🔍 Possible causes:")
        print("   - Wrong parameter types passed to from_documents()")
        print("   - API signature has changed")
        print(f"   - Passed: documents={type(documents)}, k={type(k)}")
        return None
        
    except ValueError as e:
        print(f"❌ ValueError creating BM25Retriever: {e}")
        print("   🔍 Possible causes:")
        print("   - Invalid document content")
        print("   - Empty or malformed documents")
        print("   - Invalid k value")
        return None
        
    except ZeroDivisionError as e:
        print(f"❌ ZeroDivisionError creating BM25Retriever: {e}")
        print("   🔍 This usually indicates:")
        print("   - All documents are empty after preprocessing")
        print("   - BM25 algorithm encountering empty corpus")
        print("   - Text preprocessing removed all content")
        return None
        
    except RuntimeError as e:
        print(f"❌ RuntimeError creating BM25Retriever: {e}")
        print("   🔍 This suggests:")
        print("   - Internal BM25 algorithm failure")
        print("   - Memory issues with large document sets")
        print("   - System resource problems")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected error creating BM25Retriever: {type(e).__name__}: {e}")
        print("   🔍 Full error details:")
        import traceback
        traceback.print_exc()
        return None

def load_documents_from_json(cube_id: str, doc_type: str, base_dir: str) -> List[Document]:
    """Load documents from JSON file"""
    try:
        file_path = os.path.join(base_dir, cube_id, f"{cube_id}_{doc_type}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cube data doesn't exist")
            
        with open(file_path) as f:
            data = json.load(f)
            
        # Convert to Document objects
        documents = []
        for doc in data:
            content = f"Group Name:{doc['Group Name']}--Level Name:{doc['Level Name']}--Description:{doc['Description']}"
            documents.append(Document(page_content=content))
        return documents
            
    except Exception as e:
        logging.error(f"Error loading {doc_type} documents: {str(e)}")
        raise

def setup_logging():
    """Store errors in log folder datewise and token consumptions."""
    today = date.today()
    log_folder = './log'
  
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    
    logging.basicConfig(
        filename=f"{log_folder}/{today}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class LLMConfigure:
    """Class responsible for loading and configuring LLM and embedding models from a config file."""

    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None

    def load_config(self, config_path: str) -> Dict:
        """Loads the config from a JSON file."""
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                return config
        except FileNotFoundError as e:
            logging.error(f"Config file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the config file: {e}")
            raise

    def initialize_llm(self):
        """Initializes and returns the LLM model."""
        try:
            self.llm = AzureChatOpenAI(
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                model=self.config['llm']['model'],
                temperature=self.config['llm']['temperature'],
                api_version=self.config['llm']["OPENAI_API_VERSION"],
                azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                seed=self.config['llm']["seed"]
            )
            return self.llm
        except KeyError as e:
            logging.error(f"Missing LLM configuration in config file: {e}")
            raise

    def initialize_embedding(self):
        """Initializes and returns the Embedding model."""
        try:
            self.embedding = AzureOpenAIEmbeddings(
                deployment=self.config['embedding']['deployment'],
                azure_endpoint=self.config['llm']["AZURE_OPENAI_ENDPOINT"],
                openai_api_key=self.config['llm']['OPENAI_API_KEY'],
                show_progress_bar=self.config['embedding']['show_progress_bar'],
                disallowed_special=(),
                openai_api_type=self.config['llm']['OPENAI_API_TYPE']
            )
            return self.embedding
        except KeyError as e:
            logging.error(f"Missing embedding configuration in config file: {e}")
            raise

class DimensionMeasure:
    """Class responsible for extracting dimensions and measures from the natural language query."""

    def __init__(self, llm: str, embedding: str, vectorstore: str):
        self.llm = llm
        self.embedding = embedding
        self.vector_embedding = vectorstore

    def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extracts dimensions from the query."""
        try:
            with get_openai_callback() as dim_cb:
                query_dim = """ 
                As an SQL CUBE query expert, analyze the user's question and identify all relevant cube dimensions from the dimensions delimited by ####.
                
                <instructions>
                - Select relevant dimension group, level, description according to user query from dimensions list delimited by ####
                - format of one dimension: Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description> 
                - Include all dimensions relevant to the question in the response
                - Group Name and Level Name can never be same, extract corresponding group name for a selected level name according to the user query and vice versa.
                - If relevant dimensions group and level are not present in the dimensions list, please return "Not Found"
                - If the query mentions date, year, month ranges, include corresponding dimensions in the response  
                </instructions>
                
                Response format:
                Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description>

                Review:
                - ensure dimensions are only selected from dimensions list delimited by ####
                - Group Name and Level Name can never be same, extract corresponding group name, description for a selected level name according to the user query and vice versa.
                - Kindly ensure that the retrieved dimensions group name and level name is present otherwise return "Not found".

                User Query: {question}
                ####
                {context}
                ####
                """

                print(Fore.RED + '    Identifying Dimensions group name and level name......................\n')
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "dimensions", vector_db_path)
                
                # CRITICAL LINE - Initialize BM25 retriever with error handling
                print("\n" + "="*60)
                print("CREATING BM25 RETRIEVER FOR DIMENSIONS")
                print("="*60)
                bm25_retriever = safe_create_bm25_retriever(documents, k=10)
                print("="*60 + "\n")
                
                # Set up vector store directory
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                
                load_embedding_dim = Chroma(persist_directory=cube_dim, embedding_function=self.embedding)
                vector_retriever = load_embedding_dim.as_retriever(search_type="similarity", search_kwargs={"k": 20})

                # Create ensemble retriever or fallback to vector only
                if bm25_retriever is not None:
                    print("✅ Using BM25 + Vector ensemble retriever")
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=[0.5, 0.5]
                    )
                else:
                    print("⚠️  BM25 failed - falling back to vector retriever only")
                    ensemble_retriever = vector_retriever
                
                # Initialize and run QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=ensemble_retriever,
                    return_source_documents=True,
                    verbose=True,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template=query_dim,
                            input_variables=["query", "context"]
                        ),
                        "verbose": True
                    }
                )

                # Get results
                result = qa_chain({"query": query, "context": ensemble_retriever})
                dim = result['result']
                print(Fore.GREEN + '    Identified Group and level name :        ' + str(dim))
                logging.info(f"Extracted dimensions :\n {dim}")
                return dim

        except Exception as e:
            logging.error(f"Error extracting dimensions : {e}")
            raise

    def get_measures(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extracts measures from the query."""
        try:
            with get_openai_callback() as msr_cb:
                query_msr = """ 
                As an SQL CUBE query expert, analyze the user's question and identify all relevant cube measures from the measures delimited by ####.
                
                <instructions>
                - Select relevant measure group, level, description according to user query from measures list delimited by ####
                - format of one measure: Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description> 
                - Include all measures relevant to the question in the response
                - Group Name and Level Name can never be same, extract corresponding group name for a selected level name according to the user query and vice versa.
                - If relevant measures are not present in the measures list, please return "Not Found" 
                </instructions>
                
                <examples>
                Query: Remove City
                Response: Not Found

                Query: What is the rank by Balance amount for Mutual Fund
                Response: Group Name:Business Drivers--Level Name:Balance Amount Average--Description:Average balance amount of the customer/borrower
                Group Name:Fund Investment Details--Level Name:Mutual Fund Quantity--Description:Mutual Fund Quantity
                Group Name:Fund Investment Details--Level Name:Mutual Fund QoQ EOP--Description:Mutual Fund QoQ EOP
                Group Name:Fund Investment Details--Level Name:Mutual Fund Quantity QoQ--Description:Mutual Fund Quantity QoQ
                Group Name:Fund Investment Details--Level Name:Mutual Fund MoM EOP--Description:Mutual Fund MoM EOP
                </examples>

                Response format:
                Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description>

                Review:
                - ensure measures are only selected from measures list delimited by ####
                - Group Name and Level Name can never be same, extract corresponding group name, description for a selected level name according to the user query and vice versa.
                - Kindly ensure that the retrieved measures group name and level name is present otherwise return "Not found".

                User Query: {question}
                ####
                {context}
                ####
                """

                print(Fore.RED + '    Identifying Measure group name and level name......................\n')
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "measures", vector_db_path)
                
                # CRITICAL LINE - Initialize BM25 retriever with error handling
                print("\n" + "="*60)
                print("CREATING BM25 RETRIEVER FOR MEASURES")
                print("="*60)
                bm25_retriever = safe_create_bm25_retriever(documents, k=10)
                print("="*60 + "\n")

                cube_msr = os.path.join(vector_db_path, cube_id, "measures")
                load_embedding_msr = Chroma(persist_directory=cube_msr, embedding_function=self.embedding)
                vector_retriever = load_embedding_msr.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                
                # Create ensemble retriever or fallback to vector only
                if bm25_retriever is not None:
                    print("✅ Using BM25 + Vector ensemble retriever")
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=[0.5, 0.5]
                    )
                else:
                    print("⚠️  BM25 failed - falling back to vector retriever only")
                    ensemble_retriever = vector_retriever

                # Run QA chain with ensemble retriever
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    retriever=ensemble_retriever,
                    return_source_documents=True,
                    verbose=True,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template=query_msr,
                            input_variables=["query", "context"]
                        ),
                        "verbose": True
                    }
                )
                
                result = qa_chain({"query": query, "context": ensemble_retriever})
                msr = result['result']

                print(Fore.GREEN + '    Measures result :        ' + str(result)) 
                logging.info(f"Extracted measures :\n {msr}")  
                return msr
        
        except Exception as e:
            logging.error(f"Error Extracting Measure: {e}")
            raise

class FinalQueryGenerator(LLMConfigure):
    """Class responsible for generating the final OLAP query based on dimensions and measures."""
    
    def __init__(self, query, dimensions: None, measures: None, llm: None):
        super().__init__()
        self.query = query
        self.dimensions = dimensions
        self.measures = measures
        self.llm = llm
        
    def call_gpt(self, final_prompt: str):
        """Function responsible for generating final query"""
        API_KEY = self.config['llm']['OPENAI_API_KEY']
        headers = {
            "Content-Type": "application/json",
            "api-key": API_KEY,
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that writes accurate OLAP cube queries based on given query."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": final_prompt
                        }
                    ]
                }
            ],
            "temperature": self.config['llm']['temperature'],
            "top_p": self.config['llm']['top_p'],
            "max_tokens": self.config['llm']['max_tokens']
        }
        
        ENDPOINT = self.config['llm']['ENDPOINT']
        
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")
        
        output = response.json()
        token_details = output['usage']
        output = output["choices"][0]["message"]["content"]
        return output, token_details

    def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict, cube_name: str) -> str:
        try:
            if not dimensions or not measures:
                raise ValueError("Both dimensions and measures are required to generate a query.")
                
            final_prompt = f"""You are an expert in generating SQL Cube query. You will be provided dimensions delimited by $$$$ and measures delimited by &&&&.
            Your Goal is to generate a precise single line cube query for the user query delimited by ####.

            Instructions:            
            - Generate a single-line Cube query without line breaks
            - Include 'as' aliases for all level names in double quotes. alias are always level names.
            - Choose the most appropriate dimensions group names and level from dimensions delimited by $$$$ according to the query.
            - Choose the most appropriate measures group names and level from measures delimited by &&&& according to the query.
            - check the examples to learn about correct syntax, functions and filters which can be used according to the user query requirement.
            - User Query could be a follow up query in a conversation, you will also be provided previous query, dimensions, measures, cube query. Generate the final query including the contexts from conversation as appropriate.

            Formatting Rules:
            - Dimensions format: [Dimension Group Name].[Dimension Level Name] as "Dimension Level Name"
            - Measures format: [Measure Group Name].[Measure Level Name] as "Measure Level Name"
            - Conditions in WHERE clause must be properly formatted with operators
            - For multiple conditions, use "and" "or" operators
            - All string values in conditions must be in single quotes
            - All numeric values should not have leading zeros

            User Query: ####{query}####
            
            $$$$
            Dimensions: {dimensions}
            $$$$

            &&&&
            Measures: {measures}
            &&&&

            Generate a precise single-line Cube query that exactly matches these requirements:"""

            print(Fore.CYAN + '   Generating OLAP cube Query......................\n')
            result = self.llm.invoke(final_prompt)
            output = result.content
            pred_query = self.cleanup_gen_query(output)
            print(f"{pred_query}")
            
            logging.info(f"Generated OLAP Query: {pred_query}")
            return pred_query
        
        except Exception as e:
            logging.error(f"Error generating OLAP query: {e}")
            raise
    
    def cleanup_gen_query(self, pred_query):
        pred_query = pred_query.replace("```sql", "").replace("\n", "").replace("```", "")
        check = pred_query.replace("```", "")
        final_query = check.replace("sql", "")
