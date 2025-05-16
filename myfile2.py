import json
import time
import os
import re
import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
import logging
import datetime
from decimal import Decimal
from colorama import Fore, Style, init

# Initialize colorama for colored terminal output
init()

# Define the Llama URL
LLAMA_URL = "https://ue1-llm.crisil.local/llama3_3/70b/llm/"

class LLMGenerator:
    """
    A component to generate response from hosted LLM model using Text Generative Inference.
    """
    def __init__(self,
                 url: str = LLAMA_URL,
                 generation_kwargs: Optional[Dict[str, Any]] = None):
        
        self.url = url
        # handle generation kwargs setup
        self.generation_kwargs = generation_kwargs.copy() if generation_kwargs else {
            "max_new_tokens": 5048,
            "return_full_text": False,
            "temperature": 0.1
        }

    def run(self, prompt: str):
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        x = requests.post(self.url, verify=False, json=body)
        print(f"Request status: {x.status_code}")
        response = x.json()
        response_text = response[0]['generated_text']
        return response_text

class HumanMessage:
    """A simple class to mimic LangChain's HumanMessage structure"""
    def __init__(self, content):
        self.content = content

class LlamaMessage:
    """A simple class to mimic LangChain's message structure"""
    def __init__(self, content):
        self.content = content

class LlamaAdapter:
    """
    Adapter class to make LLMGenerator compatible with a standard interface.
    """
    def __init__(self):
        self.generator = LLMGenerator()
        
    def invoke(self, messages):
        """
        Invoke Llama with messages in a format that mimics a standard interface.
        
        Args:
            messages: List of message objects with a content attribute
            
        Returns:
            A LlamaMessage object with the generated content
        """
        # Extract the content from the last message
        if isinstance(messages, list):
            prompt = messages[-1].content
        else:
            prompt = messages.content
            
        # Format the prompt for Llama
        formatted_prompt = self._format_prompt(prompt)
        
        # Run the generator
        response = self.generator.run(formatted_prompt)
        
        # Return a message-like object
        return LlamaMessage(response)
    
    def _format_prompt(self, content):
        """Format the prompt for Llama"""
        sys_message = "You are an SQL expert specialized in converting natural language to accurate PostgreSQL queries."
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return formatted_prompt

class LlamaEmbeddings:
    """
    Class to generate embeddings using Llama
    """
    def __init__(self, url: str = LLAMA_URL):
        self.url = url
        self.generator = LLMGenerator(url=url)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Using a basic implementation that requests embeddings for each text.
        
        For production use, you should implement batching for efficiency.
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text string using Llama.
        
        In this implementation, we're using a simplified approach by:
        1. Creating a deterministic vector representation of the text
        
        Note: In production, you should use a proper embedding API
        """
        # Create a pseudo-embedding from hash of text
        import hashlib
        
        # Create a 384-dimensional embedding (typical dimension)
        vector = []
        for i in range(384):
            # Create a unique hash for each position in the vector
            hash_input = f"{text}_{i}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            # Normalize to range [-1, 1]
            value = (hash_val % 1000) / 500 - 1.0
            vector.append(value)
        
        # Normalize the vector
        norm = sum(v*v for v in vector) ** 0.5
        normalized = [v/norm for v in vector]
        
        return normalized

class RetrievalQA:
    """
    A simple implementation of RetrievalQA that mimics the LangChain interface
    """
    def __init__(self, llm, retriever, chain_type="stuff", verbose=False):
        self.llm = llm
        self.retriever = retriever
        self.chain_type = chain_type
        self.verbose = verbose
    
    @classmethod
    def from_chain_type(cls, llm, retriever, chain_type="stuff", verbose=False):
        """Class method to create a RetrievalQA instance"""
        return cls(llm, retriever, chain_type, verbose)
    
    def invoke(self, query_dict: Dict) -> Dict:
        """Invoke the retrieval QA chain"""
        # Extract the query
        query = query_dict.get("query", "")
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        # Extract content from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format prompt with context and query
        prompt = f"""
        Based on the following context:
        
        {context}
        
        Answer the following query:
        {query}
        """
        
        # Generate response using LLM
        response = self.llm.invoke(HumanMessage(content=prompt))
        
        # Return result
        return {"result": response.content}

class Document:
    """A simple document class to replace LangChain's Document"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleRetriever:
    """
    A simple retriever class that mimics the LangChain retriever interface
    """
    def __init__(self, vectordb, search_kwargs=None):
        self.vectordb = vectordb
        self.search_kwargs = search_kwargs or {"k": 5}
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query"""
        return self.vectordb.similarity_search(query, k=self.search_kwargs.get("k", 5))

class SimpleVectorDB:
    """
    A simple in-memory vector database for demonstration purposes.
    """
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.documents = []
        self.embeddings = []
    
    def count(self):
        """Return the number of documents in the database"""
        return len(self.documents)
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add texts to the vector database"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        # Generate embeddings
        new_embeddings = self.embedding_function.embed_documents(texts)
        
        # Store documents and embeddings
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, new_embeddings)):
            self.documents.append({"id": len(self.documents) + i, "text": text, "metadata": metadata})
            self.embeddings.append(embedding)
            
        return [doc["id"] for doc in self.documents[-len(texts):]]
    
    def similarity_search(self, query: str, k: int = 5):
        """Search for documents similar to the query"""
        # Generate query embedding
        query_embedding = self.embedding_function.embed_query(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Calculate cosine similarity
            similarity = sum(a*b for a, b in zip(query_embedding, doc_embedding))
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return top k results
        top_k = sorted_similarities[:k]
        results = []
        for doc_id, score in top_k:
            # Create a Document object for compatibility
            doc = self.documents[doc_id]
            results.append(Document(
                page_content=doc["text"],
                metadata={"score": score, **doc["metadata"]}
            ))
            
        return results
    
    def as_retriever(self, search_kwargs=None):
        """
        Create a retriever interface that mimics the LangChain retriever interface
        """
        return SimpleRetriever(self, search_kwargs)
    
    def save(self, directory: str):
        """Save the vector database to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save documents
        with open(os.path.join(directory, "documents.json"), "w") as f:
            json.dump(self.documents, f)
        
        # Save embeddings
        np.save(os.path.join(directory, "embeddings.npy"), np.array(self.embeddings))
        
        logging.info(f"Saved vector database with {len(self.documents)} documents to {directory}")
    
    @classmethod
    def load(cls, directory: str, embedding_function):
        """Load a vector database from disk"""
        db = cls(embedding_function)
        
        if os.path.exists(os.path.join(directory, "documents.json")) and \
           os.path.exists(os.path.join(directory, "embeddings.npy")):
            # Load documents
            with open(os.path.join(directory, "documents.json"), "r") as f:
                db.documents = json.load(f)
            
            # Load embeddings
            db.embeddings = np.load(os.path.join(directory, "embeddings.npy")).tolist()
            
            logging.info(f"Loaded vector database with {len(db.documents)} documents from {directory}")
        else:
            logging.warning(f"Could not find vector database files in {directory}")
            
        return db

class Chroma:
    """
    A simple wrapper class to make our SimpleVectorDB look like LangChain's Chroma class
    """
    def __init__(self, persist_directory=None, embedding_function=None):
        if persist_directory and os.path.exists(persist_directory):
            self._collection = SimpleVectorDB.load(persist_directory, embedding_function)
        else:
            self._collection = SimpleVectorDB(embedding_function)
        
    @classmethod
    def from_documents(cls, documents, embedding, persist_directory=None):
        instance = cls(embedding_function=embedding)
        
        # Extract text and metadata from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add texts to the collection
        instance._collection.add_texts(texts, metadatas)
        
        # Save to disk if directory is provided
        if persist_directory:
            instance._collection.save(persist_directory)
            
        return instance
    
    def as_retriever(self, search_kwargs=None):
        """Create a retriever from this vector store"""
        return self._collection.as_retriever(search_kwargs)
    
    def similarity_search(self, query, k=5):
        """Search for documents similar to the query"""
        return self._collection.similarity_search(query, k=k)

class OpenAICallback:
    """
    A simple callback to track token usage.
    """
    def __init__(self):
        self.total_tokens = 0
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def get_openai_callback():
    """
    Get an OpenAI callback to track token usage.
    This is a simplified version for compatibility.
    """
    return OpenAICallback()

class DatabaseConnector:
    """Class to connect to the database and retrieve schema information."""
    def __init__(self, config_path: str = "config1.json"):
        self.config = self.load_config(config_path)
        self.connection = None
        
    def load_config(self, config_path: str) -> Dict:
        """Loads the config from a JSON file."""
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                logging.info("Config file loaded successfully.")
                return config
        except FileNotFoundError as e:
            logging.error(f"Config file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the config file: {e}")
            raise
    
    def connect(self):
        """Connects to the database."""
        try:
            self.connection = psycopg2.connect(
                host=self.config['database']['host'],
                database=self.config['database']['database'],
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                port=self.config['database']['port']
            )
            logging.info("Connected to database successfully")
            return self.connection
        except Exception as e:
            logging.error(f"Error connecting to database: {e}")
            raise
    
    def execute_query(self, query: str):
        """Execute a SQL query and return results."""
        try:
            if not self.connection or self.connection.closed:
                self.connect()
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
            
            return results
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            raise
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Gets the schema (column names and types) for a specified table."""
        try:
            if not self.connection or self.connection.closed:
                self.connect()
                
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            query = """
            SELECT 
                column_name, 
                data_type, 
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale
            FROM 
                information_schema.columns 
            WHERE 
                LOWER(table_name) = LOWER(%s)
            ORDER BY 
                ordinal_position;
            """
            cursor.execute(query, (table_name.lower(),))  # Ensure table_name is lowercase
            schema = cursor.fetchall()
            cursor.close()
            
            if not schema:
                logging.warning(f"No schema found for table {table_name}")
            else:
                logging.info(f"Retrieved schema for table {table_name}: {len(schema)} columns")
            
            return schema
        except Exception as e:
            logging.error(f"Error getting schema for table {table_name}: {e}")
            raise
    
    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            logging.info("Database connection closed")

class LLMConfigure:
    """
    Class responsible for loading and configuring LLM and embedding models.
    Modified to use only Llama models.
    """
    def __init__(self, config_path: str = "config1.json"):
        self.config = self.load_config(config_path)
        self.llm = None
        self.embedding = None
        
    def load_config(self, config_path: str) -> Dict:
        """Loads the config from a JSON file."""
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                logging.info("Config file loaded successfully.")
                return config
        except FileNotFoundError as e:
            logging.error(f"Config file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing the config file: {e}")
            raise
            
    def initialize_llm(self):
        """Initializes and returns the Llama LLM model."""
        try:
            # Use Llama adapter
            llama_url = self.config.get('llama', {}).get('url', LLAMA_URL)
            
            # Create the adapter
            self.llm = LlamaAdapter()
            
            logging.info(f"LLM model initialized: {self.llm}")
            return self.llm
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            raise
            
    def initialize_embedding(self):
        """Initializes and returns the Embedding model using Llama."""
        try:
            # Use Llama for embeddings
            llama_url = self.config.get('llama', {}).get('url', LLAMA_URL)
            self.embedding = LlamaEmbeddings(url=llama_url)
            
            logging.info(f"Embedding model initialized using Llama")
            return self.embedding
        except Exception as e:
            logging.error(f"Error initializing embedding model: {e}")
            raise

class ResultSummarizer:
    """
    Class responsible for generating summaries of SQL query results.
    Enhanced to handle both LIKE pattern matching and Levenshtein distance explanations.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter
    
    def generate_summary(self, original_query: str, results: List[Dict], sql_query: str) -> str:
        """
        Generate a concise summary of the query results.
        
        Args:
            original_query: The original natural language query from the user
            results: The results returned from the SQL query execution
            sql_query: The executed SQL query
            
        Returns:
            A string containing a concise summary of the results
        """
        try:
            # If there are no results, return a simple statement
            if not results:
                return "No matching records were found in the database for your query."
                
            # Convert results to a pandas DataFrame for easier summarization
            # Convert Decimal and non-serializable types to standard Python types
            sanitized_results = []
            for row in results:
                sanitized_row = {}
                for key, value in row.items():
                    # Convert Decimal to float
                    if isinstance(value, Decimal):
                        sanitized_row[key] = float(value)
                    # Convert date/datetime to string
                    elif isinstance(value, (datetime.date, datetime.datetime)):
                        sanitized_row[key] = value.isoformat()
                    else:
                        sanitized_row[key] = value
                sanitized_results.append(sanitized_row)
                
            df = pd.DataFrame(sanitized_results)
            
            # Get basic statistics
            num_rows = len(df)
            num_columns = len(df.columns)
            column_names = list(df.columns)
            
            # Create a sample of the results (first 5 rows) as a formatted string
            sample_rows = df.head(5).to_dict('records')
            sample_str = json.dumps(sample_rows, indent=2)
            
            summary_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a data analyst who specializes in creating extremely concise summaries of database query results.
Given the information about a database query and its results, provide a direct answer to the original question
in AT MOST 2 LINES. Do not include any additional insights or explanations.

Be direct and factual. Focus only on providing a direct answer to the original question. Do not use bullet points.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Original question: "{original_query}"

SQL query executed:
{sql_query}

Query returned {num_rows} rows with {num_columns} columns: {', '.join(column_names)}

Here's a sample of the results (first 5 rows or less):
{sample_str}

Please provide a MAXIMUM 2-LINE direct answer to the original question without any additional insights or details.
Focus only on the main facts that directly answer the question.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            # Generate the summary using Llama
            generator = LLMGenerator()
            summary = generator.run(summary_prompt)
            
            # Ensure the summary is really no more than 2 lines
            summary_lines = summary.strip().split('\n')
            if len(summary_lines) > 2:
                summary = ' '.join(summary_lines[:2])
            
            return summary.strip()
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Unable to generate summary due to an error: {str(e)}"

class ConversationHistoryManager:
    """
    Manages conversation history for each user, storing history in a JSON file.
    Maintains the last 5 interactions per user.
    """
    def __init__(self, history_file: str = "conversation_history.json"):
        """Initialize the conversation history manager with a file path."""
        self.history_file = history_file
        self.history = self._load_history()
        
    def _load_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load conversation history from file or create empty history."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Error decoding history file {self.history_file}. Creating new history.")
                return {}
            except Exception as e:
                logging.error(f"Error loading history file: {e}")
                return {}
        else:
            return {}
            
    def _save_history(self):
        """Save conversation history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
            logging.info(f"Saved conversation history to {self.history_file}")
        except Exception as e:
            logging.error(f"Error saving conversation history: {e}")
            
    def add_interaction(self, username: str, query: str, sql_query: str, results: Any, summary: str):
        """
        Add a new interaction to the user's conversation history.
        Maintains only the last 5 interactions.
        
        Args:
            username: User identifier
            query: Natural language query
            sql_query: Generated SQL query
            results: Query results (will be converted to list for JSON serialization)
            summary: Generated summary
        """
        # Initialize user history if not exists
        if username not in self.history:
            self.history[username] = []
            
        # Create interaction record with only necessary fields
        interaction = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "sql_query": sql_query
        }
        
        # Add to history
        self.history[username].append(interaction)
        
        # Keep only the last 5 interactions
        if len(self.history[username]) > 5:
            self.history[username] = self.history[username][-5:]
            
        # Save updated history
        self._save_history()
        
    def get_user_history(self, username: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific user."""
        return self.history.get(username, [])
        
    def get_last_interaction(self, username: str) -> Optional[Dict[str, Any]]:
        """Get the last interaction for a specific user."""
        user_history = self.get_user_history(username)
        if user_history:
            return user_history[-1]
        return None
        
    def get_formatted_history(self, username: str, max_entries: int = 5) -> str:
        """
        Get the user's history formatted as a string for inclusion in prompts.
        
        Args:
            username: User identifier
            max_entries: Maximum number of history entries to include
            
        Returns:
            Formatted history string for inclusion in prompts
        """
        user_history = self.get_user_history(username)
        
        if not user_history:
            return "No previous conversation history."
            
        # Get the most recent entries, limited by max_entries
        recent_history = user_history[-max_entries:]
        
        # Format the history
        formatted_history = []
        for i, interaction in enumerate(recent_history):
            entry = f"Interaction {i+1}:\n"
            entry += f"User: {interaction['query']}\n"
            entry += f"Generated SQL: {interaction['sql_query']}\n"
            formatted_history.append(entry)
            
        return "\n".join(formatted_history)
    
class FollowUpDetector:
    """
    Detects if a query is a follow-up to a previous query using LLM.
    """
    def __init__(self, llm_generator):
        """
        Initialize with an LLMGenerator instance.
        
        Args:
            llm_generator: An instance of LLMGenerator for making predictions
        """
        self.llm_generator = llm_generator
        
    def is_follow_up(self, current_query: str, previous_query: str = None) -> dict:
        """
        Determine if the current query is a follow-up to the previous query.
        
        Args:
            current_query: The current natural language query
            previous_query: The previous query (if any)
            
        Returns:
            Dictionary with 'is_follow_up' boolean and 'reasoning' string
        """
        # If no previous query, definitely not a follow-up
        if not previous_query:
            return {
                "is_follow_up": False,
                "reasoning": "No previous query exists to follow up on."
            }
            
        # Create prompt for LLM to determine if this is a follow-up
        prompt = self._create_follow_up_prompt(current_query, previous_query)
        
        # Generate response using LLM
        response = self.llm_generator.run(prompt)
        
        # Parse the response to extract yes/no and reasoning
        is_follow_up, reasoning = self._parse_follow_up_response(response)
        
        return {
            "is_follow_up": is_follow_up,
            "reasoning": reasoning
        }
        
    def _create_follow_up_prompt(self, current_query: str, previous_query: str) -> str:
        """Create prompt for follow-up detection."""
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert in analyzing conversation context and determining if a query is a follow-up to a previous query.
A follow-up query typically:
1. References the previous query or its results implicitly or explicitly
2. Contains pronouns (it, they, them, these, etc.) that refer to entities mentioned in the previous query
3. Uses ellipsis (omitting words that were present in the previous query)
4. Asks for more details about the same topic
5. Uses contextual words like "also", "additionally", "what about", "and", etc.
6. If a query is unclear or lacks context (e.g., "what is industry_id"), assume it's a follow-up query.

Respond with only "YES" or "NO" followed by a short explanation of your reasoning.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Previous query: "{previous_query}"
Current query: "{current_query}"

Is the current query a follow-up to the previous query?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
        
    def _parse_follow_up_response(self, response: str) -> tuple:
        """
        Parse the LLM response to extract yes/no decision and reasoning.
        Args:
            response: LLM response string
        Returns:
            Tuple of (is_follow_up: bool, reasoning: str)
        """
        response = response.strip()
        
        # Extract yes/no from the beginning of the response
        is_follow_up = False
        reasoning = response
        
        if response.upper().startswith("YES"):
            is_follow_up = True
            # Extract reasoning (everything after YES)
            if len(response) > 3:
                reasoning = response[3:].strip()
        elif response.upper().startswith("NO"):
            is_follow_up = False
            # Extract reasoning (everything after NO)
            if len(response) > 2:
                reasoning = response[2:].strip()
                
        return is_follow_up, reasoning  

class NLQProcessor:
    def __init__(self, query: str, config_path: str = "config1.json"):
        """Initialize NLQ Processor."""
        self.query = query.strip().lower()  # Convert query to lowercase immediately
        self.original_query = query.strip()  # Keep original query for reference
        self.config_path = config_path
        self.llm_config = LLMConfigure(config_path)
        self.llm = self.llm_config.initialize_llm()
        self.embedding = self.llm_config.initialize_embedding()
        self.db_connector = DatabaseConnector(config_path)
        self.result_summarizer = ResultSummarizer(self.llm)
        
        # Pre-defined table information
        self.tables = """Table Name:billing_type_ref--Description:Whether it is billable or non-billable company. It is master table for billing type.
Table Name:business_area_ref--Description:Business area name for the business area of the company. It is master table for business area.
Table Name:city_ref--Description:City name of the company where it is located. It is master table for city.
Table Name:code_request_approval_ref--Description:List of the request for the companies for which request is raised for code creation. It is master table for code request approval.
Table Name:companies_merged_mst--Description:It is master for the company merger list.
Table Name:companies_merged_stg--Description:It is master for the company merger list.
Table Name:company_address_mst--Description:It consists of company address. It is transaction table.
Table Name:company_business_area_mst--Description:Business area name for the business area of the company. It is transaction table for company business area.
Table Name:company_mst--Description:It consist of company_mst
Table Name:company_name_change_mst--Description:It is master for company name change.
Table Name:company_risk_status_mst--Description:It is transaction tables.
Table Name:company_stg--Description:List of all the companies.
Table Name:company_tan_pan_roc_mst--Description:List of unique identifier of the companies (i.e., TAN, PAN). It is master table for TAN PAN of the company.
Table Name:company_type_ref--Description:Type of the company (i.e., Pvt/Public). It is master table for company type.
Table Name:contact_change_request--Description:Reason for the contact change request. It is master table for contact change request.
Table Name:contact_mst--Description:Contact master for the company.
Table Name:contact_remove_reason_ref--Description:Added comments, if any for the deactivation of the contact master. It is master table for contact remove reason.
Table Name:contacts_merged_mst--Description:It is master for the contact merger list.
Table Name:country_ref--Description:Country name of the company where it is located. It is master table for country.
Table Name:department_ref--Description:Departments of the contact person. It is master table for department.
Table Name:designation_ref--Description:Designation name of the contact person. It is master table for designation.
Table Name:district_ref--Description:District name of the company where it is located. It is master table for district.
Table Name:gst_approval_status_ref--Description:GST approval status of the company. It is master table for gst approval status.
Table Name:gst_contact_request_type_ref--Description:It is an audit field for gst contact request type. It is master table for gst contact request type.
Table Name:gst_exception_request_ref--Description:GST exception list like company exception from gst, sez contact, etc. It is master table for gst exception request.
Table Name:gst_party_type_ref--Description:Type of the GST name. It is master table for GST party type.
Table Name:industry_ref--Description:Industry to which company belongs to. It is master table for industry.
Table Name:karza_company_mst--Description:List of the gst number fetched from the Karza portal for the companies.
Table Name:name_change_reason_ref--Description:Reason for the name change of the company. It is master table for name change reason.
Table Name:pincode_ref--Description:Pincode of the company. It is master table for pincode.
Table Name:region_ref--Description:Region of the company where it is located. It is master table for region.
Table Name:reject_reason_ref--Description:List of the reasons for which code creation of the company is rejected. It is master table for reject reason.
Table Name:risk_ref--Description:It is risk status of the company. It is master table for risk status.
Table Name:salutation_ref--Description:Mr/Mrs/ of the person. It is master table for salutation.
Table Name:sector_ref--Description:Sector to which company belongs to. It is master table for sector.
Table Name:state_ref--Description:State name of the company where it is located. It is master table for state.
Table Name:Taxpayer_Type_ref--Description:Type of the taxpayer for gst fillings. It is master table for taxpayer type.
"""
        # Initialize tables and vector database
        self.table_data = self.parse_tables()
        
    def parse_tables(self) -> Dict[str, str]:
        """Parse the tables string into a dictionary mapping table names to descriptions."""
        table_data = {}
        for line in self.tables.strip().split('\n'):
            if line.startswith('Table Name:'):
                parts = line.split('--Description:')
                if len(parts) == 2:
                    table_name = parts[0].replace('Table Name:', '').strip().lower()  # Convert table names to lowercase
                    description = parts[1].strip()
                    table_data[table_name] = description
        
        logging.info(f"Parsed {len(table_data)} tables")
        return table_data
        
    def format_table_info_for_prompt(self, tables: List[Dict]) -> str:
        """Format table information for inclusion in the prompt."""
        formatted_info = []
        
        for table in tables:
            table_name = table["table_name"]
            description = table["description"]
            schema = table["schema"]
            
            table_info = f"Table: {table_name}\nDescription: {description}\nColumns:"
            
            for col in schema:
                col_name = col.get("column_name", "")
                data_type = col.get("data_type", "")
                is_nullable = col.get("is_nullable", "")
                
                nullable_text = "NOT NULL" if is_nullable.lower() == 'no' else "NULL"
                table_info += f"\n  - {col_name} ({data_type}, {nullable_text})"
            
            formatted_info.append(table_info)
        
        return "\n\n".join(formatted_info)
        
    def generate_sql_query(self, relevant_tables: List[Dict]) -> str:
        """Generate SQL query from natural language query using relevant tables."""
        try:
            # Format table information for the prompt
            table_info = self.format_table_info_for_prompt(relevant_tables)
            
            # Updated fuzzy matching hint with the new LIKE patterns
            fuzzy_matching_hint = """
            11. IMPORTANT: When comparing company names or other text fields where the user might 
                have made a spelling mistake, use both pattern matching with specific variations 
                and Levenshtein distance for maximum matching capabilities.
                
                Use the following pattern for company names, BUT ONLY WHEN THE QUERY IS ABOUT SPECIFIC COMPANIES:
                
                ```sql
                -- For a company name like "Ranbaxy Drugs Limited"
                WHERE 
                    (cm.company_name = 'Ranbaxy Drugs Limited')
                    OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%')
                    OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited')
                    OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%')
                    OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited')
                    OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited')
                    OR levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9
                ORDER BY 
                    CASE 
                        WHEN cm.company_name = 'Ranbaxy Drugs Limited' THEN 0        -- Exact match first
                        WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%') THEN 1  -- Starts with exact name
                        WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited') THEN 2  -- Ends with exact name
                        WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%') THEN 3  -- Contains exact name
                        WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited') THEN 4  -- First word partial
                        WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited') THEN 5  -- Middle word partial
                        WHEN levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9 THEN 6  -- Close by Levenshtein
                        ELSE 7  -- Other matches
                    END,
                    levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) -- Sort by distance
                LIMIT 100
                ```
                
                DO NOT include these WHERE clauses for aggregate operations (COUNT, SUM, AVG, etc.) when the query is about all companies or statistics across all companies.
            """
            
            # Add instruction to always include company_name in the select clause
            company_name_instruction = """
            12. VERY IMPORTANT: Always include cm.company_name in the SELECT clause of your query, 
                but ONLY when appropriate for the query. For aggregate queries that apply to all companies
                (like "count total number of companies", "average billing amount across all companies", etc.),
                do NOT include individual company names unless specifically requested.
            """
            
            # Add specific instructions for aggregate operations
            aggregate_instruction = """
            13. For mathematical operations like SUM, AVG, COUNT, MIN, MAX:
                - Do NOT include WHERE clauses with company name matching unless a specific company is mentioned
                - For queries like "How many companies are there", "What is the average revenue", etc., use aggregate
                functions WITHOUT company name filtering
                - Only apply filters relevant to the question being asked
            """
            
            # Construct the prompt for SQL generation
            sql_prompt = f"""
            You are a SQL expert specialized in converting natural language queries to valid SQL queries.
            
            Given the following database schema information:
            
            {table_info}
            
            Generate a single, accurate SQL query for PostgreSQL that answers this question:
            "{self.original_query}"
            
            Follow these rules:
            1. Use only the tables and columns provided in the schema above
            2. Write standard SQL query for PostgreSQL
            3. Include appropriate JOINs(LEFT,RIGHT) between tables where necessary
            4. Add proper WHERE conditions to filter results as requested
            5. Use table aliases to avoid ambiguity
            6. Include aggregations (GROUP BY, HAVING) if needed
            7. Return only the SQL query without explanations or comments
            8. Never use unnecessary where clause 
            9. Give priority to the most relevant table
            10. Please go through examples for better result
            {fuzzy_matching_hint}
            {company_name_instruction}
            {aggregate_instruction}
            
            <examples>

            user query : Is Ranbaxy Drugs Limited billing or non-billing?
            expected response :select cm.company_name,btr.billing_type
            from company_mst cm left join billing_type_ref btr on cm.billing_type_id=btr.id
            where (cm.company_name='Ranbaxy Drugs Limited'
            OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited')
            OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited')
            OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited')
            OR levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9)
            ORDER BY 
                CASE 
                    WHEN cm.company_name = 'Ranbaxy Drugs Limited' THEN 0
                    WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%') THEN 1
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited') THEN 2
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%') THEN 3
                    WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited') THEN 4
                    WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited') THEN 5
                    WHEN levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9 THEN 6
                    ELSE 7
                END,
                levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited'))
            LIMIT 100

            user query: Show me contact details for Cipla Limited
            expected response: select cm.company_name, c.contact_name, c.email_id, c.contact_no
            from company_mst cm 
            left join contact_mst c on cm.id = c.company_id
            where (cm.company_name = 'Cipla Limited'
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla% Limited')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Ltd%')
            OR levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited')) <= 9)
            ORDER BY 
                CASE 
                    WHEN cm.company_name = 'Cipla Limited' THEN 0
                    WHEN LOWER(cm.company_name) LIKE LOWER('Cipla Limited%') THEN 1
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Cipla Limited') THEN 2
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Cipla Limited%') THEN 3
                    WHEN LOWER(cm.company_name) LIKE LOWER('Cipla% Limited') THEN 4
                    WHEN LOWER(cm.company_name) LIKE LOWER('Cipla Ltd%') THEN 5
                    WHEN levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited')) <= 9 THEN 6
                    ELSE 7
                END,
                levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited'))
            LIMIT 100
            
            user query: How many companies are there in the database?
            expected response: select count(*) as total_companies from company_mst
            
            user query: What is the average number of contacts per company?
            expected response: select avg(contact_count) as average_contacts_per_company
            from (
            select cm.id, count(c.id) as contact_count
            from company_mst cm
            left join contact_mst c on cm.id = c.company_id
            group by cm.id
            ) as company_contacts
            
            user query: Count companies by business area
            expected response: select ba.business_area_name, count(cm.id) as company_count
            from company_mst cm
            left join company_business_area_mst cba on cm.id = cba.company_id
            left join business_area_ref ba on cba.business_area_id = ba.id
            group by ba.business_area_name
            order by company_count desc

            </examples>

            SQL Query:
            """
            
            print(Fore.BLUE + "Generating SQL query...")
            
            # Use the LLM to generate the SQL query
            with get_openai_callback() as cb:
                message = HumanMessage(content=sql_prompt)
                response = self.llm.invoke([message])
                sql_query = response.content.strip()
                
                # Clean up the response - remove markdown code blocks if present
                if sql_query.startswith("```sql"):
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                elif sql_query.startswith("```"):
                    sql_query = sql_query.replace("```", "").strip()
                
                # Log token usage
                logging.info(f"Token usage for SQL generation: {cb.total_tokens}")
                logging.info(f"Generated SQL query: {sql_query}")
                print(Fore.GREEN + "SQL query generated successfully")
            
            return sql_query
            
        except Exception as e:
            logging.error(f"Error generating SQL query: {e}")
            raise


    def close(self):
        """Clean up resources."""
        self.db_connector.close()
    
    def identify_relevant_tables(self, top_k: int = 5) -> List[Dict]:
        """
        Identify the most relevant tables for the given query using Llama directly.
        
        Uses prompt-based approach instead of vector similarity search.
        """
        try:
            print(Fore.BLUE + f"Finding top {top_k} relevant tables for query: '{self.original_query}'")
            
            # Create a detailed prompt with all table information for Llama
            tables_info = "\n".join([f"Table: {name} - {desc}" for name, desc in self.table_data.items()])
            
            llama_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a database expert. You need to identify the most relevant tables from a database to answer a natural language query.
Given the tables and their descriptions below, identify exactly {top_k} most relevant tables needed to answer the query.

Database tables:
{tables_info}

Only respond with the table names as a comma-separated list. Do not include any explanations or additional text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
What tables would I need to answer this query: "{self.original_query}"
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            print(Fore.YELLOW + "Using Llama to identify relevant tables...")
            
            # Get response from Llama
            generator = LLMGenerator()
            response = generator.run(llama_prompt)
            
            # Parse the response to get table names
            tables_text = response.strip()
            logging.info(f"Raw tables list from Llama: {tables_text}")
            
            # Clean up and parse comma-separated table names
            table_names = [t.strip().lower() for t in tables_text.split(',')]
            
            # Filter to known tables
            known_table_names = [t for t in table_names if t in self.table_data]
            
            print(Fore.GREEN + f"Found {len(known_table_names)} relevant tables")
            
            # If we have fewer than top_k tables, fall back to the most popular tables
            if len(known_table_names) < top_k:
                print(Fore.YELLOW + f"Only found {len(known_table_names)} tables, adding fallback tables...")
                # Add the most relevant main tables that might be used frequently
                fallback_tables = ["company_mst", "contact_mst", "company_address_mst", "company_type_ref", "company_business_area_mst"]
                
                for table in fallback_tables:
                    if len(known_table_names) >= top_k:
                        break
                    if table not in known_table_names and table in self.table_data:
                        known_table_names.append(table)
            
            # Ensure we return exactly top_k tables if we have that many
            if len(known_table_names) > top_k:
                known_table_names = known_table_names[:top_k]
                
            # Get schema information for each table
            relevant_tables = []
            print(Fore.BLUE + "Retrieving schema information for tables:")
            
            for table_name in known_table_names:
                print(Fore.CYAN + f"  - {table_name}")
                try:
                    # Get schema from database
                    schema = self.db_connector.get_table_schema(table_name)
                    
                    # Add to relevant tables list
                    table_info = {
                        "table_name": table_name,
                        "description": self.table_data.get(table_name, ""),
                        "schema": schema
                    }
                    relevant_tables.append(table_info)
                except Exception as e:
                    logging.error(f"Error retrieving schema for {table_name}: {e}")
                    # Include table even if schema retrieval fails
                    table_info = {
                        "table_name": table_name,
                        "description": self.table_data.get(table_name, ""),
                        "schema": []
                    }
                    relevant_tables.append(table_info)
            
            return relevant_tables
                
        except Exception as e:
            logging.error(f"Error identifying relevant tables: {e}")
            raise

    def execute_sql_query(self, sql_query: str):
        """Execute the SQL query and return results."""
        try:
            print(Fore.BLUE + "Executing SQL query on database...")
            start_time = time.time()
            
            # Execute query using the DatabaseConnector's execute_query method
            results = self.db_connector.execute_query(sql_query)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            logging.info(f"Query executed in {execution_time:.2f} seconds, returned {len(results)} rows")
            print(Fore.GREEN + f"Query executed successfully in {execution_time:.2f} seconds")
            print(Fore.GREEN + f"Retrieved {len(results)} rows from database")
            
            return results
            
        except Exception as e:
            logging.error(f"Error executing SQL query: {e}")
            raise
            
    def summarize_results(self, results: List[Dict], sql_query: str) -> str:
        """
        Generate a human-readable summary of the query results.
        
        Args:
            results: The results returned from the SQL query
            sql_query: The executed SQL query
            
        Returns:
            A string containing a summarized interpretation of the results
        """
        try:
            print(Fore.BLUE + "Generating summary of query results...")
            
            # Use the result summarizer to generate a summary
            summary = self.result_summarizer.generate_summary(
                original_query=self.original_query,
                results=results,
                sql_query=sql_query
            )
            
            print(Fore.GREEN + "Summary generated successfully")
            return summary
            
        except Exception as e:
            logging.error(f"Error summarizing results: {e}")
            return f"Unable to generate summary: {str(e)}"

    def process_query(self) -> Dict:
        """
        Process the natural language query to SQL with the following steps:
        1. Identify relevant tables
        2. Generate SQL query with combined LIKE and Levenshtein matching
        3. Execute the query
        4. Summarize the results with appropriate matching explanations
        5. Return complete information
        
        Returns:
            Dict containing the original query, identified tables, SQL query, results, and summary
        """
        try:
            print(Fore.CYAN + "=" * 80)
            print(Fore.CYAN + f"Processing query: {self.original_query}")
            print(Fore.CYAN + "=" * 80)
            
            # Step 1: Identify relevant tables (using Llama direct prompting)
            relevant_tables = self.identify_relevant_tables(top_k=5)
            
            # Step 2: Generate SQL query using the tables and their schema (with combined matching)
            sql_query = self.generate_sql_query(relevant_tables)
            
            # Step 3: Execute the SQL query
            print(Fore.YELLOW + "SQL Query:")
            print(sql_query)
            results = self.execute_sql_query(sql_query)
            
            # Step 4: Generate a summary of the results (with appropriate matching explanation)
            summary = self.summarize_results(results, sql_query)
            
            # Step 5: Return the complete results with summary and matching information
            has_like = "like" in sql_query.lower()
            has_levenshtein = "levenshtein" in sql_query.lower()
            
            return {
                "original_query": self.original_query,
                "relevant_tables": [t["table_name"] for t in relevant_tables],
                "sql_query": sql_query,
                "results": results,
                "summary": summary,
                "has_like_pattern": has_like,
                "has_levenshtein": has_levenshtein,
                "has_fuzzy_matching": has_like or has_levenshtein
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "original_query": self.original_query,
                "error": str(e)
            }
        
class ConversationalNLQProcessor(NLQProcessor):
    """
    Enhanced NLQProcessor with conversation history and follow-up detection.
    Inherits from the base NLQProcessor and adds conversational capabilities.
    """
    
    def __init__(self, query: str, username: str, config_path: str = "config1.json"):
        """
        Initialize the conversational processor.
        
        Args:
            query: The current natural language query
            username: The user's identifier for conversation history
            config_path: Path to configuration file
        """
        # Call the parent constructor
        super().__init__(query=query, config_path=config_path)
        
        # Initialize conversation history manager
        self.history_manager = ConversationHistoryManager()
        
        # Store username for history tracking
        self.username = username
        
        # Initialize follow-up detector
        self.follow_up_detector = FollowUpDetector(LLMGenerator())
        
        # Get last interaction from history
        self.last_interaction = self.history_manager.get_last_interaction(username)
        
        # Determine if this is a follow-up question
        self.follow_up_info = self._detect_follow_up()
        
    def _detect_follow_up(self) -> dict:
        """
        Detect if the current query is a follow-up to the previous query.
        
        Returns:
            Dictionary with follow-up detection information
        """
        previous_query = None
        if self.last_interaction:
            previous_query = self.last_interaction.get("query")
            
        if previous_query:
            return self.follow_up_detector.is_follow_up(self.original_query, previous_query)
        else:
            return {
                "is_follow_up": False,
                "reasoning": "No previous query in conversation history."
            }
            
    def generate_sql_query(self, relevant_tables: List[Dict]) -> str:
        """
        Generate SQL query from natural language query using relevant tables.
        Enhanced to handle follow-up queries by incorporating conversation context.
        
        Args:
            relevant_tables: List of relevant tables with their schemas
            
        Returns:
            Generated SQL query
        """
        try:
            # Format table information for the prompt
            table_info = self.format_table_info_for_prompt(relevant_tables)
            
            # Enhanced fuzzy matching hint - same as original
            fuzzy_matching_hint = """
            11. IMPORTANT: When comparing company names or other text fields where the user might 
                have made a spelling mistake, use both pattern matching with specific variations 
                and Levenshtein distance for maximum matching capabilities.
                
                Use the following pattern for company names, BUT ONLY WHEN THE QUERY IS ABOUT SPECIFIC COMPANIES:
                
                ```sql
                -- For a company name like "Ranbaxy Drugs Limited"
                WHERE 
                    (cm.company_name = 'Ranbaxy Drugs Limited')
                    OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%')
                    OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited')
                    OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%')
                    OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited')
                    OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited')
                    OR levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9
                ORDER BY 
                    CASE 
                        WHEN cm.company_name = 'Ranbaxy Drugs Limited' THEN 0        -- Exact match first
                        WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%') THEN 1  -- Starts with exact name
                        WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited') THEN 2  -- Ends with exact name
                        WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%') THEN 3  -- Contains exact name
                        WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited') THEN 4  -- First word partial
                        WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited') THEN 5  -- Middle word partial
                        WHEN levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9 THEN 6  -- Close by Levenshtein
                        ELSE 7  -- Other matches
                    END,
                    levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) -- Sort by distance
                LIMIT 100
                ```
                
                DO NOT include these WHERE clauses for aggregate operations (COUNT, SUM, AVG, etc.) when the query is about all companies or statistics across all companies.
            """
            
            # Additional instructions - same as original
            company_name_instruction = """
            12. VERY IMPORTANT: Always include cm.company_name in the SELECT clause of your query, 
                but ONLY when appropriate for the query. For aggregate queries that apply to all companies
                (like "count total number of companies", "average billing amount across all companies", etc.),
                do NOT include individual company names unless specifically requested.
            """
            
            aggregate_instruction = """
            13. For mathematical operations like SUM, AVG, COUNT, MIN, MAX:
                - Do NOT include WHERE clauses with company name matching unless a specific company is mentioned
                - For queries like "How many companies are there", "What is the average revenue", etc., use aggregate
                functions WITHOUT company name filtering
                - Only apply filters relevant to the question being asked
            """
            
            # NEW: Add conversation context if this is a follow-up question
            conversation_context = ""
            if self.follow_up_info.get("is_follow_up", False) and self.last_interaction:
                # Include previous query and its SQL for context
                conversation_context = f"""
                14. IMPORTANT: The current query is a follow-up to a previous query. Consider the context:
                
                Previous question: "{self.last_interaction.get('query')}"
                Previous SQL query: {self.last_interaction.get('sql_query')}
                Previous results summary: {self.last_interaction.get('summary')}
                
                When generating the SQL for the current query:
                - Maintain any relevant filters from the previous query
                - Resolve pronouns and implicit references to entities mentioned in the previous query
                - Ensure the new query builds upon the context established in the previous interaction
                """
            
            # Construct the prompt for SQL generation
            sql_prompt = f"""
            You are a SQL expert specialized in converting natural language queries to valid SQL queries.
            
            Given the following database schema information:
            
            {table_info}
            
            Generate a single, accurate SQL query for PostgreSQL that answers this question:
            "{self.original_query}"
            
            Follow these rules:
            1. Use only the tables and columns provided in the schema above
            2. Write standard SQL query for PostgreSQL
            3. Include appropriate JOINs(LEFT,RIGHT) between tables where necessary
            4. Add proper WHERE conditions to filter results as requested
            5. Use table aliases to avoid ambiguity
            6. Include aggregations (GROUP BY, HAVING) if needed
            7. Return only the SQL query without explanations or comments
            8. Never use unnecessary where clause 
            9. Give priority to the most relevant table
            10. Please go through examples for better result
            {fuzzy_matching_hint}
            {company_name_instruction}
            {aggregate_instruction}
            {conversation_context}
            
            <examples>

            user query : Is Ranbaxy Drugs Limited billing or non-billing?
            expected response :select cm.company_name,btr.billing_type
            from company_mst cm left join billing_type_ref btr on cm.billing_type_id=btr.id
            where (cm.company_name='Ranbaxy Drugs Limited'
            OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited')
            OR LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited')
            OR LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited')
            OR levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9)
            ORDER BY 
                CASE 
                    WHEN cm.company_name = 'Ranbaxy Drugs Limited' THEN 0
                    WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs Limited%') THEN 1
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited') THEN 2
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Ranbaxy Drugs Limited%') THEN 3
                    WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy% Drugs Limited') THEN 4
                    WHEN LOWER(cm.company_name) LIKE LOWER('Ranbaxy Drugs% Limited') THEN 5
                    WHEN levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited')) <= 9 THEN 6
                    ELSE 7
                END,
                levenshtein(LOWER(cm.company_name), LOWER('Ranbaxy Drugs Limited'))
            LIMIT 100

            user query: Show me contact details for Cipla Limited
            expected response: select cm.company_name, c.contact_name, c.email_id, c.contact_no
            from company_mst cm 
            left join contact_mst c on cm.id = c.company_id
            where (cm.company_name = 'Cipla Limited'
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla% Limited')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Ltd%')
            OR levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited')) <= 9)
            ORDER BY 
                CASE 
                    WHEN cm.company_name = 'Cipla Limited' THEN 0
                    WHEN LOWER(cm.company_name) LIKE LOWER('Cipla Limited%') THEN 1
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Cipla Limited') THEN 2
                    WHEN LOWER(cm.company_name) LIKE LOWER('%Cipla Limited%') THEN 3
                    WHEN LOWER(cm.company_name) LIKE LOWER('Cipla% Limited') THEN 4
                    WHEN LOWER(cm.company_name) LIKE LOWER('Cipla Ltd%') THEN 5
                    WHEN levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited')) <= 9 THEN 6
                    ELSE 7
                END,
                levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited'))
            LIMIT 100
            
            user query: How many companies are there in the database?
            expected response: select count(*) as total_companies from company_mst
            
            user query: What is the average number of contacts per company?
            expected response: select avg(contact_count) as average_contacts_per_company
            from (
            select cm.id, count(c.id) as contact_count
            from company_mst cm
            left join contact_mst c on cm.id = c.company_id
            group by cm.id
            ) as company_contacts
            
            user query: Count companies by business area
            expected response: select ba.business_area_name, count(cm.id) as company_count
            from company_mst cm
            left join company_business_area_mst cba on cm.id = cba.company_id
            left join business_area_ref ba on cba.business_area_id = ba.id
            group by ba.business_area_name
            order by company_count desc
            
            -- Example of a follow-up query
            user previous query: Show me contact details for Cipla Limited
            previous SQL: select cm.company_name, c.contact_name, c.email_id, c.contact_no
            from company_mst cm 
            left join contact_mst c on cm.id = c.company_id
            where (cm.company_name = 'Cipla Limited'
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla% Limited')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Ltd%')
            OR levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited')) <= 9)
            ORDER BY CASE WHEN cm.company_name = 'Cipla Limited' THEN 0 ELSE 1 END
            LIMIT 100
            
            user query: What is their address?
            expected response: select cm.company_name, ca.address_line1, ca.address_line2, c.city_name, s.state_name, p.pincode
            from company_mst cm 
            left join company_address_mst ca on cm.id = ca.company_id
            left join city_ref c on ca.city_id = c.id
            left join state_ref s on ca.state_id = s.id
            left join pincode_ref p on ca.pincode_id = p.id
            where (cm.company_name = 'Cipla Limited'
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited')
            OR LOWER(cm.company_name) LIKE LOWER('%Cipla Limited%')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla% Limited')
            OR LOWER(cm.company_name) LIKE LOWER('Cipla Ltd%')
            OR levenshtein(LOWER(cm.company_name), LOWER('Cipla Limited')) <= 9)
            ORDER BY CASE WHEN cm.company_name = 'Cipla Limited' THEN 0 ELSE 1 END
            LIMIT 100

            </examples>

            SQL Query:
            """
            
            print(Fore.BLUE + "Generating SQL query...")
            if self.follow_up_info.get("is_follow_up", False):
                print(Fore.YELLOW + "Processing as follow-up question")
                print(Fore.YELLOW + f"Reasoning: {self.follow_up_info.get('reasoning', '')}")
            
            # Use the LLM to generate the SQL query
            with get_openai_callback() as cb:
                message = HumanMessage(content=sql_prompt)
                response = self.llm.invoke([message])
                sql_query = response.content.strip()
                
                # Clean up the response - remove markdown code blocks if present
                if sql_query.startswith("```sql"):
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                elif sql_query.startswith("```"):
                    sql_query = sql_query.replace("```", "").strip()
                
                # Log token usage
                logging.info(f"Token usage for SQL generation: {cb.total_tokens}")
                logging.info(f"Generated SQL query: {sql_query}")
                print(Fore.GREEN + "SQL query generated successfully")
            
            return sql_query
            
        except Exception as e:
            logging.error(f"Error generating SQL query: {e}")
            raise
            
    def process_query(self) -> Dict:
        """
        Process the natural language query to SQL, enhancing the base method
        to include conversation history and follow-up handling.
        
        Returns:
            Dictionary containing query processing results and follow-up detection information
        """
        try:
            print(Fore.CYAN + "=" * 80)
            print(Fore.CYAN + f"Processing query for user '{self.username}': '{self.original_query}'")
            if self.follow_up_info.get("is_follow_up", False):
                print(Fore.CYAN + "Detected as follow-up question")
            print(Fore.CYAN + "=" * 80)
            
            # Step 1: Identify relevant tables
            relevant_tables = self.identify_relevant_tables(top_k=5)
            
            # Step 2: Generate SQL query with conversation context
            sql_query = self.generate_sql_query(relevant_tables)
            
            # Step 3: Execute the SQL query
            print(Fore.YELLOW + "SQL Query:")
            print(sql_query)
            results = self.execute_sql_query(sql_query)
            
            # Step 4: Generate a summary of the results
            summary = self.summarize_results(results, sql_query)
            
            # Step 5: Add interaction to conversation history
            self.history_manager.add_interaction(
                username=self.username,
                query=self.original_query,
                sql_query=sql_query,
                results=results,
                summary=summary
            )
            
            # Step 6: Return the complete results with follow-up information
            has_like = "like" in sql_query.lower()
            has_levenshtein = "levenshtein" in sql_query.lower()
            
            return {
                "original_query": self.original_query,
                "relevant_tables": [t["table_name"] for t in relevant_tables],
                "sql_query": sql_query,
                "results": results,
                "summary": summary,
                "has_like_pattern": has_like,
                "has_levenshtein": has_levenshtein,
                "has_fuzzy_matching": has_like or has_levenshtein,
                "is_follow_up": self.follow_up_info.get("is_follow_up", False),
                "follow_up_reasoning": self.follow_up_info.get("reasoning", "")
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "original_query": self.original_query,
                "error": str(e),
                "is_follow_up": self.follow_up_info.get("is_follow_up", False) 
            }

def main():
    config_path = "config1.json"
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("nlq_processor.log"),
            logging.StreamHandler()
        ]
    )
    
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "Conversational Natural Language to SQL Query Processor")
    print(Fore.CYAN + "=" * 80)
    
    # Get username for conversation tracking
    print(Fore.WHITE + "\nPlease enter your username:")
    username = input("> ")
    
    # Load conversation history manager to check previous conversations
    history_manager = ConversationHistoryManager()
    user_history = history_manager.get_user_history(username)
    
    if user_history:
        print(Fore.GREEN + f"Welcome back, {username}! Found {len(user_history)} previous interactions.")
    else:
        print(Fore.GREEN + f"Welcome, {username}! This is your first conversation.")
      
    while True:
        print(Fore.WHITE + "\nEnter your Query (or 'exit' to quit):")
        user_query = input("> ")

        if user_query.lower() in ['exit', 'quit', 'q']:
            print(Fore.GREEN + "Exiting. Goodbye!")
            break

        if not user_query.strip():
            continue
            
        try:
            # Use the conversational processor instead of the basic one
            processor = ConversationalNLQProcessor(user_query, username, config_path)
            result = processor.process_query()
            
            # Display follow-up detection info
            if result.get("is_follow_up", False):
                print(Fore.MAGENTA + "\nDetected as a follow-up question")
                print(Fore.MAGENTA + f"Reasoning: {result.get('follow_up_reasoning', '')}")
            
            # Display results
            if "error" in result:
                print(Fore.RED + f"Error: {result['error']}")
            else:
                # Display the identified tables
                print(Fore.BLUE + "\nIdentified Tables:")
                for table in result["relevant_tables"]:
                    print(Fore.CYAN + f"  - {table}")
                
                # Display the SQL query
                print(Fore.BLUE + "\nGenerated SQL Query:")
                print(Fore.YELLOW + result["sql_query"])
             
                # Display the results
                if result["results"]:
                    print(Fore.BLUE + f"\nResults: ({len(result['results'])} rows)")
                    
                    # Convert to DataFrame for nicer display
                    df = pd.DataFrame(result["results"])
                    
                    # Display up to 100 rows
                    num_rows = min(100, len(df))
                    
                    # Method 1: Use pandas head with increased row limit
                    print(df.head(num_rows))
                    
                    if len(df) > num_rows:
                        print(Fore.CYAN + f"... and {len(df) - num_rows} more rows")
                    
                    # Display the summary
                    print(Fore.BLUE + "\nSummary:")
                    print(Fore.GREEN + "=" * 80)
                    print(Fore.WHITE + result["summary"])
                    print(Fore.GREEN + "=" * 80)
                    
                else:
                    print(Fore.YELLOW + "No results returned from the query.")
                    if "summary" in result:
                        print(Fore.BLUE + "\nSummary:")
                        print(Fore.WHITE + result["summary"])
            processor.close()
            
        except Exception as e:
            print(Fore.RED + f"Error: {str(e)}")
            logging.error(f"Unhandled error: {str(e)}", exc_info=True)
            
if __name__ == "__main__":
    main()
