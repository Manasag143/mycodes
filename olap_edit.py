# CHANGE 1: Replace global shared objects in olap_details_generat.py

# REMOVE these lines:
# llm_config = LLMConfigure(config_file)
# llm = llm_config.initialize_llm()
# embedding = llm_config.initialize_embedding()

# ADD this processor manager class:
class ProcessorManager:
    def __init__(self):
        self.processors = {}
    
    def get_processor(self, user_id: str):
        try:
            if user_id not in self.processors:
                self.processors[user_id] = OLAPQueryProcessor(config_file)
            return self.processors[user_id]
        except Exception as e:
            # Clean up corrupted processor
            if user_id in self.processors:
                del self.processors[user_id]
            raise
    
    def cleanup_processor(self, user_id: str):
        if user_id in self.processors:
            try:
                del self.processors[user_id]
                import gc
                gc.collect()
            except Exception as e:
                logging.error(f"Error cleaning up processor: {e}")

# Initialize global processor manager
processor_manager = ProcessorManager()

# CHANGE 2: Update generate_cube_query endpoint

@app.post("/genai/cube_query_generation", response_model=QueryResponse)
async def generate_cube_query(request: CubeQueryRequest, user_details: str = Depends(verify_token)):
    user_id = f"user_{user_details}"
    
    try:
        cube_id = request.cube_id
        cube_dir = os.path.join(vector_db_path, cube_id)
        
        if not os.path.exists(cube_dir):
            return QueryResponse(
                message="failure",
                cube_query="Cube data doesn't exist",
                dimensions="",
                measures=""
            )
        
        # Get conversation context
        history_manager = History()
        if request.include_conv.lower() == "no":
            prev_conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": "",
                "dimensions": "",
                "measures": "",
                "response": "",
                "cube_id": cube_id,
                "cube_name": request.cube_name
            }
        else:
            prev_conversation = history_manager.retrieve(user_id, request.cube_id)
            if "cube_name" not in prev_conversation:
                prev_conversation["cube_name"] = request.cube_name

        # Get processor with error recovery
        processor = processor_manager.get_processor(user_id)
        
        query, final_query, processing_time, dimensions, measures = processor.process_query(
            request.user_query,
            request.cube_id,
            prev_conversation,
            request.cube_name,
            request.include_conv
        )
        
        # Update history
        response_data = {
            "query": request.user_query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query
        }
        history_manager.update(user_id, response_data, request.cube_id, request.cube_name)
        
        return QueryResponse(
            message="success",
            cube_query=final_query,
            dimensions=dimensions,
            measures=measures
        )
    
    except Exception as e:
        logging.error(f"Error in generate_cube_query: {str(e)}")
        
        # CRITICAL: Clean up on error
        processor_manager.cleanup_processor(user_id)
        
        return QueryResponse(
            message="failure", 
            cube_query=f"Processing error: {str(e)}",
            dimensions="",
            measures=""
        )

# CHANGE 3: Update process_cube_details to clean up processors after import

async def process_cube_details(cube_json_dim, cube_json_msr, cube_id: str) -> Dict:
    try:
        # ... existing code ...
        
        # At the end, add this cleanup:
        processor_manager.processors.clear()  # Force reload of all processors
        
        return {"message": "success"}
        
    except Exception as e:
        logging.error(f"Error processing cube details: {e}")
        return {"message": f"failure: {str(e)}"}

# CHANGE 4: Add error recovery to DimensionMeasure class in cube_query_v4.py

class DimensionMeasure:
    # ... existing code ...
    
    def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extracts dimensions from the query with error recovery."""
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                # ... existing dimension extraction code ...
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "dimensions", vector_db_path)
                
                # Initialize BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(documents, k=10)
                    
                # Set up vector store directory
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                persist_directory_dim = cube_dim
                
                # CRITICAL: Create new embedding instance each time
                embedding_instance = self.embedding
                if attempt > 0:
                    # Force new embedding on retry
                    import gc
                    gc.collect()
                
                load_embedding_dim = Chroma(persist_directory=persist_directory_dim, embedding_function=embedding_instance)
                vector_retriever = load_embedding_dim.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                
                # ... rest of existing code ...
                
                return dim
                
            except Exception as e:
                logging.error(f"Dimension extraction attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries:
                    raise
                
                # Clean up before retry
                import gc
                gc.collect()

    def get_measures(self, query: str, cube_id: str, prev_conv: dict) -> str:
        """Extracts measures from the query with error recovery."""
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                # ... existing measure extraction code ...
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "measures", vector_db_path)
                
                bm25_retriever = BM25Retriever.from_documents(documents, k=10)

                cube_msr = os.path.join(vector_db_path, cube_id)
                cube_msr = os.path.join(cube_msr, "measures")
                
                persist_directory_msr = cube_msr
                
                # CRITICAL: Create new embedding instance each time
                embedding_instance = self.embedding
                if attempt > 0:
                    import gc
                    gc.collect()
                
                load_embedding_msr = Chroma(persist_directory=persist_directory_msr, embedding_function=embedding_instance)
                vector_retriever = load_embedding_msr.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                
                # ... rest of existing code ...
                
                return msr
                
            except Exception as e:
                logging.error(f"Measure extraction attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries:
                    raise
                
                # Clean up before retry
                import gc
                gc.collect()

# CHANGE 5: Add state reset method to OLAPQueryProcessor

class OLAPQueryProcessor(LLMConfigure):
    # ... existing code ...
    
    def reset_state(self):
        """Reset processor state after error"""
        try:
            logging.info("Resetting processor state")
            
            # Reset components
            self.llm = None
            self.embedding = None
            self.dim_measure = None
            self.final = None
            self.conversational = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reinitialize
            self.llm = self.initialize_llm()
            self.embedding = self.initialize_embedding()
            self.dim_measure = DimensionMeasure(self.llm, self.embedding, self.load_json)
            self.final = FinalQueryGenerator(query="", dimensions=None, measures=None, llm=self.llm)
            self.conversational = ConversationalQueryGenerator(query="", dimensions=None, measures=None, llm=self.llm)
            
        except Exception as e:
            logging.error(f"Error resetting state: {e}")
            raise

    def process_query(self, query: str, cube_id: str, prev_conv: dict, cube_name: str, include_conv: str = "no") -> Tuple[str, str, float]:
        try:
            # ... existing process_query code ...
            
            return query, final_query, processing_time, dimensions, measures
            
        except Exception as e:
            logging.error(f"Error in query processing: {e}")
            
            # CRITICAL: Reset state on error
            try:
                self.reset_state()
            except Exception as reset_error:
                logging.error(f"State reset failed: {reset_error}")
            
            raise

# CHANGE 6: Enhanced error handling in load_documents_from_json

def load_documents_from_json(cube_id: str, doc_type: str, base_dir: str) -> List[Document]:
    """Load documents from JSON file with validation"""
    try:
        file_path = os.path.join(base_dir, cube_id, f"{cube_id}_{doc_type}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cube data doesn't exist at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate data
        if not isinstance(data, list) or not data:
            raise ValueError(f"Invalid or empty {doc_type} data")
            
        # Convert to Document objects
        documents = []
        for doc in data:
            if all(key in doc for key in ['Group Name', 'Level Name', 'Description']):
                content = f"Group Name:{doc['Group Name']}--Level Name:{doc['Level Name']}--Description:{doc['Description']}"
                documents.append(Document(page_content=content))
        
        if not documents:
            raise ValueError(f"No valid documents found in {doc_type} data")
            
        return documents
            
    except Exception as e:
        logging.error(f"Error loading {doc_type} documents: {str(e)}")
        raise

# CHANGE 7: Add cleanup to cube_details_import endpoint

@app.post("/genai/cube_details_import", response_model=CubeDetailsResponse)
async def import_cube_details(request: CubeDetailsRequest, user_details: str = Depends(verify_token)):
    try:
        user_id = f"user_{user_details}"
        
        result = await process_cube_details(
            request.cube_json_dim,
            request.cube_json_msr,
            request.cube_id
        )
        
        # CRITICAL: Clean up all processors after cube import
        processor_manager.processors.clear()
        
        return CubeDetailsResponse(message=result["message"])
        
    except Exception as e:
        logging.error(f"Error in import_cube_details: {e}")
        return CubeDetailsResponse(message=f"failure: {str(e)}")
