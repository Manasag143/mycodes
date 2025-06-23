# Add these logging statements to your existing code at the specified locations:

# In cube_query_v4.py - Add to DimensionMeasure.get_dimensions method after line with "Identifying Dimensions"
def get_dimensions(self, query: str, cube_id: str, prev_conv: dict) -> str:
    """Extracts dimensions from the query."""
    logging.info(f"PIPELINE_START - Starting dimension extraction for query: '{query}' with cube_id: {cube_id}")
    max_retries = 2
    
    for attempt in range(max_retries + 1):
        try:
            logging.info(f"DIMENSION_EXTRACTION - Attempt {attempt + 1}/{max_retries + 1} for cube_id: {cube_id}")
            
            with get_openai_callback() as dim_cb:
                # ... existing code ...
                
                print(Fore.RED + '    Identifying Dimensions group name and level name......................\n')
                logging.info(f"DIMENSION_PIPELINE - Loading documents from JSON for cube_id: {cube_id}")
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "dimensions", vector_db_path)
                logging.info(f"DIMENSION_DOCS - Successfully loaded {len(documents)} dimension documents")
                
                # Initialize BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(documents, k=10)
                logging.info(f"DIMENSION_BM25 - BM25 retriever initialized successfully")
                
                # Set up vector store directory
                cube_dir = os.path.join(vector_db_path, cube_id)
                cube_dim = os.path.join(cube_dir, "dimensions")
                persist_directory_dim = cube_dim
                
                if attempt > 0:
                    logging.info(f"DIMENSION_RETRY - Forcing garbage collection on retry attempt {attempt}")
                    gc.collect()

                load_embedding_dim = Chroma(persist_directory=persist_directory_dim, embedding_function=self.embedding)
                vector_retriever = load_embedding_dim.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                logging.info(f"DIMENSION_VECTOR - Vector retriever initialized successfully")
                
                # Create ensemble retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.5, 0.5]
                )
                logging.info(f"DIMENSION_ENSEMBLE - Ensemble retriever created with BM25 and vector retrievers")
                
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
                
                logging.info(f"DIMENSION_LLM - Invoking LLM for dimension extraction")
                # Get results
                result = qa_chain.invoke({"query": query, "context": ensemble_retriever})
                dim = result['result']
                
                logging.info(f"DIMENSION_SUCCESS - Dimensions extracted successfully: {dim}")
                logging.info(f"DIMENSION_TOKENS - Token usage: {dim_cb}")
                print(Fore.GREEN + '    Identified Group and level name :        ' + str(dim))
                return dim

        except Exception as e:
            logging.error(f"DIMENSION_ERROR - Error extracting dimensions on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                logging.error(f"DIMENSION_FAILED - All dimension extraction attempts failed for cube_id: {cube_id}")
                raise

# In cube_query_v4.py - Add to DimensionMeasure.get_measures method
def get_measures(self, query: str, cube_id: str, prev_conv: dict) -> str:
    """Extracts measures from the query."""
    logging.info(f"MEASURE_START - Starting measure extraction for query: '{query}' with cube_id: {cube_id}")
    max_retries = 2
    
    for attempt in range(max_retries + 1):
        try:
            logging.info(f"MEASURE_EXTRACTION - Attempt {attempt + 1}/{max_retries + 1} for cube_id: {cube_id}")
            
            with get_openai_callback() as msr_cb:
                # ... existing code ...
                
                print(Fore.RED + '    Identifying Measure group name and level name......................\n')
                logging.info(f"MEASURE_PIPELINE - Loading documents from JSON for cube_id: {cube_id}")
                
                # Load documents from JSON
                documents = load_documents_from_json(cube_id, "measures", vector_db_path)
                logging.info(f"MEASURE_DOCS - Successfully loaded {len(documents)} measure documents")
                
                bm25_retriever = BM25Retriever.from_documents(documents, k=10)
                logging.info(f"MEASURE_BM25 - BM25 retriever initialized successfully")

                cube_msr = os.path.join(vector_db_path, cube_id)
                cube_msr = os.path.join(cube_msr, "measures")
                persist_directory_msr = cube_msr

                if attempt > 0:
                    logging.info(f"MEASURE_RETRY - Forcing garbage collection on retry attempt {attempt}")
                    gc.collect()

                load_embedding_msr = Chroma(persist_directory=persist_directory_msr, embedding_function=self.embedding)
                vector_retriever = load_embedding_msr.as_retriever(search_type="similarity", search_kwargs={"k": 20})
                logging.info(f"MEASURE_VECTOR - Vector retriever initialized successfully")
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.5, 0.5]
                )
                logging.info(f"MEASURE_ENSEMBLE - Ensemble retriever created with BM25 and vector retrievers")

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
                
                logging.info(f"MEASURE_LLM - Invoking LLM for measure extraction")
                result = qa_chain.invoke({"query": query, "context": ensemble_retriever})
                msr = result['result']

                logging.info(f"MEASURE_SUCCESS - Measures extracted successfully: {msr}")
                logging.info(f"MEASURE_TOKENS - Token usage: {msr_cb}")
                print(Fore.GREEN + '    Measures result :        ' + str(result)) 
                return msr
            
        except Exception as e:
            logging.error(f"MEASURE_ERROR - Error extracting measures on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                logging.error(f"MEASURE_FAILED - All measure extraction attempts failed for cube_id: {cube_id}")
                raise

# In cube_query_v4.py - Add to FinalQueryGenerator.generate_query method
def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict, cube_name: str) -> str:
    logging.info(f"QUERY_GENERATION_START - Starting OLAP query generation for query: '{query}'")
    
    try:
        if not dimensions or not measures:
            logging.error(f"QUERY_GENERATION_ERROR - Missing dimensions or measures. Dimensions: {bool(dimensions)}, Measures: {bool(measures)}")
            raise ValueError("Both dimensions and measures are required to generate a query.")

        logging.info(f"QUERY_GENERATION_INPUT - Dimensions: {dimensions}")
        logging.info(f"QUERY_GENERATION_INPUT - Measures: {measures}")
        
        dynamic_functions = self.functions_manager.build_dynamic_functions_section(query)
        logging.info(f"QUERY_GENERATION_FUNCTIONS - Dynamic functions loaded for query analysis")

        # ... existing final_prompt code ...
        
        print(Fore.CYAN + '   Generating OLAP cube Query......................\n')
        logging.info(f"QUERY_GENERATION_LLM - Invoking LLM for final query generation")
        
        result = self.llm.invoke(final_prompt)
        output = result.content
        token_details = result.response_metadata['token_usage']
        
        logging.info(f"QUERY_GENERATION_LLM_RESPONSE - LLM responded successfully")
        logging.info(f"QUERY_GENERATION_TOKENS - Token usage: {token_details}")
        
        pred_query = self.cleanup_gen_query(output)
        
        logging.info(f"QUERY_GENERATION_SUCCESS - Final OLAP query generated: {pred_query}")
        logging.info(f"QUERY_GENERATION_OPTIMIZATION - Function categories used: {selected_categories}")
        
        return pred_query
    
    except Exception as e:
        logging.error(f"QUERY_GENERATION_FAILED - Error generating OLAP query: {e}")
        raise

# In olap_details_generat.py - Add to verify_token function
async def verify_token(authorization: str = Header(None)):
    logging.info(f"AUTH_START - Token verification initiated")
    
    if not authorization:
        logging.error(f"AUTH_FAILED - No authorization token provided")
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    try:
        token = authorization.split(" ")[1]
        logging.info(f"AUTH_TOKEN - Token extracted from authorization header")
        
        # Updated JWT decode for newer PyJWT versions
        try:
            payload = jwt.decode(token, options={"verify_signature": False, "verify_exp": False})
        except TypeError:
            # For older PyJWT versions
            payload = jwt.decode(token, verify=False)
            
        user_details = payload.get("preferred_username")
        if not user_details:
            logging.error(f"AUTH_FAILED - No user details found in token")
            raise ValueError("No user details in token")
        
        logging.info(f"AUTH_SUCCESS - Token verified successfully for user: {user_details}")
        return user_details
    except Exception as e:
        logging.error(f"AUTH_ERROR - Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

# In olap_details_generat.py - Add to generate_cube_query endpoint
@app.post("/genai/cube_query_generation", response_model=QueryResponse)
async def generate_cube_query(request: CubeQueryRequest, user_details: str = Depends(verify_token)):
    user_id = f"user_{user_details}"
    
    logging.info(f"API_HIT - cube_query_generation endpoint hit by user: {user_details}")
    logging.info(f"API_REQUEST - Query: '{request.user_query}', Cube_ID: {request.cube_id}, Include_Conv: {request.include_conv}")
    
    try:
        cube_id = request.cube_id
        cube_dir = os.path.join(vector_db_path, cube_id)
        
        if not os.path.exists(cube_dir):
            logging.error(f"API_ERROR - Cube directory not found for cube_id: {cube_id}")
            return QueryResponse(
                message="failure",
                cube_query="Cube data doesn't exist",
                dimensions="",
                measures=""
            )
        
        logging.info(f"API_CUBE_VALIDATION - Cube directory validated for cube_id: {cube_id}")
        
        # Get conversation context
        history_manager = History()
        logging.info(f"API_HISTORY - History manager initialized")
        
        if request.include_conv.lower() == "no":
            logging.info(f"API_CONTEXT - Creating new conversation context (include_conv=no)")
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
            logging.info(f"API_CONTEXT - Retrieving existing conversation context (include_conv=yes)")
            prev_conversation = history_manager.retrieve(user_id, request.cube_id)
            if "cube_name" not in prev_conversation:
                prev_conversation["cube_name"] = request.cube_name

        # Get processor with error recovery
        logging.info(f"API_PROCESSOR - Getting processor for user_id: {user_id}")
        processor = processor_manager.get_processor(user_id)
        logging.info(f"API_PROCESSOR - Processor obtained successfully")
        
        logging.info(f"API_PROCESSING - Starting query processing pipeline")
        query, final_query, processing_time, dimensions, measures = processor.process_query(
            request.user_query,
            request.cube_id,
            prev_conversation,
            request.cube_name,
            request.include_conv
        )
        logging.info(f"API_PROCESSING - Query processing completed successfully in {processing_time:.2f} seconds")
        
        # Update history
        response_data = {
            "query": request.user_query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query
        }
        logging.info(f"API_HISTORY_UPDATE - Updating conversation history")
        history_manager.update(user_id, response_data, request.cube_id, request.cube_name)
        logging.info(f"API_HISTORY_UPDATE - History updated successfully")
  
        logging.info(f"API_SUCCESS - Query generation completed successfully")
        logging.info(f"API_RESPONSE - Final query: {final_query}")

        return QueryResponse(
            message="success",
            cube_query=final_query,
            dimensions=dimensions,
            measures=measures
        )
    
    except Exception as e:
        logging.error(f"API_ERROR - Error in generate_cube_query: {str(e)}")
        logging.error(f"API_CLEANUP - Cleaning up processor for user_id: {user_id}")
        
        # CRITICAL: Clean up on error
        processor_manager.cleanup_processor(user_id)
        
        return QueryResponse(
            message="failure", 
            cube_query=f"Processing error: {str(e)}",
            dimensions="",
            measures=""
        )

# In olap_details_generat.py - Add to import_cube_details endpoint
@app.post("/genai/cube_details_import", response_model=CubeDetailsResponse)
async def import_cube_details(request: CubeDetailsRequest, user_details: str = Depends(verify_token)):
    logging.info(f"API_HIT - cube_details_import endpoint hit by user: {user_details}")
    logging.info(f"API_REQUEST - Cube_ID: {request.cube_id}, Dimensions: {len(request.cube_json_dim)}, Measures: {len(request.cube_json_msr)}")
    
    try:
        user_id = f"user_{user_details}"
        logging.info(f"API_USER - Processing import for user_id: {user_id}")
        
        logging.info(f"API_PROCESSING - Starting cube details processing")
        result = await process_cube_details(
            request.cube_json_dim,
            request.cube_json_msr,
            request.cube_id
        )
        logging.info(f"API_PROCESSING - Cube details processing completed with result: {result['message']}")
        
        processor_manager.processors.clear()
        logging.info(f"API_CLEANUP - All processors cleared after import")
        
        logging.info(f"API_SUCCESS - Cube import completed successfully")
        
        return CubeDetailsResponse(message=result["message"])
    except HTTPException as he:
        logging.error(f"API_HTTP_ERROR - HTTP exception in import_cube_details: {he}")
        return CubeDetailsResponse(message=f"failure:{he}")
    except Exception as e:
        logging.error(f"API_ERROR - Error in import_cube_details: {e}")
        return CubeDetailsResponse(message=f"failure:{e}")

# In olap_details_generat.py - Add to process_cube_details function
async def process_cube_details(cube_json_dim, cube_json_msr, cube_id: str) -> Dict:
    logging.info(f"CUBE_IMPORT_START - Starting cube details processing for cube_id: {cube_id}")
    
    try:
        # Define paths
        cube_dir = os.path.join(vector_db_path, cube_id)
        logging.info(f"CUBE_IMPORT_PATH - Target cube directory: {cube_dir}")
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"CUBE_IMPORT_TEMP - Created temporary directory: {temp_dir}")
            
            # Create subdirectories
            temp_cube_dir = os.path.join(temp_dir, cube_id)
            temp_dim_dir = os.path.join(temp_cube_dir, "dimensions")
            temp_msr_dir = os.path.join(temp_cube_dir, "measures")
            
            os.makedirs(temp_cube_dir, exist_ok=True)
            os.makedirs(temp_dim_dir, exist_ok=True)
            os.makedirs(temp_msr_dir, exist_ok=True)
            logging.info(f"CUBE_IMPORT_DIRS - Created temporary subdirectories")
            
            # Save dimension and measure JSON documents to temp location
            temp_dim_file = os.path.join(temp_cube_dir, f"{cube_id}_dimensions.json")
            with open(temp_dim_file, 'w', encoding='utf-8') as f:
                json.dump(cube_json_dim, f, indent=2)
            logging.info(f"CUBE_IMPORT_DIM - Saved {len(cube_json_dim)} dimensions to temp file")
                
            temp_msr_file = os.path.join(temp_cube_dir, f"{cube_id}_measures.json")
            with open(temp_msr_file, 'w', encoding='utf-8') as f:
                json.dump(cube_json_msr, f, indent=2)
            logging.info(f"CUBE_IMPORT_MSR - Saved {len(cube_json_msr)} measures to temp file")
            
            # Format measure documents
            measure_docs = format_measure_documents(cube_json_msr)
            logging.info(f"CUBE_IMPORT_FORMAT - Formatted {len(measure_docs)} measure documents")
            
            # Save BM25 documents
            temp_bm25_file = os.path.join(temp_cube_dir, f"{cube_id}_measures.pkl")
            with open(temp_bm25_file, 'wb') as f:
                pickle.dump(measure_docs, f)
            logging.info(f"CUBE_IMPORT_BM25 - Saved BM25 documents to pickle file")
            
            # Process documents for vector stores
            cube_str_dim = [f"Group Name:{d['Group Name']}--Level Name:{d['Level Name']}--Description:{d['Description']}" for d in cube_json_dim]
            text_list_dim = [Document(i) for i in cube_str_dim]
            logging.info(f"CUBE_IMPORT_DIM_DOCS - Created {len(text_list_dim)} dimension documents")
            
            cube_str_msr = [f"Group Name:{d['Group Name']}--Level Name:{d['Level Name']}--Description:{d['Description']}" for d in cube_json_msr]
            text_list_msr = [Document(i) for i in cube_str_msr]
            logging.info(f"CUBE_IMPORT_MSR_DOCS - Created {len(text_list_msr)} measure documents")
            
            # Create vector stores in temporary location
            logging.info(f"CUBE_IMPORT_VECTOR - Creating dimension vector store")
            vectordb_dim = Chroma.from_documents(
                documents=text_list_dim,
                embedding=embeddings,
                persist_directory=temp_dim_dir
            )
            
            logging.info(f"CUBE_IMPORT_VECTOR - Creating measure vector store")
            vectordb_msr = Chroma.from_documents(
                documents=text_list_msr,
                embedding=embeddings,
                persist_directory=temp_msr_dir
            )
            
            # Ensure the vector stores are properly persisted
            vectordb_dim.persist()
            vectordb_msr.persist()
            logging.info(f"CUBE_IMPORT_PERSIST - Vector stores persisted successfully")
            
            # Now, delete the existing cube directory (if exists)
            if os.path.exists(cube_dir):
                logging.info(f"CUBE_IMPORT_CLEANUP - Deleting existing cube directory: {cube_dir}")
                shutil.rmtree(cube_dir, ignore_errors=True)
            
            # Create the target directory
            os.makedirs(os.path.join(vector_db_path, cube_id), exist_ok=True)
            logging.info(f"CUBE_IMPORT_TARGET - Created target directory")
            
            # Copy the successfully created content from temp to actual location
            shutil.copytree(temp_cube_dir, cube_dir, dirs_exist_ok=True)
            logging.info(f"CUBE_IMPORT_COPY - Copied processed data to final location")
            
        # Verify the transfer was successful
        if os.path.exists(os.path.join(cube_dir, f"{cube_id}_dimensions.json")) and \
           os.path.exists(os.path.join(cube_dir, f"{cube_id}_measures.json")):
            logging.info(f"CUBE_IMPORT_SUCCESS - Successfully processed cube details for cube_id: {cube_id}")
            processor_manager.processors.clear()  # Force reload of all processors
            logging.info(f"CUBE_IMPORT_PROCESSORS - Cleared all processors for reload")
        
        return {"message": "success"}
        
    except Exception as e:
        logging.error(f"CUBE_IMPORT_ERROR - Error processing cube details: {e}")
        return {"message": f"failure: {str(e)}"}

# In olap_details_generat.py - Add to OLAPQueryProcessor.process_query method
def process_query(self, query: str, cube_id: str, prev_conv: dict, cube_name: str, include_conv: str = "no") -> Tuple[str, str, float]:
    logging.info(f"OLAP_PROCESS_START - Starting OLAP query processing")
    logging.info(f"OLAP_PROCESS_INPUT - Query: '{query}', Cube_ID: {cube_id}, Include_Conv: {include_conv}")
    
    try:
        cube_dir = os.path.join(vector_db_path, cube_id)
        if not os.path.exists(cube_dir):
            logging.error(f"OLAP_PROCESS_ERROR - Cube directory doesn't exist for cube_id: {cube_id}")
            return query, "Cube data doesn,t exists", 0.0, "",""

        start_time = time.time()
        logging.info(f"OLAP_PROCESS_DIMS - Starting dimension extraction")
        dimensions = self.dim_measure.get_dimensions(query, cube_id, prev_conv)
        logging.info(f"OLAP_PROCESS_DIMS - Dimension extraction completed")
        
        logging.info(f"OLAP_PROCESS_MSRS - Starting measure extraction")
        measures = self.dim_measure.get_measures(query, cube_id, prev_conv)
        logging.info(f"OLAP_PROCESS_MSRS - Measure extraction completed")

        if not dimensions or not measures:
            logging.error(f"OLAP_PROCESS_ERROR - Failed to extract dimensions or measures")
            raise ValueError("Failed to extract dimensions or measures")

        # Choose the appropriate query generator based on include_conv
        if include_conv.lower() == "yes" and prev_conv.get('query'):
            logging.info(f"OLAP_PROCESS_CONV - Using conversational query generator")
            query_generator = ConversationalQueryGenerator(query, dimensions, measures, self.llm)
        else:
            logging.info(f"OLAP_PROCESS_FINAL - Using final query generator")
            query_generator = FinalQueryGenerator(query, dimensions, measures, self.llm)

        logging.info(f"OLAP_PROCESS_GENERATE - Starting final query generation")
        final_query = query_generator.generate_query(query, dimensions, measures, prev_conv, cube_name)
        
        processing_time = time.time() - start_time
        logging.info(f"OLAP_PROCESS_SUCCESS - OLAP processing completed in {processing_time:.2f} seconds")
        logging.info(f"OLAP_PROCESS_RESULT - Final query: {final_query}")
        
        return query, final_query, processing_time, dimensions, measures
        
    except Exception as e:
        logging.error(f"OLAP_PROCESS_ERROR - Error in query processing: {e}")
        
        try:
            logging.info(f"OLAP_PROCESS_RESET - Attempting to reset processor state")
            self.reset_state()
            logging.info(f"OLAP_PROCESS_RESET - State reset successful")
        except Exception as reset_error:
            logging.error(f"OLAP_PROCESS_RESET_ERROR - State reset failed: {reset_error}")    
        raise
