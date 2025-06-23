# Modified Pydantic models for error injection API

class CubeErrorRequest(BaseModel):
    user_query: str
    cube_id: str
    error_message: str
    cube_name: str
    application_name: str

class CubeErrorResponse(BaseModel):
    message: str
    cube_query: str

# Modified error injection endpoint
@app.post("/genai/cube_error_injection", response_model=CubeErrorResponse)
async def handle_cube_error(request: CubeErrorRequest, user_details: str = Depends(verify_token)):
    user_id = f"user_{user_details}"
    
    logging.info(f"API_HIT - cube_error_injection endpoint hit by user: {user_details}")
    logging.info(f"API_REQUEST - User_Query: '{request.user_query}', Cube_ID: {request.cube_id}, Application: {request.application_name}")
    logging.info(f"ERROR_INJECTION_START - Processing error correction for error: {request.error_message}")
    
    try:
        cube_id = request.cube_id
        cube_dir = os.path.join(vector_db_path, cube_id)
        
        if not os.path.exists(cube_dir):
            logging.error(f"ERROR_INJECTION_FAILED - Cube directory not found for cube_id: {cube_id}")
            return CubeErrorResponse(
                message="failure",
                cube_query="Cube data doesn't exist"
            )
        
        logging.info(f"ERROR_INJECTION_CUBE_VALIDATION - Cube directory validated for cube_id: {cube_id}")
        
        # Get conversation history for this user and cube
        history_manager = History()
        logging.info(f"ERROR_INJECTION_HISTORY - Retrieving conversation history for user: {user_id}")
        
        prev_conversation = history_manager.retrieve(user_id, request.cube_id)
        
        # Ensure cube_name is set in conversation history
        if "cube_name" not in prev_conversation:
            prev_conversation["cube_name"] = request.cube_name
            
        # Set the current query as the user query (same query, new response)
        prev_conversation["query"] = request.user_query
        
        logging.info(f"ERROR_INJECTION_CONTEXT - Previous conversation retrieved with query: '{prev_conversation.get('query', 'None')}'")
        
        # Get or create processor for this user
        logging.info(f"ERROR_INJECTION_PROCESSOR - Getting processor for user_id: {user_id}")
        processor = processor_manager.get_processor(user_id)
        logging.info(f"ERROR_INJECTION_PROCESSOR - Processor obtained successfully")
        
        logging.info(f"ERROR_INJECTION_PROCESSING - Starting error correction processing")
        query, final_query, processing_time, dimensions, measures = processor.process_query_with_error(
            request.user_query,
            request.cube_id,
            prev_conversation,
            request.error_message,
            request.cube_name
        )
        logging.info(f"ERROR_INJECTION_PROCESSING - Error correction completed in {processing_time:.2f} seconds")
        
        # Update conversation history with corrected response
        response_data = {
            "query": request.user_query,  # Same query
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query,  # New corrected response
        }
        
        logging.info(f"ERROR_INJECTION_HISTORY_UPDATE - Updating conversation history with corrected response")
        history_manager.update(user_id, response_data, request.cube_id, request.cube_name)
        logging.info(f"ERROR_INJECTION_HISTORY_UPDATE - History updated successfully")
        
        logging.info(f"ERROR_INJECTION_SUCCESS - Error correction completed successfully")
        logging.info(f"ERROR_INJECTION_RESPONSE - Corrected query: {final_query}")
        logging.info(f"ERROR_INJECTION_METADATA - Application: {request.application_name}, Processing_time: {processing_time}")
        
        return CubeErrorResponse(
            message="success",
            cube_query=final_query
        )
        
    except Exception as e:
        logging.error(f"ERROR_INJECTION_ERROR - Error in handle_cube_error: {str(e)}")
        logging.error(f"ERROR_INJECTION_CLEANUP - Cleaning up processor for user_id: {user_id}")
        
        # Clean up processor on error
        processor_manager.cleanup_processor(user_id)
        
        return CubeErrorResponse(
            message="failure",
            cube_query=f"Error processing correction: {str(e)}"
        )

# Enhanced error correction method in DimensionMeasure class
def get_dimensions_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str) -> str:
    """Extract dimensions with error correction."""
    logging.info(f"DIMENSION_ERROR_CORRECTION_START - Starting dimension error correction")
    logging.info(f"DIMENSION_ERROR_INPUT - Query: '{query}', Error: '{error}'")
    
    try:
        with get_openai_callback() as dim_cb:
            query_dim_error_inj = f"""
            As a user query to cube query conversion expert, analyze the error message from applying the cube query. 
            Your goal is to correct the identification of dimensions if they are incorrect.
            
            Below details using which the error occurred:-
            User Query: {query}
            Current Dimensions: {prev_conv["dimensions"]}
            Current Cube Query Response: {prev_conv["response"]}
            Error Message: {error}

            <error_analysis_context>
            - Analyze the error message to identify incorrect dimension references
            - Check for syntax errors in dimension names/hierarchy levels
            - Verify dimension existence in cube structure
            - Identify missing or invalid dimension combinations
            - Consider temporal context for date/time dimensions
            - Focus on correcting the SAME user query with proper dimensions
            </error_analysis_context>

            <correction_guidelines>
            - Provide corrected dimensions for the SAME user query
            - Fix incorrect dimension names based on error message
            - Correct hierarchy references
            - Add required missing dimensions
            - Remove invalid combinations
            - Preserve valid dimensions that work
            - Maintain temporal relationships
            
            Response format (provide only the corrected dimensions):
            Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description>
            </correction_guidelines>

            <examples>
            1. Error: "Unknown dimension [Time].[Date]"
            Correction: Change to [Time].[Month] since Date level doesn't exist

            2. Error: "Invalid hierarchy reference [Geography].[City]"
            Correction: Change to [Geography].[Region].[City] to include parent

            3. Error: "Missing required dimension [Product]"
            Correction: Add [Product].[Category] based on context

            4. Error: "Incompatible dimensions [Customer] and [Item]"
            Correction: Remove [Item] dimension as it's incompatible
            </examples>
            """
            
            cube_dir = os.path.join(vector_db_path, cube_id)
            cube_dim = os.path.join(cube_dir, "dimensions")
            
            logging.info(f"DIMENSION_ERROR_VECTOR - Loading dimension vector store")
            load_embedding_dim = Chroma(
                persist_directory=cube_dim,
                embedding_function=self.embedding
            )
            retriever_dim = load_embedding_dim.as_retriever(search_kwargs={"k": 15})
            
            logging.info(f"DIMENSION_ERROR_LLM - Invoking LLM for dimension error correction")
            chain_dim = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever_dim,
                verbose=True,
                return_source_documents=True
            )
            
            result_dim = chain_dim.invoke({"query": query_dim_error_inj})
            corrected_dimensions = result_dim.get('result')
            
            logging.info(f"DIMENSION_ERROR_SUCCESS - Corrected dimensions: {corrected_dimensions}")
            logging.info(f"DIMENSION_ERROR_TOKENS - Token usage: {dim_cb}")
            
            return corrected_dimensions

    except Exception as e:
        logging.error(f"DIMENSION_ERROR_FAILED - Error in dimension correction: {e}")
        raise

def get_measures_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str) -> str:
    """Extracts measures with error correction logic"""
    logging.info(f"MEASURE_ERROR_CORRECTION_START - Starting measure error correction")
    logging.info(f"MEASURE_ERROR_INPUT - Query: '{query}', Error: '{error}'")
    
    try:
        with get_openai_callback() as msr_cb:
            query_msr_error_inj = f"""
            As a user query to cube query conversion expert, analyze the error message from applying the cube query. 
            Your goal is to correct the identification of measures if they are incorrect.
            
            Below details using which the error occurred:-
            User Query: {query}
            Current Measures: {prev_conv["measures"]}
            Current Cube Query Response: {prev_conv["response"]}
            Error Message: {error}

            <error_analysis_context>
            - Analyze error message for specific measure-related issues
            - Check for syntax errors in measure references
            - Verify measures exist in the cube structure
            - Look for aggregation function errors
            - Identify calculation or formula errors
            - Check for measure compatibility issues
            - Focus on correcting the SAME user query with proper measures
            </error_analysis_context>

            <correction_guidelines>
            - Provide corrected measures for the SAME user query
            - Fix measure name mismatches
            - Correct aggregation functions
            - Fix calculation formulas
            - Add missing required measures
            - Remove invalid measure combinations
            - Preserve valid measure selections
            
            Response format (provide only the corrected measures):
            Group Name:<Group Name>--Level Name:<Level Name>--Description:<Description>
            </correction_guidelines>

            <examples>
            1. Error: "Unknown measure [Sales].[Amount]"
            Correction: Change to [Sales].[Sales Amount] as the correct measure name

            2. Error: "Invalid aggregation function SUM for [Average Price]"
            Correction: Change to AVG([Price].[Unit Price]) for correct aggregation

            3. Error: "Incompatible measures [Profit] and [Margin %]"
            Correction: Use [Sales].[Profit Amount] instead of incompatible combination

            4. Error: "Calculation error in [Growth].[YoY]"
            Correction: Add proper year-over-year calculation formula

            5. Error: "Missing required base measure for [Running Total]"
            Correction: Add base measure [Sales].[Amount] for running total calculation
            </examples>
            """

            cube_dir = os.path.join(vector_db_path, cube_id)
            cube_msr = os.path.join(cube_dir, "measures")
            
            logging.info(f"MEASURE_ERROR_VECTOR - Loading measure vector store")
            load_embedding_msr = Chroma(
                persist_directory=cube_msr,
                embedding_function=self.embedding
            )
            retriever_msr = load_embedding_msr.as_retriever(search_kwargs={"k": 15})
            
            logging.info(f"MEASURE_ERROR_LLM - Invoking LLM for measure error correction")
            chain_msr = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever_msr,
                verbose=True,
                return_source_documents=True
            )
            
            result_msr = chain_msr.invoke({"query": query_msr_error_inj})
            corrected_measures = result_msr.get('result')
            
            logging.info(f"MEASURE_ERROR_SUCCESS - Corrected measures: {corrected_measures}")
            logging.info(f"MEASURE_ERROR_TOKENS - Token usage: {msr_cb}")
            
            return corrected_measures

    except Exception as e:
        logging.error(f"MEASURE_ERROR_FAILED - Error in measure correction: {str(e)}")
        raise

# Enhanced process_query_with_error method in OLAPQueryProcessor class
def process_query_with_error(self, query: str, cube_id: str, prev_conv: dict, error: str, cube_name: str) -> Tuple[str, str, float, Dict, Dict]:
    """Process a query with error correction."""
    logging.info(f"OLAP_ERROR_PROCESS_START - Starting OLAP query processing with error correction")
    logging.info(f"OLAP_ERROR_INPUT - Query: '{query}', Cube_ID: {cube_id}, Error: '{error}'")
    
    try:
        start_time = time.time()
        
        # Get corrected dimensions and measures
        logging.info(f"OLAP_ERROR_DIMS - Starting corrected dimension extraction")
        dimensions = self.dim_measure.get_dimensions_with_error(query, cube_id, prev_conv, error)
        logging.info(f"OLAP_ERROR_DIMS - Corrected dimension extraction completed")
        
        logging.info(f"OLAP_ERROR_MSRS - Starting corrected measure extraction")
        measures = self.dim_measure.get_measures_with_error(query, cube_id, prev_conv, error)
        logging.info(f"OLAP_ERROR_MSRS - Corrected measure extraction completed")

        if not dimensions or not measures:
            logging.error(f"OLAP_ERROR_FAILED - Failed to extract dimensions or measures after error correction")
            raise ValueError("Failed to extract dimensions or measures after error correction")

        logging.info(f"OLAP_ERROR_GENERATE - Starting corrected final query generation")
        final_query = self.final.generate_query(query, dimensions, measures, prev_conv, cube_name)
        
        processing_time = time.time() - start_time
        logging.info(f"OLAP_ERROR_SUCCESS - Error correction processing completed in {processing_time:.2f} seconds")
        logging.info(f"OLAP_ERROR_RESULT - Corrected final query: {final_query}")
        
        return query, final_query, processing_time, dimensions, measures
        
    except Exception as e:
        logging.error(f"OLAP_ERROR_PROCESS_ERROR - Error in query processing with error correction: {e}")
        raise
