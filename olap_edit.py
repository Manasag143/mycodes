# Updated Pydantic Models
class CubeQueryRequest(BaseModel):
    user_query: str          # The natural language query from user
    cube_id: str            # Identifier for the specific cube
    cube_name: str          # Name of the cube (e.g., "Credit One View")
    application_name: str   # Name of the application making the request
    include_conv: str       # Whether to include conversation context ("yes" or "no")
    regenerate: str         # Whether to regenerate response ("yes" or "no")

# Updated History class methods
class History:
    def __init__(self, history_file: str = history_file):
        self.history_file = history_file        
        try:
            # Create file if it doesn't exist
            if not os.path.exists(self.history_file):
                with open(self.history_file, 'w') as f:
                    json.dump({"users": {}}, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to create history file: {str(e)}")
            raise
        
        self.history = self.load()

    def load(self) -> Dict:
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)                    
                    # Migrate old format to new format if needed
                    if "users" not in data:
                        migrated_data = {"users": {}}
                        for key, value in data.items():
                            if key != "cube_id":  # Skip the old top-level cube_id
                                migrated_data["users"][key] = {}
                                if isinstance(value, list):
                                    old_cube_id = data.get("cube_id", "")
                                    migrated_data["users"][key][old_cube_id] = value
                        return migrated_data
                    
                    return data
            return {"users": {}}
        except Exception as e:
            logging.error(f"Error loading conversation history: {str(e)}")
            return {"users": {}}

    def save(self, history: Dict):
        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=4)
            
            # Verify the save
            with open(self.history_file, 'r') as f:
                saved_data = json.load(f)
                
        except Exception as e:
            logging.error(f"Error saving conversation history: {str(e)}")
            raise

    def update(self, user_id: str, query_data: Dict, cube_id: str, cube_name: str = None):
        try:
            
            # Initialize nested structure if needed
            if "users" not in self.history:
                self.history["users"] = {}
                
            if user_id not in self.history["users"]:
                self.history["users"][user_id] = {}
            
            if cube_id not in self.history["users"][user_id]:
                self.history["users"][user_id][cube_id] = []
            
            # Add new conversation without cube_id inside the block
            new_conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": query_data["query"],
                "dimensions": query_data["dimensions"],
                "measures": query_data["measures"],
                "response": query_data["response"]
            }
            
            # Add cube_name if provided
            if cube_name:
                new_conversation["cube_name"] = cube_name
            
            self.history["users"][user_id][cube_id].append(new_conversation)
            # Keep last 5 conversations for this user and cube
            self.history["users"][user_id][cube_id] = self.history["users"][user_id][cube_id][-5:]
            self.save(self.history)
        
        except Exception as e:
            logging.error(f"Error in update: {str(e)}")
            raise
    
    def retrieve(self, user_id: str, cube_id: str, regenerate: str = "no"):
        """Retrieve conversation for this user and cube"""
        try:            
            # Check if we have history for this user and cube
            if "users" not in self.history:
                self.history["users"] = {}
            
            if user_id not in self.history["users"]:
                self.history["users"][user_id] = {}
                return self._empty_conversation(cube_id)
            
            if cube_id not in self.history["users"][user_id]:
                self.history["users"][user_id][cube_id] = []
                return self._empty_conversation(cube_id)
            
            try:
                conversations = self.history["users"][user_id][cube_id]
                if not conversations:
                    return self._empty_conversation(cube_id)
                
                # If regenerate is "yes", get 2nd last conversation (not the most recent)
                if regenerate.lower() == "yes" and len(conversations) >= 2:
                    logging.info(f"REGENERATE_MODE - Using 2nd last conversation for regeneration")
                    return conversations[-2]  # Return 2nd last conversation
                else:
                    # Normal mode: return last conversation
                    logging.info(f"NORMAL_MODE - Using last conversation")
                    return conversations[-1]
                    
            except IndexError:
                logging.info("No conversations found in history")
                return self._empty_conversation(cube_id)
        except Exception as e:
            logging.error(f"Error in retrieve: {str(e)}")
            return self._empty_conversation(cube_id)
    
    def _empty_conversation(self, cube_id: str, cube_name: str = None):
        """Helper method to return an empty conversation structure"""
        empty_conv = {
            "timestamp": datetime.now().isoformat(),
            "query": "",
            "dimensions": "",
            "measures": "",
            "response": ""
        }
        
        if cube_name:
            empty_conv["cube_name"] = cube_name
            
        return empty_conv
    
    def clear_history(self, user_id: str, cube_id: str):
        """Clear conversation history for a specific user and cube"""
        try:
            if "users" in self.history and user_id in self.history["users"] and cube_id in self.history["users"][user_id]:
                self.history["users"][user_id][cube_id] = []
                self.save(self.history)
                return True
            return False
        except Exception as e:
            logging.error(f"Error clearing history: {str(e)}")
            return False

# Updated process_query function
async def process_query(user_query: str, cube_id: str, user_id: str, cube_name="Credit One View", include_conv="no", regenerate="no") -> Dict:
    logging.info(f"OLAP_PROCESS_START - Starting OLAP query processing")
    logging.info(f"OLAP_PROCESS_INPUT - Query: '{user_query}', Cube_ID: {cube_id}, Include_Conv: {include_conv}, Regenerate: {regenerate}")
    try:
        # Get or create processor for this user
        with open(r'conversation_history.json') as conv_file: 
            conv_json = json.load(conv_file)
            
            # Initialize user structure if needed
            if "users" not in conv_json:
                conv_json["users"] = {}
                
            if conv_json.get("users", {}).get(user_id) is None:
                print("Initializing user in conversation history")
                if "users" not in conv_json:
                    conv_json["users"] = {}
                conv_json["users"][user_id] = {}
                with open(r'conversation_history.json', 'w') as conv_file:
                    json.dump(conv_json, conv_file)

        # Handle conversation context based on include_conv and regenerate parameters
        if include_conv.lower() == "no":
            # No conversation context
            logging.info(f"CONVERSATION_CONTEXT - No conversation context requested")
            prev_conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": "",
                "dimensions": "",
                "measures": "",
                "response": "",
                "cube_id": cube_id,
                "cube_name": cube_name
            }
        else:
            # Get conversation history with regenerate consideration
            history_manager = History()
            logging.info(f"CONVERSATION_CONTEXT - Getting conversation history with regenerate={regenerate}")
            prev_conversation = history_manager.retrieve(user_id, cube_id, regenerate)
            
            # Ensure cube_name is set (it might be missing in older records)
            if "cube_name" not in prev_conversation:
                prev_conversation["cube_name"] = cube_name

        # Get processor for this user
        olap_processors[user_id] = OLAPQueryProcessor(config_file)
        processor = olap_processors[user_id]

        # Process query and get results
        query, final_query, processing_time, dimensions, measures = processor.process_query(
            user_query, cube_id, prev_conversation, cube_name, include_conv
        )
        
        # Prepare response data
        response_data = {
            "query": query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query,
        }

        # Update history logic based on regenerate parameter
        if regenerate.lower() == "yes":
            # For regeneration, replace the last conversation instead of adding new one
            logging.info(f"REGENERATE_UPDATE - Replacing last conversation with regenerated response")
            history_manager = History()
            # Remove the last conversation and add the new regenerated one
            if "users" in history_manager.history and user_id in history_manager.history["users"] and cube_id in history_manager.history["users"][user_id]:
                if history_manager.history["users"][user_id][cube_id]:
                    # Remove the last (incorrect) conversation
                    history_manager.history["users"][user_id][cube_id].pop()
            # Add the new regenerated conversation
            history_manager.update(user_id, response_data, cube_id, cube_name)
        elif include_conv.lower() == "yes":
            # Normal conversation flow - add new conversation
            logging.info(f"NORMAL_UPDATE - Adding new conversation to history")
            history_manager = History()
            history_manager.update(user_id, response_data, cube_id, cube_name)

        return {
            "message": "success",
            "cube_query": final_query,
            "dimensions": dimensions,
            "measures": measures
        }
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return {
            "message": f"failure{e}",
            "cube_query": None,
            "dimensions": "",
            "measures": ""
        }

# Updated API endpoint
@app.post("/genai/cube_query_generation", response_model=QueryResponse)
async def generate_cube_query(request: CubeQueryRequest, user_details: str = Depends(verify_token)):
    user_id = f"user_{user_details}"
    logging.info(f"API_HIT - cube_query_generation endpoint hit by user: {user_details}")
    logging.info(f"API_REQUEST - Query: '{request.user_query}', Cube_ID: {request.cube_id}, Include_Conv: {request.include_conv}, Regenerate: {request.regenerate}, App: {request.application_name}")

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
        
        # Get conversation context with regenerate consideration
        history_manager = History()
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
            logging.info(f"API_CONTEXT - Retrieving conversation context (include_conv=yes, regenerate={request.regenerate})")
            prev_conversation = history_manager.retrieve(user_id, request.cube_id, request.regenerate)
            if "cube_name" not in prev_conversation:
                prev_conversation["cube_name"] = request.cube_name

        # Get processor with error recovery
        processor = processor_manager.get_processor(user_id)
        logging.info(f"API_PROCESSOR - Processor obtained successfully")
        
        query, final_query, processing_time, dimensions, measures = processor.process_query(
            request.user_query,
            request.cube_id,
            prev_conversation,
            request.cube_name,
            request.include_conv
        )
        logging.info(f"API_PROCESSING - Query processing completed successfully in {processing_time:.2f} seconds")
        
        # Update history based on regenerate parameter
        response_data = {
            "query": request.user_query,
            "dimensions": dimensions,
            "measures": measures,
            "response": final_query
        }
        
        if request.regenerate.lower() == "yes":
            # For regeneration, replace the last conversation instead of adding new one
            logging.info(f"API_REGENERATE - Replacing last conversation with regenerated response")
            # Remove the last conversation and add the new regenerated one
            if "users" in history_manager.history and user_id in history_manager.history["users"] and request.cube_id in history_manager.history["users"][user_id]:
                if history_manager.history["users"][user_id][request.cube_id]:
                    # Remove the last (incorrect) conversation
                    history_manager.history["users"][user_id][request.cube_id].pop()
            # Add the new regenerated conversation
            history_manager.update(user_id, response_data, request.cube_id, request.cube_name)
        elif request.include_conv.lower() == "yes":
            # Normal conversation flow - add new conversation
            logging.info(f"API_NORMAL_UPDATE - Adding new conversation to history")
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
        logging.error(f"Error in generate_cube_query: {str(e)}")
        
        # CRITICAL: Clean up on error
        processor_manager.cleanup_processor(user_id)
        
        return QueryResponse(
            message="failure", 
            cube_query=f"Processing error: {str(e)}",
            dimensions="",
            measures=""
        )
