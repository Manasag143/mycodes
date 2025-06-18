@app.post("/genai/cube_details_import", response_model=CubeDetailsResponse)
async def import_cube_details(request: CubeDetailsRequest, user_details: str = Depends(verify_token)):
    try:
        # ... your existing code ...
        
        # Add these logs before return:
        logging.info(f"API_HIT - User: {user_details}, Cube: {request.cube_id}, Endpoint: cube_details_import")
        logging.info(f"LLM_RESPONDED - Success: True, Action: Import cube details")
        logging.info(f"DIMENSIONS_MEASURES - Dim_count: {len(request.cube_json_dim)}, Msr_count: {len(request.cube_json_msr)}")
        logging.info(f"QUERY_GENERATED - Success: True, Result: {result['message']}")
        
        return CubeDetailsResponse(message=result["message"])
    except HTTPException as he:
        logging.error(f"HTTP_ERROR - User: {user_details}, Endpoint: cube_details_import, Error: {str(he)}")
        return CubeDetailsResponse(message=f"failure:{he}")
    except Exception as e:
        logging.error(f"GENERAL_ERROR - User: {user_details}, Endpoint: cube_details_import, Error: {str(e)}")
        return CubeDetailsResponse(message=f"failure:{e}")




@app.post("/genai/clear_chat", response_model=ClearChatResponse)
async def clear_chat(request: ClearChatRequest, user_details: str = Depends(verify_token)):
    try:
        # ... your existing code ...
        
        # Add these logs before return:
        logging.info(f"API_HIT - User: {user_details}, Cube: {request.cube_id}, Endpoint: clear_chat")
        logging.info(f"LLM_RESPONDED - Success: True, Action: Clear chat history")
        logging.info(f"DIMENSIONS_MEASURES - Action: Chat cleared for cube {request.cube_id}")
        logging.info(f"QUERY_GENERATED - Success: True, Result: Chat history cleared")
        
        return ClearChatResponse(status="success")
    
    except Exception as e:
        logging.error(f"GENERAL_ERROR - User: {user_details}, Endpoint: clear_chat, Error: {str(e)}")
        return ClearChatResponse(status=f"failure: {str(e)}")



@app.post("/genai/cube_error_injection", response_model=CubeErrorResponse)
async def handle_cube_error(request: CubeErrorRequest, user_details: str = Depends(verify_token)):
    try:
        # ... your existing code ...
        
        # Add these logs before return:
        logging.info(f"API_HIT - User: {user_details}, Cube: {request.cube_id}, Endpoint: cube_error_injection")
        logging.info(f"LLM_RESPONDED - Success: True, Processing_time: {processing_time}")
        logging.info(f"DIMENSIONS_MEASURES - Dimensions: {dimensions}, Measures: {measures}")
        logging.info(f"QUERY_GENERATED - Success: True, Corrected_Query: {final_query}")
        
        return CubeErrorResponse(
            message="success",
            cube_query=final_query,
            error_details={
                "original_error": request.error_message,
                "correction_timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logging.error(f"GENERAL_ERROR - User: {user_details}, Endpoint: cube_error_injection, Error: {str(e)}")
        return CubeErrorResponse(
            message="failure",
            cube_query=None,
            error_details={"error_type": "processing_error", "details": str(e)}
        )




@app.post("/genai/user_feedback_injection", response_model=UserFeedbackResponse)
async def handle_user_feedback(request: UserFeedbackRequest, user_details: str = Depends(verify_token)):
    try:
        # ... your existing code ...
        
        # Add these logs before return:
        logging.info(f"API_HIT - User: {user_details}, Cube: {request.cube_id}, Endpoint: user_feedback_injection")
        logging.info(f"LLM_RESPONDED - Success: True, Feedback: {request.feedback}")
        logging.info(f"DIMENSIONS_MEASURES - Feedback_processed: {request.user_feedback}")
        logging.info(f"QUERY_GENERATED - Success: True, New_Query: {final_query if 'final_query' in locals() else 'None'}")
        
        return UserFeedbackResponse(
            message="success",
            cube_query=final_query if 'final_query' in locals() else "None"
        )
        
    except Exception as e:
        logging.error(f"GENERAL_ERROR - User: {user_details}, Endpoint: user_feedback_injection, Error: {str(e)}")
        return UserFeedbackResponse(
            message="failure",
            cube_query=None
        )



