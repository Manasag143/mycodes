from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import uvicorn
from datetime import datetime
import json

# Import your existing classes (assuming they're in the same file or properly imported)
# from your_nlq_module import ConversationalNLQProcessor

# Pydantic models for request/response
class NLQRequest(BaseModel):
    username: str = Field(..., description="Username for conversation tracking", min_length=1)
    query: str = Field(..., description="Natural language query", min_length=1)

class NLQResponse(BaseModel):
    sql_query: str = Field(..., description="Generated SQL query")
    summary: str = Field(..., description="Human-readable summary of results")
    data: List[Dict[str, Any]] = Field(..., description="Retrieved data from database")



# Initialize FastAPI app
app = FastAPI(
    title="Natural Language to SQL Query Processor API",
    description="Convert natural language queries to SQL and execute them on the database",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fastapi_nlq.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global configuration
CONFIG_PATH = "config1.json"

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "Natural Language to SQL Query Processor API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/process-query", response_model=NLQResponse)
async def process_nlq_query(request: NLQRequest):
    """
    Process a natural language query and return SQL, summary, and data.
    
    Args:
        request: NLQRequest containing username and query
        
    Returns:
        NLQResponse with sql_query, summary, and data only
    """
    try:
        logger.info(f"Processing query for user '{request.username}': '{request.query}'")
        
        # Validate input
        if not request.username.strip():
            raise HTTPException(status_code=400, detail="Username cannot be empty")
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Initialize the conversational processor
        processor = ConversationalNLQProcessor(
            query=request.query,
            username=request.username,
            config_path=CONFIG_PATH
        )
        
        # Process the query
        result = processor.process_query()
        
        # Close processor resources
        processor.close()
        
        # Handle errors
        if "error" in result:
            logger.error(f"Error processing query: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return only the three required fields
        return NLQResponse(
            sql_query=result.get("sql_query", ""),
            summary=result.get("summary", "No summary available"),
            data=result.get("results", [])
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/user-history/{username}")
async def get_user_history(username: str):
    """
    Get conversation history for a specific user.
    
    Args:
        username: Username to get history for
        
    Returns:
        User's conversation history
    """
    try:
        if not username.strip():
            raise HTTPException(status_code=400, detail="Username cannot be empty")
        
        # Import and use the history manager
        from your_nlq_module import ConversationHistoryManager  # Adjust import as needed
        
        history_manager = ConversationHistoryManager()
        user_history = history_manager.get_user_history(username)
        
        return {
            "username": username,
            "history": user_history,
            "interaction_count": len(user_history)
        }
        
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving user history: {str(e)}")

@app.post("/process-query-batch")
async def process_batch_queries(requests: List[NLQRequest]):
    """
    Process multiple queries in batch.
    
    Args:
        requests: List of NLQRequest objects
        
    Returns:
        List of NLQResponse objects with sql_query, summary, and data only
    """
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 10 queries")
    
    results = []
    
    for req in requests:
        try:
            # Process each query individually
            result = await process_nlq_query(req)
            results.append(result)
        except HTTPException as e:
            # Add error result for failed queries - return empty response with error in summary
            results.append(NLQResponse(
                sql_query="",
                summary=f"Error: {e.detail}",
                data=[]
            ))
    
    return results

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return NLQResponse(
        sql_query="",
        summary=f"Error: {exc.detail}",
        data=[]
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return NLQResponse(
        sql_query="",
        summary="Error: Internal server error",
        data=[]
    )

# Additional utility endpoints
@app.get("/tables")
async def get_available_tables():
    """Get list of available tables in the database"""
    try:
        # You can implement this by creating a simple processor instance
        # and returning the table information
        processor = ConversationalNLQProcessor("dummy", "system", CONFIG_PATH)
        tables = list(processor.table_data.keys())
        processor.close()
        
        return {
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        logger.error(f"Error getting tables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving tables: {str(e)}")

# Main function to run the server
def run_server():
    """Run the FastAPI server"""
    uvicorn.run(
        "main:app",  # Adjust this based on your file name
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
