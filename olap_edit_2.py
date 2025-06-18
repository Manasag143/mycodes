if __name__ == "__main__":
    import logging
    import uvicorn
    
    # Completely disable uvicorn logging
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.disabled = True
    uvicorn_logger.handlers.clear()
    
    uvicorn_error = logging.getLogger("uvicorn.error") 
    uvicorn_error.disabled = True
    uvicorn_error.handlers.clear()
    
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True
    uvicorn_access.handlers.clear()
    
    # Disable watchfiles completely
    watchfiles_logger = logging.getLogger("watchfiles")
    watchfiles_logger.disabled = True
    watchfiles_logger.handlers.clear()
    
    watchfiles_main = logging.getLogger("watchfiles.main")
    watchfiles_main.disabled = True
    watchfiles_main.handlers.clear()
    
    # Also try these additional loggers
    logging.getLogger("uvicorn.protocols").disabled = True
    logging.getLogger("uvicorn.protocols.http").disabled = True
    logging.getLogger("uvicorn.lifespan").disabled = True
    logging.getLogger("uvicorn.lifespan.on").disabled = True
    
    num_cores = multiprocessing.cpu_count()
    optimal_workers = 2 * num_cores + 1
    
    # Run with minimal logging
    uvicorn.run(
        "olap_details_generat:app", 
        host="0.0.0.0", 
        port=8085, 
        reload=False,
        workers=1,  # Use 1 worker to avoid multiple processes
        access_log=False,  # Disable access logging
        log_level="warning"  # Set to warning level
    )
