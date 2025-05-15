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
    
    # Set pandas to display more rows and columns
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
                    
                    # Alternative method 2 (uncomment if method 1 doesn't show all rows):
                    # print(tabulate(df.head(num_rows), headers='keys', tablefmt='psql'))
                    
                    # Alternative method 3 (uncomment if methods 1 and 2 don't work):
                    # for i in range(num_rows):
                    #     print(f"Row {i+1}:")
                    #     print(df.iloc[i])
                    #     print("-" * 40)
                    
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
            
            # Display conversation history
            if history_manager.get_user_history(username):
                print(Fore.BLUE + "\nYour recent conversation history:")
                history = history_manager.get_user_history(username)
                for i, interaction in enumerate(history[-min(3, len(history)):]):
                    print(Fore.CYAN + f"  {i+1}. {interaction['time']}: {interaction['query']}")
            
            processor.close()
            
        except Exception as e:
            print(Fore.RED + f"Error: {str(e)}")
            logging.error(f"Unhandled error: {str(e)}", exc_info=True)
            
if __name__ == "__main__":
    main()
