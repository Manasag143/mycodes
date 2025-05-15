# Display the results
if result["results"]:
    print(Fore.BLUE + f"\nResults: ({len(result['results'])} rows)")
    
    # Convert to DataFrame for nicer display
    df = pd.DataFrame(result["results"])
    
    # Display first 100 rows (increased from 10)
    num_rows = min(100, len(df))
    print(df.head(num_rows))
    
    if len(df) > num_rows:
        print(Fore.CYAN + f"... and {len(df) - num_rows} more rows")
    
    # Display the summary
    print(Fore.BLUE + "\nSummary:")
    print(Fore.GREEN + "=" * 80)
    print(Fore.WHITE + result["summary"])
    print(Fore.GREEN + "=" * 80)
