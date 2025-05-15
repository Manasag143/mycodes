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
        
        # Check what types of matching were used
        is_like_match = "like" in sql_query.lower()
        is_levenshtein_match = "levenshtein" in sql_query.lower()
        
        # Prepare matching explanation if needed
        matching_explanation = ""
        if (is_like_match or is_levenshtein_match) and len(df) > 0 and 'company_name' in df.columns:
            # Extract the original company name from the query if possible
            import re
            company_name_match = re.search(r"company_name\s*=\s*'([^']+)'", sql_query)
            original_name = company_name_match.group(1) if company_name_match else "the requested company"
            
            # If we found results with a different name, explain the matching techniques
            top_result = df.iloc[0]['company_name'] if 'company_name' in df.columns else None
            if top_result and top_result.lower() != original_name.lower():
                if is_like_match and is_levenshtein_match:
                    # Combined approach explanation
                    matching_explanation = f"""
                    Note: You searched for "{original_name}", but found similar company names using a combined approach.
                    The closest match was "{top_result}".
                    """
                elif is_like_match:
                    # LIKE pattern matching explanation
                    matching_explanation = f"""
                    Note: You searched for "{original_name}", but found similar company names using pattern matching.
                    The closest match was "{top_result}".
                    """
                elif is_levenshtein_match:
                    # Levenshtein distance explanation
                    matching_explanation = f"""
                    Note: You searched for "{original_name}", but found similar company names using approximate matching.
                    The closest match was "{top_result}".
                    """
        
        # Get basic statistics
        num_rows = len(df)
        num_columns = len(df.columns)
        column_names = list(df.columns)
        
        # Create a sample of the results (first 5 rows) as a formatted string
        sample_rows = df.head(5).to_dict('records')
        sample_str = json.dumps(sample_rows, indent=2)
        
        # Modified prompt to focus on direct answers only without additional insights
        summary_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a data analyst expert who specializes in creating clear, direct summaries of database query results.
Given the information about a database query and its results, provide a concise summary that answers
the original question directly without any additional insights or analysis.

Be clear, direct, and factual. Focus ONLY on answering the original question with the facts from the data.
Do not provide additional insights, interpretations, or suggestions beyond what was explicitly asked.
Your summary should be brief and to the point.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Original question: "{original_query}"

SQL query executed:
{sql_query}

Query returned {num_rows} rows with {num_columns} columns: {', '.join(column_names)}

Here's a sample of the results (first 5 rows or less):
{sample_str}

{matching_explanation}

Please provide only the direct answer to the original question using the data.
Keep your answer under 100 words and focus only on the facts from the results.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        # Generate the summary using Llama
        generator = LLMGenerator()
        summary = generator.run(summary_prompt)
        
        return summary.strip()
        
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return f"Unable to generate summary due to an error: {str(e)}"



# Display the results
if result["results"]:
    print(Fore.BLUE + f"\nResults: ({len(result['results'])} rows)")
    
    # Convert to DataFrame for nicer display
    df = pd.DataFrame(result["results"])
    
    # Display up to 100 rows (increased from 10)
    num_rows = min(100, len(df))
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
        
