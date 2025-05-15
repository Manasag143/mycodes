class ResultSummarizer:
    """
    Class responsible for generating summaries of SQL query results.
    Enhanced to handle both LIKE pattern matching and Levenshtein distance explanations.
    """
    def __init__(self, llm_adapter):
        self.llm = llm_adapter
    
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
            
            # Get basic statistics
            num_rows = len(df)
            num_columns = len(df.columns)
            column_names = list(df.columns)
            
            # Create a sample of the results (first 5 rows) as a formatted string
            sample_rows = df.head(5).to_dict('records')
            sample_str = json.dumps(sample_rows, indent=2)
            
            # Construct the prompt for summary generation - now requesting only 2 lines
            summary_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a data analyst who specializes in creating extremely concise summaries of database query results.
Given the information about a database query and its results, provide a direct answer to the original question
in AT MOST 2 LINES. Do not include any additional insights or explanations.

Be direct and factual. Focus only on providing a direct answer to the original question. Do not use bullet points.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Original question: "{original_query}"

SQL query executed:
{sql_query}

Query returned {num_rows} rows with {num_columns} columns: {', '.join(column_names)}

Here's a sample of the results (first 5 rows or less):
{sample_str}

Please provide a MAXIMUM 2-LINE direct answer to the original question without any additional insights or details.
Focus only on the main facts that directly answer the question.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            # Generate the summary using Llama
            generator = LLMGenerator()
            summary = generator.run(summary_prompt)
            
            # Ensure the summary is really no more than 2 lines
            summary_lines = summary.strip().split('\n')
            if len(summary_lines) > 2:
                summary = ' '.join(summary_lines[:2])
            
            return summary.strip()
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return f"Unable to generate summary due to an error: {str(e)}"
