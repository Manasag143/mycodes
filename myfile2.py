# ============================================================================
# COMPLETE SMARTFUNCTIONSMANAGER CLASS (with all methods)
# ============================================================================

class SmartFunctionsManager:
    """
    Manages OLAP functions with smart selection based on query analysis
    """
    
    def __init__(self, functions_file: str = "olap_functions.yaml"):
        self.functions_file = functions_file
        self.functions_library = self._load_functions_library()
        
    def _load_functions_library(self) -> Dict:
        """Load functions from YAML file"""
        try:
            with open(self.functions_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Functions file {self.functions_file} not found. Using default functions.")
            return self._get_default_functions()
        except Exception as e:
            logging.error(f"Error loading functions file: {e}")
            return self._get_default_functions()
    
    def _get_default_functions(self) -> Dict:
        """Fallback default functions if YAML file not available"""
        return {
            "time_functions": {
                "TimeBetween": {
                    "syntax": "TimeBetween(start_date, end_date, time_level, include_end)",
                    "example": "TimeBetween(20120101,20171231,[Time].[Year], false)",
                    "keywords": ["between", "range", "from", "to", "during", "date"]
                },
                "TRENDNUMBER": {
                    "syntax": "TRENDNUMBER(measure, time_level, periods, trend_type)",
                    "example": "TRENDNUMBER([Measures.PROFIT], [Calendar.Year], 2, 'percentage')",
                    "keywords": ["trend", "change", "growth", "yoy", "mom", "previous"]
                }
            },
            "ranking_functions": {
                "Head": {
                    "syntax": "Head(dimension, measure, count, undefined)",
                    "example": "Head([Branch Details].[City], [Business Drivers].[Balance Amount], 5, undefined)",
                    "keywords": ["top", "best", "highest", "first", "maximum"]
                },
                "Tail": {
                    "syntax": "Tail(dimension, measure, count, undefined)",
                    "example": "Tail([Time].[Year], [Financial Data].[Total Revenue], 4, undefined)",
                    "keywords": ["bottom", "worst", "lowest", "last", "minimum"]
                }
            },
            "utility_functions": {
                "ROUND": {
                    "syntax": "ROUND(kpi, decimal_places)",
                    "example": "ROUND([Measures.PROFIT], 3)",
                    "keywords": ["round", "decimal", "precision"]
                }
            }
        }
    
    def _analyze_query_intent(self, query: str) -> List[str]:
        """Analyze query to determine which function categories are needed"""
        query_lower = query.lower()
        needed_categories = []
        
        # Check for time-related queries
        time_keywords = ["between", "range", "from", "to", "year", "month", "date", 
                         "yoy", "mom", "trend", "previous", "next", "change", "growth"]
        if any(keyword in query_lower for keyword in time_keywords):
            needed_categories.append("time_functions")
        
        # Check for ranking queries  
        ranking_keywords = ["top", "bottom", "best", "worst", "highest", "lowest", 
                           "first", "last", "rank", "maximum", "minimum"]
        if any(keyword in query_lower for keyword in ranking_keywords):
            needed_categories.append("ranking_functions")
        
        # Check for conditional queries
        conditional_keywords = ["where", "if", "when", "only", "filter", "exclude", 
                               "condition", "greater", "less", "equal"]
        if any(keyword in query_lower for keyword in conditional_keywords):
            needed_categories.append("conditional_functions")
        
        # Check for aggregation queries
        agg_keywords = ["sum", "total", "average", "count", "percentage", "%", 
                       "cumulative", "running"]
        if any(keyword in query_lower for keyword in agg_keywords):
            needed_categories.append("aggregation_functions")
        
        # Check for comparison queries
        comp_keywords = ["in", "like", "contains", "not"]
        if any(keyword in query_lower for keyword in comp_keywords):
            needed_categories.append("comparison_functions")
        
        # Check for mathematical operations
        math_keywords = ["greater", "less", "above", "below", "more", "exceeds"]
        if any(keyword in query_lower for keyword in math_keywords):
            needed_categories.append("mathematical_operations")
        
        # Always include utility functions (small category)
        needed_categories.append("utility_functions")
        
        # If no specific functions detected, include all (fallback)
        if len(needed_categories) == 1:  # Only utility
            needed_categories = list(self.functions_library.keys())
        
        return needed_categories
    
    # THIS IS THE MISSING METHOD!
    def build_dynamic_functions_section(self, query: str) -> str:
        """Build functions section with only relevant functions"""
        needed_categories = self._analyze_query_intent(query)
        functions_text = "<functions>\n"
        
        for category in needed_categories:
            if category in self.functions_library:
                category_funcs = self.functions_library[category]
                
                # Add category header
                category_name = category.replace("_", " ").title()
                functions_text += f"\n## {category_name}:\n"
                
                # Add each function in this category
                for func_name, func_info in category_funcs.items():
                    functions_text += f"- {func_name}: {func_info['syntax']}\n"
                    functions_text += f"  Example: {func_info['example']}\n"
                    if 'use_case' in func_info:
                        functions_text += f"  Use: {func_info['use_case']}\n"
        
        functions_text += "</functions>"
        
        # Log optimization metrics
        total_functions = sum(len(funcs) for funcs in self.functions_library.values())
        selected_functions = sum(len(self.functions_library[cat]) for cat in needed_categories if cat in self.functions_library)
        
        print(f"Function optimization: Using {selected_functions}/{total_functions} functions ({(selected_functions/total_functions)*100:.1f}%)")
        logging.info(f"Function optimization: Using {selected_functions}/{total_functions} functions ({(selected_functions/total_functions)*100:.1f}%)")
        
        return functions_text

# ============================================================================
# CORRECTED generate_query METHOD
# ============================================================================

def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict, cube_name: str) -> str:
    """
    Optimized query generation with smart function selection
    """
    try:
        if not dimensions or not measures:
            raise ValueError("Both dimensions and measures are required to generate a query.")
            
        # CORRECTED: Get dynamic functions section based on query analysis
        dynamic_functions = self.functions_manager.build_dynamic_functions_section(query)
        
        # Load sample queries (existing code)
        sample_queries = self.load_sample_queries(cube_id=cube_name)
        sample_query_examples = ""
        if sample_queries and len(sample_queries) > 0:
            sample_query_examples = "\n<sample_queries_for_this_cube>\n"
            # Limit to first 5 examples for efficiency
            for idx, sq in enumerate(sample_queries[:5], 1):
                sample_query_examples += f"user query:{sq['user_query']}\n"
                sample_query_examples += f"Expected Response:-{sq['cube_query']}\n\n"
            sample_query_examples += "</sample_queries_for_this_cube>\n"
        
        # UPDATED: Optimized prompt with dynamic function selection
        final_prompt = f"""You are an expert in generating SQL Cube query. You will be provided dimensions delimited by $$$$ and measures delimited by &&&&.
        Your Goal is to generate a precise single line cube query for the user query delimited by ####.

        Instructions:            
        - Generate a single-line Cube query without line breaks
        - Include 'as' aliases for all level names in double quotes. alias are always level names.
        - Choose the most appropriate dimensions group names and level from dimensions delimited by $$$$ according to the query.
        - Choose the most appropriate measures group names and level from measures delimited by &&&& according to the query.
        - check the examples to learn about correct syntax, functions and filters which can be used according to the user query requirement.
        - User Query could be a follow up query in a conversation, you will also be provided previous query, dimensions, measures, cube query. Generate the final query including the contexts from conversation as appropriate.

        Formatting Rules:
        - Dimensions format: [Dimension Group Name].[Dimension Level Name] as "Dimension Level Name"
        - Measures format: [Measure Group Name].[Measure Level Name] as "Measure Level Name"
        - Conditions in WHERE clause must be properly formatted with operators
        - For multiple conditions, use "and" "or" operators
        - All string values in conditions must be in single quotes
        - All numeric values should not have leading zeros

        {dynamic_functions}

        {sample_query_examples}

        <final_review>
        - ensure if the query has been generated with dimensions and measures extracted only from the current and previous conversation
        - check if functions and filters have been used appropriately in the final cube query, ensure generated query contains filters and functions from given supported functions only
        - review the syntax of the final cube query, refer the examples to help with the review for syntax check, functions and filters usage
        </final_review>

        User Query: ####{query}####
        
        $$$$
        Dimensions: {dimensions}
        $$$$

        &&&&
        Measures: {measures}
        &&&&

        Generate a precise single-line Cube query that exactly matches these requirements:"""

        print(Fore.CYAN + '   Generating optimized OLAP cube Query......................\n')
        
        # Generate query
        result = self.llm.invoke(final_prompt)
        output = result.content
        token_details = result.response_metadata.get('token_usage', {})
        pred_query = self.cleanup_gen_query(output)
        
        # Log optimization results
        selected_categories = self.functions_manager._analyze_query_intent(query)
        print(f"Selected function categories: {selected_categories}")
        print(f"Generated Query: {pred_query}")
        
        logging.info(f"Generated OLAP Query with {token_details.get('total_tokens', 'unknown')} tokens: {pred_query}")
        return pred_query
    
    except Exception as e:
        logging.error(f"Error generating OLAP query: {e}")
        raise
