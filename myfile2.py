{
    "time_functions": {
        "TimeBetween": {
            "syntax": "TimeBetween(start_date, end_date, time_level, include_end)",
            "example": "TimeBetween(20120101,20171231,[Time].[Year], false)",
            "keywords": ["between", "range", "from", "to", "during", "date"],
            "description": "Filter data within date ranges"
        },
        "TRENDNUMBER": {
            "syntax": "TRENDNUMBER(measure, time_level, periods, trend_type)",
            "example": "TRENDNUMBER([Measures.PROFIT], [Calendar.Year], 2, 'percentage')",
            "keywords": ["trend", "change", "growth", "yoy", "mom", "previous"],
            "description": "Year-over-year and period comparisons",
            "options": ["average", "value", "percentage", "sum", "delta"]
        }
    },
    "ranking_functions": {
        "Head": {
            "syntax": "Head(dimension, measure, count, undefined)",
            "example": "Head([Branch Details].[City], [Business Drivers].[Balance Amount], 5, undefined)",
            "keywords": ["top", "best", "highest", "first", "maximum"],
            "description": "Get top N results"
        }
    }
}













# Add these imports at the top
import json
from typing import List, Dict

# Add these methods to your FinalQueryGenerator class
def _load_functions_from_file(self, file_path: str = "olap_functions.json") -> Dict:
    """Load functions from external JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to hardcoded functions if file not found
        return self._get_default_functions()

def _get_default_functions(self) -> Dict:
    """Fallback functions if JSON file not available"""
    # Your current functions converted to structured format
    pass

def _analyze_query_intent(self, query: str) -> List[str]:
    """Analyze query to determine needed function categories"""
    # Implementation as shown above
    pass

def _build_dynamic_functions_section(self, query: str) -> str:
    """Build optimized functions section"""
    # Implementation as shown above
    pass














def generate_query(self, query: str, dimensions: str, measures: str, prev_conv: dict, cube_name: str) -> str:
    try:
        if not dimensions or not measures:
            raise ValueError("Both dimensions and measures are required to generate a query.")
        
        # NEW: Get dynamic functions section
        dynamic_functions = self._build_dynamic_functions_section(query)
        
        # NEW: Get relevant examples based on query type
        relevant_examples = self._get_relevant_examples(query, cube_name)
        
        # Load sample queries (existing code)
        sample_queries = self.load_sample_queries(cube_id=cube_name)
        sample_query_examples = ""
        if sample_queries and len(sample_queries) > 0:
            sample_query_examples = "\n<sample_queries_for_this_cube>\n"
            # Only include first 3 examples instead of all
            for idx, sq in enumerate(sample_queries[:3], 1):
                sample_query_examples += f"user query:{sq['user_query']}\n"
                sample_query_examples += f"Expected Response:-{sq['cube_query']}\n\n"
            sample_query_examples += "</sample_queries_for_this_cube>\n"
        
        # UPDATED: Use dynamic functions instead of hardcoded
        final_prompt = f"""You are an expert in generating SQL Cube query...

        INSTRUCTIONS:            
        - Generate a single-line Cube query without line breaks
        - Include 'as' aliases for all level names in double quotes
        - Choose appropriate dimensions and measures according to the query
        - Use functions from the functions section below based on query requirements

        FORMATTING RULES:
        - Dimensions format: [Dimension Group Name].[Dimension Level Name] as "Dimension Level Name"
        - Measures format: [Measure Group Name].[Measure Level Name] as "Measure Level Name"
        
        {dynamic_functions}

        {relevant_examples}

        {sample_query_examples}

        User Query: ####{query}####
        
        Dimensions: {dimensions}
        Measures: {measures}

        Generate a precise single-line Cube query:"""

        print(Fore.CYAN + '   Generating optimized OLAP cube Query......................\n')
        
        result = self.llm.invoke(final_prompt)
        output = result.content
        token_details = result.response_metadata['token_usage']
        pred_query = self.cleanup_gen_query(output)
        
        print(f"Functions used: {len(self._analyze_query_intent(query))} categories")
        print(f"Generated Query: {pred_query}")
        
        logging.info(f"Generated OLAP Query with optimized functions: {pred_query}")
        return pred_query
    
    except Exception as e:
        logging.error(f"Error generating OLAP query: {e}")
        raise





class FinalQueryGenerator(LLMConfigure):
    def __init__(self, query, dimensions: None, measures: None, llm: None):
        super().__init__()
        self.query = query
        self.dimensions = dimensions
        self.measures = measures
        self.llm = llm
        
        # NEW: Add functions library
        self.functions_library = self._load_functions_library()
    
    def _load_functions_library(self):
        """Load structured functions instead of hardcoded text"""
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
                    "keywords": ["trend", "change", "growth", "yoy", "mom", "previous", "compare"]
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
            # ... more categories
        }
