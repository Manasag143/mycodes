import fitz  # PyMuPDF
import re
from collections import Counter

class PDFQueryEngine:
    def __init__(self, pdf_path):
        """
        Initialize the PDF query engine with a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.pdf_document = fitz.open(pdf_path)
        self.full_text = ""
        self.page_texts = []
        self._extract_all_text()
        
    def _extract_all_text(self):
        """Extract and store text from all pages of the PDF."""
        for page_num in range(len(self.pdf_document)):
            page = self.pdf_document[page_num]
            page_text = page.get_text()
            self.page_texts.append(page_text)
            self.full_text += page_text
    
    def find_by_keyword(self, keyword, context_size=100):
        """
        Find text snippets containing the given keyword.
        
        Args:
            keyword (str): Keyword to search for
            context_size (int): Number of characters to include around the keyword
            
        Returns:
            list: List of text snippets containing the keyword with context
        """
        results = []
        keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        
        for page_num, page_text in enumerate(self.page_texts):
            matches = keyword_pattern.finditer(page_text)
            for match in matches:
                start_pos = max(0, match.start() - context_size)
                end_pos = min(len(page_text), match.end() + context_size)
                
                snippet = page_text[start_pos:end_pos]
                results.append({
                    "page": page_num + 1,
                    "snippet": snippet,
                    "keyword": keyword
                })
        
        return results
    
    def answer_question(self, question):
        """
        Answer a question based on PDF content using keyword matching.
        
        Args:
            question (str): Question to answer
            
        Returns:
            dict: Answer information including relevant snippets
        """
        # Extract key terms from the question
        question_lower = question.lower()
        
        # Common words to ignore in keyword extraction
        stop_words = {"what", "where", "when", "how", "why", "who", "is", "are", "the", "in", "on", "at", "to", "a", "an"}
        
        # Extract potential keywords
        words = re.findall(r'\b\w+\b', question_lower)
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Find potential KPI indicators
        kpi_indicators = ["revenue", "profit", "growth", "percentage", "ratio", "sales", 
                         "cost", "margin", "roi", "return", "investment", "target"]
        
        # Add any KPI indicators found in the question
        kpi_keywords = [word for word in keywords if word in kpi_indicators]
        if kpi_keywords:
            keywords = keywords + kpi_keywords
        
        # If question is about numbers or specific metrics
        if any(term in question_lower for term in ["how many", "how much", "percentage", "number", "total"]):
            number_pattern = r'\b\d+(?:\.\d+)?%?\b'
            # Look for numbers near keywords
            results = []
            for keyword in keywords:
                keyword_results = self.find_by_keyword(keyword)
                for result in keyword_results:
                    numbers = re.findall(number_pattern, result["snippet"])
                    if numbers:
                        result["numbers"] = numbers
                        results.append(result)
            
            if results:
                return {
                    "question": question,
                    "answer_type": "numerical",
                    "results": results
                }
        
        # General keyword search
        all_results = []
        for keyword in keywords:
            results = self.find_by_keyword(keyword)
            all_results.extend(results)
        
        # Rank results by relevance (number of keywords found in each snippet)
        ranked_results = []
        for result in all_results:
            score = 0
            for keyword in keywords:
                if re.search(re.escape(keyword), result["snippet"], re.IGNORECASE):
                    score += 1
            result["relevance_score"] = score
            ranked_results.append(result)
        
        # Sort by relevance score
        ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Take top 3 most relevant results
        top_results = ranked_results[:3] if ranked_results else []
        
        return {
            "question": question,
            "answer_type": "descriptive",
            "results": top_results,
            "keywords_used": keywords
        }
    
    def get_kpi(self, kpi_name):
        """
        Extract KPI values from the PDF.
        
        Args:
            kpi_name (str): Name of the KPI to extract
            
        Returns:
            dict: KPI information
        """
        # Find KPI mentions
        kpi_results = self.find_by_keyword(kpi_name)
        
        # Look for patterns like "KPI: value" or "KPI is value"
        kpi_values = []
        
        for result in kpi_results:
            snippet = result["snippet"]
            
            # Pattern for "KPI: 123" or "KPI: 12.3%" or "KPI is 123"
            pattern = f"{re.escape(kpi_name)}(?::| is | was | of )? ?(\d+(?:\.\d+)?%?)"
            matches = re.findall(pattern, snippet, re.IGNORECASE)
            
            if matches:
                for match in matches:
                    kpi_values.append({
                        "value": match,
                        "page": result["page"],
                        "context": snippet
                    })
        
        return {
            "kpi": kpi_name,
            "values": kpi_values
        }
    
    def close(self):
        """Close the PDF document."""
        self.pdf_document.close()

# Example usage
if __name__ == "__main__":
    # Replace with your PDF file path
    pdf_file = "example.pdf"
    
    try:
        # Initialize the query engine
        query_engine = PDFQueryEngine(pdf_file)
        
        # Example: Get a specific KPI
        revenue_kpi = query_engine.get_kpi("revenue")
        print("\n=== Revenue KPI ===")
        if revenue_kpi["values"]:
            for item in revenue_kpi["values"]:
                print(f"Value: {item['value']} (Page {item['page']})")
        else:
            print("No revenue KPI found")
        
        # Example: Answer a question
        question = "What was the growth percentage in Q2?"
        answer = query_engine.answer_question(question)
        
        print(f"\n=== Question: {question} ===")
        print(f"Keywords used: {answer['keywords_used']}")
        
        if answer["results"]:
            print("\nRelevant snippets:")
            for i, result in enumerate(answer["results"]):
                print(f"\nSnippet {i+1} (Page {result['page']}, Relevance: {result['relevance_score']}):")
                print(result["snippet"])
        else:
            print("No relevant information found")
        
        # Close the PDF document
        query_engine.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")
