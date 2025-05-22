import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
import warnings
from typing import Dict, List, Any

# Suppress warnings
warnings.filterwarnings('ignore')

class PDFExtractor:
    """Class for extracting text from PDF files"""
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page of a PDF file"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                pages.append({
                    "page_num": page_num + 1,
                    "text": text
                })
                
            # Explicitly close the document to free memory
            doc.close()
            print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages
        except Exception as e:
            print(f"PDF extraction error after {time.time() - start_time:.2f} seconds: {e}")
            raise

class SimplePatternExtractor:
    """Simple pattern extractor that matches exact questions as keywords"""
    
    def __init__(self):
        # Define the questions and their target pages
        self.questions_config = {
            "Corporate identity number (CIN) of company": {
                "page": 1,
                "patterns": [
                    r'Corporate identity number \(CIN\) of company[:\s]*([A-Z0-9]{21})',
                    r'([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})'
                ]
            },
            "Financial year to which financial statements relates": {
                "page": 1,
                "patterns": [
                    r'Financial year to which financial statements relates[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                    r'([0-9]{4}[-/][0-9]{2,4})'
                ]
            },
            "Product or service category code (ITC/ NPCS 4 digit code)": {
                "page": 10,
                "patterns": [
                    r'Product or service category code \(ITC/ NPCS 4 digit code\)[:\s]*([0-9]{4})',
                    r'([0-9]{4})'
                ]
            },
            "Description of the product or service category": {
                "page": 10,
                "patterns": [
                    r'Description of the product or service category[:\s]*([^\n\r]{10,150})',
                    r'([A-Za-z\s,.-]{10,150})'
                ]
            },
            "Turnover of the product or service category (in Rupees)": {
                "page": 10,
                "patterns": [
                    r'Turnover of the product or service category \(in Rupees\)[:\s]*([0-9,]+)',
                    r'([0-9,]{6,})'
                ]
            },
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": {
                "page": 10,
                "patterns": [
                    r'Highest turnover contributing product or service code \(ITC/ NPCS 8 digit code\)[:\s]*([0-9]{8})',
                    r'([0-9]{8})'
                ]
            },
            "Description of the product or service": {
                "page": 10,
                "patterns": [
                    r'Description of the product or service[:\s]*([^\n\r]{10,150})',
                    r'([A-Za-z\s,.-]{10,150})'
                ]
            },
            "Turnover of highest contributing product or service (in Rupees)": {
                "page": 10,
                "patterns": [
                    r'Turnover of highest contributing product or service \(in Rupees\)[:\s]*([0-9,]+)',
                    r'([0-9,]{6,})'
                ]
            }
        }
    
    def extract_answer_for_question(self, question: str, pages: List[Dict[str, Any]]) -> str:
        """Extract answer for a specific question"""
        start_time = time.time()
        
        if question not in self.questions_config:
            return "Question not configured"
        
        config = self.questions_config[question]
        target_page = config["page"]
        patterns = config["patterns"]
        
        # Get the target page
        if target_page > len(pages):
            print(f"Target page {target_page} not found, total pages: {len(pages)}")
            return "Target page not found"
        
        page_text = pages[target_page - 1]["text"]
        
        # Special handling for the last column (highest turnover)
        if "Turnover of highest contributing product or service" in question:
            return self._extract_highest_turnover(page_text)
        
        # Try patterns in order
        for pattern in patterns:
            matches = re.finditer(pattern, page_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                result = match.group(1).strip()
                if result and len(result) > 0:
                    cleaned_result = self._clean_result(question, result)
                    if self._validate_result(question, cleaned_result):
                        print(f"Found answer for '{question[:30]}...' in {time.time() - start_time:.2f} seconds")
                        return cleaned_result
        
        print(f"No answer found for '{question[:30]}...' in {time.time() - start_time:.2f} seconds")
        return "Information not found"
    
    def _extract_highest_turnover(self, page_text: str) -> str:
        """Special extraction for highest turnover - search for the exact question and get the number after it"""
        
        # Strategy 1: Look for the exact question followed by a number
        exact_pattern = r'Turnover of highest contributing product or service \(in Rupees\)[\s\n\r]*([0-9,]+)'
        match = re.search(exact_pattern, page_text, re.IGNORECASE)
        if match:
            result = match.group(1).replace(',', '')
            if result.isdigit() and len(result) >= 6:
                return result
        
        # Strategy 2: Look for the question anywhere and find the closest number
        question_pattern = r'Turnover of highest contributing product or service'
        question_match = re.search(question_pattern, page_text, re.IGNORECASE)
        if question_match:
            # Get text after the question (next 200 characters)
            start_pos = question_match.end()
            text_after = page_text[start_pos:start_pos + 200]
            
            # Find the first substantial number
            number_match = re.search(r'([0-9,]{6,})', text_after)
            if number_match:
                result = number_match.group(1).replace(',', '')
                if result.isdigit() and len(result) >= 6:
                    return result
        
        # Strategy 3: Look for any large number in the page (fallback)
        all_numbers = re.findall(r'([0-9,]{8,})', page_text)
        for num in all_numbers:
            cleaned = num.replace(',', '')
            if cleaned.isdigit() and 6 <= len(cleaned) <= 12:
                # Return the first reasonable number found
                return cleaned
        
        return "Information not found"
    
    def _clean_result(self, question: str, result: str) -> str:
        """Clean the extracted result"""
        # Remove extra whitespace
        result = result.strip()
        
        # Question-specific cleaning
        if "CIN" in question:
            # Keep only alphanumeric for CIN
            result = re.sub(r'[^A-Z0-9]', '', result.upper())
        
        elif "code" in question.lower():
            # Keep only digits for codes
            result = re.sub(r'[^0-9]', '', result)
        
        elif "turnover" in question.lower():
            # Keep only digits for turnover (remove commas, currency symbols)
            result = re.sub(r'[^0-9]', '', result)
        
        elif "description" in question.lower():
            # Clean descriptions
            result = re.sub(r'[^\w\s\-.,]', '', result)
            result = result[:100]  # Limit length
        
        return result
    
    def _validate_result(self, question: str, result: str) -> bool:
        """Validate the extracted result"""
        if not result or len(result.strip()) == 0:
            return False
        
        # Question-specific validation
        if "CIN" in question:
            return len(result) == 21 and result[0] in ['L', 'U']
        
        elif "4 digit code" in question:
            return len(result) == 4 and result.isdigit()
        
        elif "8 digit code" in question:
            return len(result) == 8 and result.isdigit()
        
        elif "turnover" in question.lower():
            return result.isdigit() and len(result) >= 6
        
        elif "description" in question.lower():
            return 5 <= len(result) <= 150
        
        elif "financial year" in question.lower():
            return bool(re.search(r'\d{4}', result))
        
        return True

class FastFormExtractor:
    """Fast extractor for standardized forms"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.pattern_extractor = SimplePatternExtractor()
        
        # Define the questions in order
        self.questions = [
            "Corporate identity number (CIN) of company",
            "Financial year to which financial statements relates",
            "Product or service category code (ITC/ NPCS 4 digit code)",
            "Description of the product or service category",
            "Turnover of the product or service category (in Rupees)",
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
            "Description of the product or service",
            "Turnover of highest contributing product or service (in Rupees)"
        ]
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a PDF file using simple pattern matching"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            # Process each question
            answers = {}
            for question in self.questions:
                answer = self.pattern_extractor.extract_answer_for_question(question, pages)
                answers[question] = answer
                print(f"  {question[:50]}... -> {answer}")
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """Process multiple PDF files and save results to an Excel file"""
        batch_start_time = time.time()
        print(f"Processing all PDFs in directory: {pdf_dir}")
        
        if not os.path.isdir(pdf_dir):
            print(f"Directory not found: {pdf_dir}")
            return
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        results = []
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"\nProcessing {i+1}/{len(pdf_files)}: {pdf_file}")
            
            answers = self.process_pdf(pdf_path)
            result = {"PDF Filename": pdf_file}
            result.update(answers)
            results.append(result)
        
        # Save to Excel
        if results:
            df = pd.DataFrame(results)
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            df.to_excel(output_excel, index=False)
            print(f"\nResults saved to {output_excel}")
        
        print(f"Total processing completed in {time.time() - batch_start_time:.2f} seconds")
        print(f"Average time per PDF: {(time.time() - batch_start_time) / len(pdf_files):.2f} seconds")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple Form-Based PDF Extractor")
    parser.add_argument("--pdf_dir", type=str, default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="simple_extraction_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF")
    args = parser.parse_args()
    
    try:
        print("Starting Simple Form-Based PDF Extraction")
        pipeline = FastFormExtractor()
        
        if args.single_pdf:
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
            
            answers = pipeline.process_pdf(args.single_pdf)
            
            # Save single PDF results
            result = {"PDF Filename": os.path.basename(args.single_pdf)}
            result.update(answers)
            df = pd.DataFrame([result])
            columns = ["PDF Filename"] + pipeline.questions
            df = df[columns]
            df.to_excel(args.output, index=False)
            print(f"Results saved to {args.output}")
        else:
            pipeline.process_pdfs_batch(args.pdf_dir, args.output)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
