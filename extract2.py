import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
import warnings
from typing import Dict, List, Any, Optional

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

class PatternBasedExtractor:
    """Class for extracting regulatory information using pattern matching"""
    
    def __init__(self):
        # Define extraction patterns for each question
        self.patterns = {
            "Corporate identity number (CIN) of company": [
                r'CIN[:\s]*([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
                r'Corporate Identity Number[:\s]*([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
                r'([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
            ],
            "Financial year to which financial statements relates": [
                r'Financial Year[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'FY[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'for the year ended[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'([0-9]{4}[-/][0-9]{2,4})',
            ],
            "Product or service category code (ITC/ NPCS 4 digit code)": [
                r'ITC[:\s]*(\d{4})',
                r'NPCS[:\s]*(\d{4})',
                r'Product.*code[:\s]*(\d{4})',
                r'Service.*code[:\s]*(\d{4})',
                r'Category.*code[:\s]*(\d{4})',
            ],
            "Description of the product or service category": [
                r'Product category[:\s]*([^\n\r]{10,100})',
                r'Service category[:\s]*([^\n\r]{10,100})',
                r'Business segment[:\s]*([^\n\r]{10,100})',
                r'Main business[:\s]*([^\n\r]{10,100})',
            ],
            "Turnover of the product or service category (in Rupees)": [
                r'Category turnover[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'Segment turnover[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'Product category.*turnover[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'Service category.*turnover[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
            ],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": [
                r'Highest.*code[:\s]*(\d{8})',
                r'Main product.*code[:\s]*(\d{8})',
                r'Primary.*code[:\s]*(\d{8})',
                r'8 digit.*code[:\s]*(\d{8})',
            ],
            "Description of the product or service": [
                r'Product description[:\s]*([^\n\r]{10,150})',
                r'Service description[:\s]*([^\n\r]{10,150})',
                r'Main product[:\s]*([^\n\r]{10,150})',
                r'Primary service[:\s]*([^\n\r]{10,150})',
            ],
            "Turnover of highest contributing product or service (in Rupees)": [
                # Table-based patterns - most common in forms
                r'Turnover of highest contributing product or service \(in Rupees\)[:\s]*([0-9,]+)',
                r'Turnover of highest contributing product or service[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                
                # Variations with line breaks and formatting
                r'Turnover of highest[\s\n]*contributing[\s\n]*product or service[\s\n]*\(in Rupees\)[\s\n]*([0-9,]+)',
                r'Turnover of highest[\s\n]*contributing[\s\n]*product or service[\s\n]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                
                # Shortened versions
                r'Highest contributing.*turnover[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'Highest turnover[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                
                # Table cell patterns (common in structured forms)
                r'(?:Rs\.?|₹)?\s*([0-9,]{6,})\s*(?=\n|$)',  # Large numbers on their own line
                r'\|\s*([0-9,]{6,})\s*\|',  # Numbers in table cells
                r'\s([0-9]{6,})\s',  # 6+ digit numbers with spaces around them
                
                # Financial statement patterns
                r'Revenue from operations[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'Sale of products[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'Total revenue[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                r'Net sales[:\s]*(?:Rs\.?|₹)?\s*([0-9,]+)',
                
                # Generic large number patterns (as fallback)
                r'([0-9]{8,})',  # Any 8+ digit number
                r'([0-9,]{10,})',  # Any 10+ character number with commas
            ]
        }
        
        # Define which pages to search for each question
        self.page_preferences = {
            "Corporate identity number (CIN) of company": [1, 2, 3],  # Usually on first few pages
            "Financial year to which financial statements relates": [1, 2, 3],  # Usually on first few pages
            "Product or service category code (ITC/ NPCS 4 digit code)": [10, 11, 12, 9, 8],  # Around page 10
            "Description of the product or service category": [10, 11, 12, 9, 8],
            "Turnover of the product or service category (in Rupees)": [10, 11, 12, 9, 8],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": [10, 11, 12, 9, 8],
            "Description of the product or service": [10, 11, 12, 9, 8],
            "Turnover of highest contributing product or service (in Rupees)": [10, 11, 12, 13, 9, 8, 7, 6]  # Expanded search for turnover
        }
    
    def extract_with_patterns(self, question: str, pages: List[Dict[str, Any]]) -> str:
        """Extract answer using pattern matching"""
        start_time = time.time()
        
        # Get patterns for this question
        question_patterns = self.patterns.get(question, [])
        if not question_patterns:
            return "No patterns defined for this question"
        
        # Get preferred pages for this question
        preferred_pages = self.page_preferences.get(question, [1, 2, 3])
        
        # Special handling for the problematic last column
        if "Turnover of highest contributing product or service" in question:
            return self._extract_highest_turnover_special(pages)
        
        # Search in preferred pages first
        for page_num in preferred_pages:
            if page_num <= len(pages):
                page_text = pages[page_num - 1]["text"]
                
                # Try each pattern
                for pattern in question_patterns:
                    matches = re.finditer(pattern, page_text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        result = match.group(1).strip()
                        if result and len(result) > 0:
                            # Clean up the result
                            cleaned_result = self._clean_extracted_text(question, result)
                            if self._validate_result(question, cleaned_result):
                                print(f"Pattern extraction for '{question[:30]}...' took {time.time() - start_time:.2f} seconds")
                                return cleaned_result
        
        # If not found in preferred pages, search all pages
        for page in pages:
            page_text = page["text"]
            
            # Try each pattern
            for pattern in question_patterns:
                matches = re.finditer(pattern, page_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    result = match.group(1).strip()
                    if result and len(result) > 0:
                        # Clean up the result
                        cleaned_result = self._clean_extracted_text(question, result)
                        if self._validate_result(question, cleaned_result):
                            print(f"Pattern extraction for '{question[:30]}...' took {time.time() - start_time:.2f} seconds")
                            return cleaned_result
        
        print(f"Pattern extraction failed for '{question[:30]}...' after {time.time() - start_time:.2f} seconds")
        return "Information not found in the provided context"
    
    def _clean_extracted_text(self, question: str, text: str) -> str:
        """Clean extracted text based on question type"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Question-specific cleaning
        if "CIN" in question:
            # CIN should be alphanumeric only
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        elif "financial year" in question.lower():
            # Clean financial year format
            text = re.sub(r'[^\d\-/]', '', text)
        
        elif "code" in question.lower():
            # Codes should be numeric only
            text = re.sub(r'[^\d]', '', text)
        
        elif "turnover" in question.lower() or "in Rupees" in question:
            # Remove currency symbols and clean numbers
            text = re.sub(r'[^\d,]', '', text)
            text = text.replace(',', '')  # Remove commas for consistency
        
        elif "description" in question.lower():
            # Clean descriptions - remove excessive punctuation
            text = re.sub(r'[^\w\s\-.,]', '', text)
            text = text[:100]  # Limit length
        
        return text.strip()
    
    def _validate_result(self, question: str, result: str) -> bool:
        """Validate extracted result based on question type"""
        if not result or len(result.strip()) == 0:
            return False
        
        # Question-specific validation
        if "CIN" in question:
            # CIN should be exactly 21 characters, starting with L or U
            return len(result) == 21 and result[0] in ['L', 'U']
        
        elif "financial year" in question.lower():
            # Should contain year pattern
            return bool(re.search(r'\d{4}', result))
        
        elif "4 digit code" in question:
            # Should be exactly 4 digits
            return len(result) == 4 and result.isdigit()
        
        elif "8 digit code" in question:
            # Should be exactly 8 digits
            return len(result) == 8 and result.isdigit()
        
        elif "turnover" in question.lower():
            # Special handling for turnover amounts
            if "highest" in question.lower():
                # For highest turnover, be more lenient with validation
                # Accept any number with 6+ digits (reasonable turnover amount)
                cleaned_digits = re.sub(r'[^\d]', '', result)
                return len(cleaned_digits) >= 6 and cleaned_digits.isdigit()
            else:
                # Regular turnover validation
                return result.isdigit() and len(result) > 0
        
        elif "description" in question.lower():
            # Should have reasonable length
            return 5 <= len(result) <= 150
        
        return True
    
    def _extract_highest_turnover_special(self, pages: List[Dict[str, Any]]) -> str:
        """Special extraction method for highest turnover with aggressive pattern matching"""
        
        # Extended page search for turnover data
        search_pages = [10, 11, 12, 13, 9, 8, 7, 6, 14, 15]
        
        # Multiple extraction strategies
        strategies = [
            # Strategy 1: Look for exact question match with number following
            r'Turnover of highest contributing product or service.*?([0-9,]{6,})',
            
            # Strategy 2: Look for the question in a table format
            r'Turnover of highest[\s\S]*?([0-9,]{6,})',
            
            # Strategy 3: Look for large numbers near "highest" or "turnover"
            r'(?:highest|turnover)[\s\S]{0,100}?([0-9,]{8,})',
            
            # Strategy 4: Look for numbers in specific ranges (typical turnover amounts)
            r'([0-9,]{8,})',  # Any 8+ digit number
            
            # Strategy 5: Look for numbers with specific formatting patterns
            r'(\d{1,3},\d{2,3},\d{2,3},\d{3})',  # Pattern like 12,34,56,789
            r'(\d{3,},\d{3,},\d{3,})',  # Pattern like 123,456,789
        ]
        
        for page_num in search_pages:
            if page_num <= len(pages):
                page_text = pages[page_num - 1]["text"]
                
                # Try each strategy
                for strategy in strategies:
                    matches = re.finditer(strategy, page_text, re.IGNORECASE | re.DOTALL)
                    
                    for match in matches:
                        candidate = match.group(1).strip()
                        # Clean the candidate
                        cleaned = re.sub(r'[^\d,]', '', candidate)
                        numbers_only = cleaned.replace(',', '')
                        
                        # Validate: should be a reasonable turnover amount (6+ digits)
                        if numbers_only.isdigit() and len(numbers_only) >= 6:
                            # Additional validation: not too large (reasonable business turnover)
                            try:
                                amount = int(numbers_only)
                                if 100000 <= amount <= 999999999999:  # Between 1 lakh and 99,999 crores
                                    return numbers_only
                            except ValueError:
                                continue
        
        # If no specific match found, look for the largest number in the target pages
        all_numbers = []
        for page_num in search_pages:
            if page_num <= len(pages):
                page_text = pages[page_num - 1]["text"]
                # Find all numbers with 6+ digits
                numbers = re.findall(r'([0-9,]{6,})', page_text)
                for num in numbers:
                    cleaned_num = num.replace(',', '')
                    if cleaned_num.isdigit() and len(cleaned_num) >= 6:
                        try:
                            amount = int(cleaned_num)
                            if 100000 <= amount <= 999999999999:
                                all_numbers.append((amount, cleaned_num))
                        except ValueError:
                            continue
        
        # Return the largest reasonable number found
        if all_numbers:
            all_numbers.sort(reverse=True)
            return all_numbers[0][1]  # Return the string version of the largest number
        
        return "Information not found in the provided context"

class FastPDFRegulatoryExtractor:
    """Fast pattern-based extractor for standardized forms"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.pattern_extractor = PatternBasedExtractor()
        
        # Define the specific regulatory questions to extract
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
        """Process a PDF file using pattern matching"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            extract_start = time.time()
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            print(f"PDF text extraction completed in {time.time() - extract_start:.2f} seconds")
            
            # Create a dictionary to store answers
            answers = {}
            
            # Process each question using pattern matching
            for question in self.questions:
                question_start = time.time()
                print(f"Processing question: {question[:50]}...")
                
                # Extract using patterns
                answer = self.pattern_extractor.extract_with_patterns(question, pages)
                answers[question] = answer
                
                print(f"Question processed in {time.time() - question_start:.2f} seconds")
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return error message for all questions
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """Process multiple PDF files and save results to an Excel file"""
        batch_start_time = time.time()
        print(f"Processing all PDFs in directory: {pdf_dir}")
        
        # Check if directory exists
        if not os.path.isdir(pdf_dir):
            print(f"Directory not found: {pdf_dir}")
            return
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF and collect results
        results = []
        
        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                # Get answers for this PDF
                pdf_start_time = time.time()
                answers = self.process_pdf(pdf_path)
                
                # Add filename to results
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                
                results.append(result)
                print(f"Processed {pdf_file} ({i+1}/{len(pdf_files)}) in {time.time() - pdf_start_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                # Add error entry
                result = {"PDF Filename": pdf_file}
                result.update({question: f"Error: {str(e)}" for question in self.questions})
                results.append(result)
        
        # Create a DataFrame and save to Excel
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Reorder columns to have PDF Filename first, then questions
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
        else:
            print("No results to save")
        
        print(f"Total batch processing completed in {time.time() - batch_start_time:.2f} seconds")
        print(f"Average time per PDF: {(time.time() - batch_start_time) / len(pdf_files):.2f} seconds")

def main():
    # Simple command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description="Fast Pattern-Based PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="fast_regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    args = parser.parse_args()
    
    try:
        print("Starting Fast Pattern-Based PDF Regulatory Information Extraction")
        start_time = time.time()
        
        # Create pipeline
        pipeline = FastPDFRegulatoryExtractor()
        
        # Process single PDF or directory
        if args.single_pdf:
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
                
            # Process single PDF
            results = []
            pdf_file = os.path.basename(args.single_pdf)
            
            try:
                # Process the PDF
                answers = pipeline.process_pdf(args.single_pdf)
                
                # Add to results
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                results.append(result)
                
                # Create a DataFrame and save to Excel
                df = pd.DataFrame(results)
                columns = ["PDF Filename"] + pipeline.questions
                df = df[columns]
                df.to_excel(args.output, index=False)
                print(f"Results saved to {args.output}")
                
            except Exception as e:
                print(f"Error processing {args.single_pdf}: {e}")
        else:
            # Process all PDFs in directory
            pipeline.process_pdfs_batch(args.pdf_dir, args.output)
        
        print(f"Processing completed successfully in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
