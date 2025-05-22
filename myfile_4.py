import fitz  # PyMuPDF
import pandas as pd
import re
import os
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFDataExtractor:
    def __init__(self):
        # Define the questions/keywords to search for
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
        
        # Page mapping: first 2 questions on page 1, rest on page 10
        self.page_mapping = {
            0: 1, 1: 1,  # Questions 1-2 on page 1
            2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10  # Questions 3-8 on page 10
        }

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from specific pages of a PDF."""
        try:
            doc = fitz.open(pdf_path)
            page_texts = {}
            
            # Extract text from page 1 and page 10
            for page_num in [1, 10]:
                if page_num <= len(doc):
                    page = doc.load_page(page_num - 1)  # PyMuPDF uses 0-based indexing
                    page_texts[page_num] = page.get_text()
                else:
                    logger.warning(f"Page {page_num} not found in {pdf_path}")
                    page_texts[page_num] = ""
            
            doc.close()
            return page_texts
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return {}

    def debug_text_structure(self, pdf_path: str):
        """Debug function to analyze and understand the text structure."""
        print(f"\n{'='*80}")
        print(f"DEBUGGING TEXT STRUCTURE FOR: {pdf_path}")
        print(f"{'='*80}")
        
        page_texts = self.extract_text_from_pdf(pdf_path)
        
        for page_num, text in page_texts.items():
            print(f"\n--- PAGE {page_num} CONTENT ---")
            print(f"Total characters: {len(text)}")
            print(f"Total lines: {len(text.split(chr(10)))}")
            
            lines = text.split('\n')
            print(f"\nFirst 20 lines of page {page_num}:")
            for i, line in enumerate(lines[:20]):
                if line.strip():
                    print(f"{i+1:2d}: {line.strip()}")
            
            if len(lines) > 20:
                print(f"\n... (showing first 20 of {len(lines)} lines)")
                print(f"\nLast 10 lines of page {page_num}:")
                for i, line in enumerate(lines[-10:], len(lines)-9):
                    if line.strip():
                        print(f"{i:2d}: {line.strip()}")
            
            # Search for potential patterns
            print(f"\n--- PATTERN ANALYSIS FOR PAGE {page_num} ---")
            
            # Look for CIN patterns
            cin_patterns = re.findall(r'[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}', text)
            if cin_patterns:
                print(f"CIN patterns found: {cin_patterns}")
            
            # Look for year patterns
            year_patterns = re.findall(r'20\d{2}-\d{2}|20\d{2}', text)
            if year_patterns:
                print(f"Year patterns found: {year_patterns}")
            
            # Look for 4-digit codes
            four_digit = re.findall(r'\b\d{4}\b', text)
            if four_digit:
                print(f"4-digit codes found: {four_digit[:10]}...")  # Show first 10
            
            # Look for 8-digit codes
            eight_digit = re.findall(r'\b\d{8}\b', text)
            if eight_digit:
                print(f"8-digit codes found: {eight_digit}")
            
            # Look for rupee amounts
            rupee_patterns = re.findall(r'[\d,]+\.?\d*\s*(?:crore|lakh|rupees?|rs\.?|₹)', text, re.IGNORECASE)
            if rupee_patterns:
                print(f"Rupee amounts found: {rupee_patterns}")
            
            # Look for tables or structured data
            if page_num == 10:
                print(f"\n--- SEARCHING FOR TABLE STRUCTURES ON PAGE {page_num} ---")
                table_lines = []
                for line in lines:
                    # Look for lines that might be table headers or contain structured data
                    if any(keyword in line.lower() for keyword in ['product', 'service', 'code', 'description', 'turnover', 'category']):
                        table_lines.append(line.strip())
                
                if table_lines:
                    print("Potential table/structured content:")
                    for i, line in enumerate(table_lines[:15]):  # Show first 15 relevant lines
                        print(f"  {i+1}: {line}")

    def find_answer_context(self, text: str, question_index: int) -> Tuple[str, str]:
        """Find the context and exact location where the answer might be."""
        lines = text.split('\n')
        
        # Define search strategies for each question
        search_strategies = {
            0: {  # CIN
                'keywords': ['cin', 'corporate identity number', 'corporate identity'],
                'patterns': [r'[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}'],
                'context_lines': 3
            },
            1: {  # Financial Year
                'keywords': ['financial year', 'financial statements', 'fy', 'year ended'],
                'patterns': [r'20\d{2}-\d{2}', r'20\d{2}'],
                'context_lines': 2
            },
            2: {  # Product/Service Code 4-digit
                'keywords': ['product code', 'service code', 'itc', 'npcs', '4 digit'],
                'patterns': [r'\b\d{4}\b'],
                'context_lines': 2
            },
            3: {  # Description of product/service
                'keywords': ['description', 'product category', 'service category'],
                'patterns': [],
                'context_lines': 2
            },
            4: {  # Turnover
                'keywords': ['turnover', 'revenue', 'sales'],
                'patterns': [r'[\d,]+\.?\d*\s*(?:crore|lakh|rupees?|rs\.?|₹)', r'[\d,]+\.?\d*'],
                'context_lines': 2
            },
            5: {  # Highest turnover code 8-digit
                'keywords': ['highest turnover', 'highest contributing', 'maximum', '8 digit'],
                'patterns': [r'\b\d{8}\b'],
                'context_lines': 3
            },
            6: {  # Description of highest contributing
                'keywords': ['highest contributing', 'maximum contributing', 'description'],
                'patterns': [],
                'context_lines': 2
            },
            7: {  # Turnover of highest contributing
                'keywords': ['highest turnover', 'maximum turnover', 'highest contributing'],
                'patterns': [r'[\d,]+\.?\d*\s*(?:crore|lakh|rupees?|rs\.?|₹)', r'[\d,]+\.?\d*'],
                'context_lines': 2
            }
        }
        
        strategy = search_strategies.get(question_index, {})
        keywords = strategy.get('keywords', [])
        patterns = strategy.get('patterns', [])
        context_lines = strategy.get('context_lines', 2)
        
        best_match = ""
        best_context = ""
        
        # First try to find lines with keywords
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            if not line_clean:
                continue
            
            # Check if line contains any of the keywords
            keyword_found = False
            for keyword in keywords:
                if keyword in line_clean:
                    keyword_found = True
                    break
            
            if keyword_found:
                # Get context (surrounding lines)
                start_idx = max(0, i - context_lines)
                end_idx = min(len(lines), i + context_lines + 1)
                context_lines_list = lines[start_idx:end_idx]
                context = '\n'.join([l.strip() for l in context_lines_list if l.strip()])
                
                # Try to extract answer using patterns
                if patterns:
                    for pattern in patterns:
                        matches = re.findall(pattern, context, re.IGNORECASE)
                        if matches:
                            best_match = matches[0] if isinstance(matches[0], str) else str(matches[0])
                            best_context = context
                            return best_match, best_context
                
                # If no pattern match, use the line itself as potential answer
                if not best_match:
                    # Try to extract meaningful part from the line
                    potential_answer = self._extract_meaningful_part(line, keywords[0] if keywords else "")
                    if potential_answer:
                        best_match = potential_answer
                        best_context = context
        
        # If no keyword match, try patterns across entire text
        if not best_match and patterns:
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    best_match = matches[0] if isinstance(matches[0], str) else str(matches[0])
                    best_context = "Pattern found in text"
                    break
        
        return best_match if best_match else "Not found", best_context

    def _extract_meaningful_part(self, line: str, keyword: str) -> str:
        """Extract meaningful part from a line containing the keyword."""
        line = line.strip()
        if not line:
            return ""
        
        # Remove common prefixes and suffixes
        line = re.sub(r'^[•\-\*\d\.]+\s*', '', line)  # Remove bullet points, numbers
        line = re.sub(r'[:\-]\s*$', '', line)  # Remove trailing colons/dashes
        
        # If keyword is found, try to get what comes after it
        if keyword:
            keyword_pos = line.lower().find(keyword.lower())
            if keyword_pos != -1:
                after_keyword = line[keyword_pos + len(keyword):].strip()
                after_keyword = re.sub(r'^[:\-\s]+', '', after_keyword)  # Remove separators
                if after_keyword and len(after_keyword) > 2:
                    return after_keyword
        
        return line

    def process_single_pdf_with_debug(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF with detailed debugging information."""
        logger.info(f"Processing PDF with debug: {pdf_path}")
        
        # First, debug the structure
        self.debug_text_structure(pdf_path)
        
        # Extract text from relevant pages
        page_texts = self.extract_text_from_pdf(pdf_path)
        
        results = {}
        
        print(f"\n{'='*80}")
        print("ANSWER EXTRACTION RESULTS")
        print(f"{'='*80}")
        
        # Process each question
        for i, question in enumerate(self.questions):
            target_page = self.page_mapping[i]
            page_text = page_texts.get(target_page, "")
            
            print(f"\nQuestion {i+1}: {question}")
            print(f"Searching on page: {target_page}")
            
            # Find answer with context
            answer, context = self.find_answer_context(page_text, i)
            results[question] = answer
            
            print(f"Answer found: {answer}")
            if context and answer != "Not found":
                print(f"Context: {context[:200]}...")
            print("-" * 50)
        
        return results

    def process_single_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF and extract all required information."""
        page_texts = self.extract_text_from_pdf(pdf_path)
        results = {}
        
        for i, question in enumerate(self.questions):
            target_page = self.page_mapping[i]
            page_text = page_texts.get(target_page, "")
            answer, _ = self.find_answer_context(page_text, i)
            results[question] = answer
        
        return results

    def process_multiple_pdfs(self, pdf_folder: str, output_excel: str, debug_mode: bool = False):
        """Process multiple PDFs and save results to Excel."""
        if not os.path.exists(pdf_folder):
            logger.error(f"Folder {pdf_folder} does not exist")
            return
        
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_folder}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_results = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                if debug_mode:
                    results = self.process_single_pdf_with_debug(pdf_path)
                else:
                    results = self.process_single_pdf(pdf_path)
                results['PDF_File'] = pdf_file
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                empty_results = {question: "Error" for question in self.questions}
                empty_results['PDF_File'] = pdf_file
                all_results.append(empty_results)
        
        # Save results
        if all_results:
            df = pd.DataFrame(all_results)
            columns = ['PDF_File'] + self.questions
            df = df[columns]
            df.to_excel(output_excel, index=False)
            logger.info(f"Results saved to {output_excel}")
            self.display_summary(df)

    def display_summary(self, df: pd.DataFrame):
        """Display a summary of the extraction results."""
        print("\n" + "="*80)
        print("EXTRACTION SUMMARY")
        print("="*80)
        print(f"Total PDFs processed: {len(df)}")
        
        for question in self.questions:
            found_count = len(df[df[question] != "Not found"])
            print(f"{question[:50]}... : {found_count}/{len(df)} found")
        
        print("="*80)


def main():
    """Main function to run the PDF extraction pipeline."""
    extractor = PDFDataExtractor()
    
    PDF_FOLDER = "pdfs"
    OUTPUT_EXCEL = "extracted_data.xlsx"
    
    print("PDF Data Extraction Pipeline - Enhanced Version")
    print("="*60)
    
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created folder: {PDF_FOLDER}")
        print(f"Please place your PDF files in the '{PDF_FOLDER}' folder and run again.")
        return
    
    # Ask user if they want debug mode
    debug_choice = input("Run in debug mode to see text structure? (y/n): ").lower().strip()
    debug_mode = debug_choice == 'y'
    
    extractor.process_multiple_pdfs(PDF_FOLDER, OUTPUT_EXCEL, debug_mode)


def debug_single_pdf():
    """Function to debug a single PDF file."""
    extractor = PDFDataExtractor()
    
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    if os.path.exists(pdf_path):
        results = extractor.process_single_pdf_with_debug(pdf_path)
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        for question, answer in results.items():
            print(f"{question}: {answer}")
    else:
        print(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Process multiple PDFs")
    print("2. Debug single PDF")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        debug_single_pdf()
    else:
        print("Invalid choice. Running main function.")
        main()
