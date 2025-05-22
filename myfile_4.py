import fitz  # PyMuPDF
import pandas as pd
import re
import os
import logging
from typing import Dict, List, Optional

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
        
        # Keywords to search for each question (simplified versions)
        self.keywords = [
            ["CIN", "Corporate Identity Number", "corporate identity"],
            ["Financial year", "financial statements relates", "FY"],
            ["Product code", "service code", "ITC", "NPCS 4 digit"],
            ["Description", "product category", "service category"],
            ["Turnover", "product turnover", "service turnover", "Rupees"],
            ["Highest turnover", "contributing product", "8 digit code", "ITC", "NPCS"],
            ["Description", "highest contributing", "product description"],
            ["Turnover", "highest contributing", "turnover", "Rupees"]
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

    def search_answer_in_text(self, text: str, keywords: List[str], question_index: int) -> str:
        """Search for answers using keywords in the text."""
        if not text:
            return "Not found"
        
        lines = text.split('\n')
        
        # Search for lines containing any of the keywords
        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue
                
            for keyword in keywords:
                if keyword.lower() in line_lower:
                    # Clean and return the relevant part of the line
                    answer = self._extract_answer_from_line(line, keyword, question_index)
                    if answer and answer != "Not found":
                        return answer
        
        return "Not found"

    def _extract_answer_from_line(self, line: str, keyword: str, question_index: int) -> str:
        """Extract the actual answer from the line containing the keyword."""
        line = line.strip()
        
        # Different extraction strategies based on question type
        if question_index == 0:  # CIN
            # Look for CIN pattern (usually alphanumeric)
            cin_match = re.search(r'[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}', line)
            if cin_match:
                return cin_match.group()
            # Fallback: extract text after keyword
            return self._extract_after_keyword(line, keyword)
            
        elif question_index == 1:  # Financial Year
            # Look for year patterns
            year_match = re.search(r'20\d{2}-\d{2}|20\d{2}', line)
            if year_match:
                return year_match.group()
            return self._extract_after_keyword(line, keyword)
            
        elif question_index in [2, 5]:  # Product/Service codes
            # Look for 4 or 8 digit codes
            if question_index == 2:
                code_match = re.search(r'\b\d{4}\b', line)
            else:
                code_match = re.search(r'\b\d{8}\b', line)
            if code_match:
                return code_match.group()
            return self._extract_after_keyword(line, keyword)
            
        elif question_index in [4, 7]:  # Turnover amounts
            # Look for monetary values
            amount_match = re.search(r'[\d,]+\.?\d*\s*(?:crore|lakh|rupees?|rs\.?|₹)', line, re.IGNORECASE)
            if amount_match:
                return amount_match.group()
            # Look for just numbers
            number_match = re.search(r'[\d,]+\.?\d*', line)
            if number_match:
                return number_match.group()
            return self._extract_after_keyword(line, keyword)
            
        else:  # Description fields
            # Return the line after cleaning
            return self._extract_after_keyword(line, keyword)

    def _extract_after_keyword(self, line: str, keyword: str) -> str:
        """Extract text that appears after the keyword in the line."""
        line_lower = line.lower()
        keyword_lower = keyword.lower()
        
        # Find keyword position and extract what comes after
        keyword_pos = line_lower.find(keyword_lower)
        if keyword_pos != -1:
            after_keyword = line[keyword_pos + len(keyword):].strip()
            # Remove common separators
            after_keyword = re.sub(r'^[:\-\s]+', '', after_keyword)
            if after_keyword:
                return after_keyword
        
        # If no clear separation, return the whole line cleaned
        return re.sub(r'[^\w\s\d.,()-]', '', line).strip()

    def process_single_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF and extract all required information."""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from relevant pages
        page_texts = self.extract_text_from_pdf(pdf_path)
        
        results = {}
        
        # Process each question
        for i, question in enumerate(self.questions):
            target_page = self.page_mapping[i]
            page_text = page_texts.get(target_page, "")
            
            # Search for answer using keywords
            answer = self.search_answer_in_text(page_text, self.keywords[i], i)
            results[question] = answer
            
            logger.info(f"Question {i+1}: {answer}")
        
        return results

    def process_multiple_pdfs(self, pdf_folder: str, output_excel: str):
        """Process multiple PDFs and save results to Excel."""
        if not os.path.exists(pdf_folder):
            logger.error(f"Folder {pdf_folder} does not exist")
            return
        
        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_folder}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        all_results = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                results = self.process_single_pdf(pdf_path)
                results['PDF_File'] = pdf_file  # Add filename for reference
                all_results.append(results)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                # Add empty results for failed PDFs
                empty_results = {question: "Error" for question in self.questions}
                empty_results['PDF_File'] = pdf_file
                all_results.append(empty_results)
        
        # Create DataFrame and save to Excel
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Reorder columns to have PDF_File first, then questions
            columns = ['PDF_File'] + self.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            logger.info(f"Results saved to {output_excel}")
            
            # Display summary
            self.display_summary(df)
        else:
            logger.error("No results to save")

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
    
    # Initialize the extractor
    extractor = PDFDataExtractor()
    
    # Configuration
    PDF_FOLDER = "pdfs"  # Change this to your PDF folder path
    OUTPUT_EXCEL = "extracted_data.xlsx"
    
    print("PDF Data Extraction Pipeline")
    print("="*50)
    
    # Create sample folder structure if it doesn't exist
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created folder: {PDF_FOLDER}")
        print(f"Please place your PDF files in the '{PDF_FOLDER}' folder and run again.")
        return
    
    # Process all PDFs
    extractor.process_multiple_pdfs(PDF_FOLDER, OUTPUT_EXCEL)


# Example usage for single PDF
def process_single_pdf_example():
    """Example of how to process a single PDF file."""
    extractor = PDFDataExtractor()
    
    # Replace with your PDF path
    pdf_path = "sample_document.pdf"
    
    if os.path.exists(pdf_path):
        results = extractor.process_single_pdf(pdf_path)
        
        # Display results
        print("\nExtraction Results:")
        print("-" * 50)
        for question, answer in results.items():
            print(f"{question}: {answer}")
    else:
        print(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()
    
    # Uncomment below to test with a single PDF
    # process_single_pdf_example()
