import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Any
from datetime import datetime

class PDFExtractor:
    """Class for extracting text from specific PDF pages"""
    
    def extract_text_from_specific_pages(self, pdf_path: str, page_numbers: List[int]) -> Dict[int, str]:
        """Extract text from specific pages of a PDF file"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            pages_text = {}
            
            for page_num in page_numbers:
                if page_num <= len(doc):
                    page = doc[page_num - 1]  # PyMuPDF uses 0-based indexing
                    text = page.get_text()
                    pages_text[page_num] = text
                else:
                    pages_text[page_num] = ""
                    
            doc.close()
            print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages_text
        except Exception as e:
            print(f"PDF extraction error after {time.time() - start_time:.2f} seconds: {e}")
            raise

class RegulatoryInfoExtractor:
    """Class for extracting regulatory information using pattern matching"""
    
    def __init__(self):
        # Define extraction patterns for each question
        self.patterns = {
            "cin": [
                r'(?:CIN|Corporate\s+Identity\s+Number)[:\s]*([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
                r'([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
            ],
            "financial_year": [
                r'(?:Financial\s+Year|FY|Year\s+ended)[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'(?:for\s+the\s+year\s+ended|period\s+ended)[:\s]*([0-9]{1,2}[a-zA-Z]*\s+[a-zA-Z]+[,\s]+[0-9]{4})',
                r'(?:Annual\s+Report)[^0-9]*([0-9]{4}[-/][0-9]{2,4})',
                r'([0-9]{4}[-/][0-9]{2,4})',
            ],
            "product_code_4": [
                r'(?:ITC|NPCS|Product\s+Code)[:\s]*([0-9]{4})',
                r'(?:4\s+digit\s+code)[:\s]*([0-9]{4})',
                r'\b([0-9]{4})\b',
            ],
            "product_code_8": [
                r'(?:ITC|NPCS|Product\s+Code)[:\s]*([0-9]{8})',
                r'(?:8\s+digit\s+code)[:\s]*([0-9]{8})',
                r'\b([0-9]{8})\b',
            ],
            "description": [
                r'(?:Description|Business|Activity|Product|Service)[:\s]*([A-Za-z][^.]*)',
                r'(?:engaged\s+in)[:\s]*([A-Za-z][^.]*)',
                r'(?:business\s+of)[:\s]*([A-Za-z][^.]*)',
            ],
            "turnover": [
                # Pattern for amounts in various formats
                r'(?:Turnover|Revenue|Sales)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:lakhs?|crores?|millions?|thousands?)?',
                r'(?:Rs\.?|₹|INR)\s*([0-9,]+(?:\.[0-9]+)?)',
                r'\b([0-9]{6,})\b',  # Large numbers (6+ digits)
                r'([0-9,]+(?:\.[0-9]+)?)\s*(?:lakhs?|crores?)',
            ]
        }
        
        # Questions to extract
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
    
    def extract_cin(self, text: str) -> str:
        """Extract Corporate Identity Number"""
        for pattern in self.patterns["cin"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        return "Information not found"
    
    def extract_financial_year(self, text: str) -> str:
        """Extract Financial Year with full date format"""
        # First try to find explicit financial year mentions
        fy_patterns = [
            r'(?:Financial\s+Year|FY)[:\s]*([0-9]{4}[-/][0-9]{2,4})',
            r'(?:for\s+the\s+year\s+ended)[:\s]*([0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+[,\s]+[0-9]{4})',
            r'(?:Annual\s+Report)[^0-9]*([0-9]{4}[-/][0-9]{2,4})',
        ]
        
        for pattern in fy_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fy = match.group(1)
                # Convert to standard format if it's a full date
                if any(month in fy.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                        'july', 'august', 'september', 'october', 'november', 'december']):
                    return fy  # Return full date as found
                return fy
        
        # Fallback: look for year patterns
        year_match = re.search(r'([0-9]{4}[-/][0-9]{2,4})', text)
        if year_match:
            return year_match.group(1)
            
        return "Information not found"
    
    def extract_product_code(self, text: str, digits: int = 4) -> str:
        """Extract product/service code"""
        pattern_key = f"product_code_{digits}"
        if pattern_key in self.patterns:
            for pattern in self.patterns[pattern_key]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    code = match.group(1) if len(match.groups()) > 0 else match.group(0)
                    if len(code) == digits:
                        return code
        return "Information not found"
    
    def extract_description(self, text: str) -> str:
        """Extract business/product description"""
        for pattern in self.patterns["description"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                desc = match.group(1).strip()
                # Clean up the description
                desc = re.sub(r'[^\w\s\-.,]', '', desc)
                if len(desc) > 10:  # Ensure it's a meaningful description
                    return desc[:200]  # Limit length
        return "Information not found"
    
    def extract_turnover(self, text: str, context: str = "") -> str:
        """Extract turnover/revenue information with enhanced pattern matching"""
        # Enhanced patterns for turnover extraction
        turnover_patterns = [
            # Direct turnover mentions with amounts
            r'(?:Turnover|Revenue|Sales)(?:\s+of)?(?:\s+highest)?(?:\s+contributing)?(?:\s+product)?(?:\s+or)?(?:\s+service)?[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:lakhs?|crores?|millions?|thousands?)?',
            
            # Segment reporting patterns
            r'(?:Segment|Product|Service)\s+(?:Revenue|Turnover)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+(?:\.[0-9]+)?)',
            
            # Revenue from operations
            r'(?:Revenue\s+from\s+operations)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+(?:\.[0-9]+)?)',
            
            # Sale of products/goods
            r'(?:Sale\s+of\s+(?:products|goods))[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+(?:\.[0-9]+)?)',
            
            # Table format - look for large numbers that could be turnover
            r'(?:Rs\.?|₹|INR)\s*([0-9,]+(?:\.[0-9]+)?)',
            
            # Large standalone numbers (likely in lakhs/crores)
            r'\b([0-9]{6,})\b',  # 6+ digit numbers
            
            # Numbers with units
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:lakhs?|crores?)',
        ]
        
        # Look for turnover in different contexts
        found_amounts = []
        
        for pattern in turnover_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount = match.group(1)
                # Clean the amount
                amount_clean = amount.replace(',', '')
                try:
                    amount_num = float(amount_clean)
                    # Filter reasonable turnover amounts (avoid small numbers like years)
                    if amount_num > 1000:  # Minimum threshold
                        found_amounts.append({
                            'amount': amount,
                            'numeric': amount_num,
                            'context': match.group(0)
                        })
                except ValueError:
                    continue
        
        # If we found amounts, return the largest one (likely the main turnover)
        if found_amounts:
            # Sort by numeric value and return the largest
            found_amounts.sort(key=lambda x: x['numeric'], reverse=True)
            return found_amounts[0]['amount'].replace(',', '')
        
        return "Information not found"
    
    def process_pdf_content(self, pages_text: Dict[int, str]) -> Dict[str, str]:
        """Process PDF content and extract all regulatory information"""
        results = {}
        
        # Page 1 content (for CIN and Financial Year)
        page1_text = pages_text.get(1, "")
        
        # Page 10 content (for other information)  
        page10_text = pages_text.get(10, "")
        
        # Extract information based on question mapping
        results[self.questions[0]] = self.extract_cin(page1_text)
        results[self.questions[1]] = self.extract_financial_year(page1_text)
        results[self.questions[2]] = self.extract_product_code(page10_text, 4)
        results[self.questions[3]] = self.extract_description(page10_text)
        results[self.questions[4]] = self.extract_turnover(page10_text, "category")
        results[self.questions[5]] = self.extract_product_code(page10_text, 8)
        results[self.questions[6]] = self.extract_description(page10_text)
        results[self.questions[7]] = self.extract_turnover(page10_text, "highest")
        
        return results

class PDFRegulatoryPipeline:
    """Main pipeline for processing PDFs without LLM"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.info_extractor = RegulatoryInfoExtractor()
        self.required_pages = [1, 10]  # Pages we need to extract
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF file"""
        start_time = time.time()
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        
        try:
            # Extract text from required pages only
            pages_text = self.pdf_extractor.extract_text_from_specific_pages(pdf_path, self.required_pages)
            
            # Extract regulatory information
            results = self.info_extractor.process_pdf_content(pages_text)
            
            print(f"Completed in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return error for all questions
            return {question: f"Error: {str(e)}" for question in self.info_extractor.questions}
    
    def process_pdf_directory(self, pdf_dir: str, output_excel: str):
        """Process all PDFs in a directory"""
        batch_start_time = time.time()
        print(f"Processing all PDFs in directory: {pdf_dir}")
        
        # Check if directory exists
        if not os.path.isdir(pdf_dir):
            print(f"Directory not found: {pdf_dir}")
            return
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        results = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                # Process the PDF
                answers = self.process_single_pdf(pdf_path)
                
                # Add filename to results
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                # Add error entry
                result = {"PDF Filename": pdf_file}
                result.update({question: f"Error: {str(e)}" for question in self.info_extractor.questions})
                results.append(result)
        
        # Save results to Excel
        if results:
            df = pd.DataFrame(results)
            
            # Reorder columns
            columns = ["PDF Filename"] + self.info_extractor.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            
            # Print summary
            print(f"\nSummary:")
            print(f"Total PDFs processed: {len(results)}")
            print(f"Total processing time: {time.time() - batch_start_time:.2f} seconds")
            print(f"Average time per PDF: {(time.time() - batch_start_time) / len(results):.2f} seconds")
        else:
            print("No results to save")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast PDF Regulatory Information Extraction (No LLM)")
    parser.add_argument("--pdf_dir", type=str, default="./pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF file")
    
    args = parser.parse_args()
    
    try:
        print("Starting Fast PDF Regulatory Information Extraction")
        start_time = time.time()
        
        # Create pipeline
        pipeline = PDFRegulatoryPipeline()
        
        if args.single_pdf:
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
            
            # Process single PDF
            results = []
            pdf_file = os.path.basename(args.single_pdf)
            
            try:
                answers = pipeline.process_single_pdf(args.single_pdf)
                
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                results.append(result)
                
                # Save to Excel
                df = pd.DataFrame(results)
                columns = ["PDF Filename"] + pipeline.info_extractor.questions
                df = df[columns]
                df.to_excel(args.output, index=False)
                print(f"Results saved to {args.output}")
                
            except Exception as e:
                print(f"Error processing {args.single_pdf}: {e}")
        else:
            # Process directory
            pipeline.process_pdf_directory(args.pdf_dir, args.output)
        
        print(f"Processing completed in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Uncomment for testing with sample data
    """
    # Example of how the turnover extraction works
    sample_text = '''
    Business Segment Reporting:
    
    Product Category: Textiles
    Revenue from operations: Rs. 309,223,000
    Sale of products: 30,92,23,000
    
    Turnover of highest contributing product: 309223000
    
    Segment wise revenue:
    Textile Products: Rs. 25,00,00,000
    Other Products: Rs. 5,00,00,000
    '''
    
    extractor = RegulatoryInfoExtractor()
    turnover = extractor.extract_turnover(sample_text)
    print(f"Extracted turnover: {turnover}")
    """
    
    main()
