import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional

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
    
    def extract_specific_pages(self, pdf_path: str, page_numbers: List[int]) -> Dict[int, str]:
        """Extract text from specific pages only"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            extracted_pages = {}
            
            for page_num in page_numbers:
                if page_num <= len(doc):  # Check if page exists
                    page = doc[page_num - 1]  # PyMuPDF uses 0-based indexing
                    text = page.get_text()
                    extracted_pages[page_num] = text
                else:
                    print(f"Page {page_num} does not exist in PDF")
                    extracted_pages[page_num] = ""
                    
            doc.close()
            print(f"Specific pages extraction took {time.time() - start_time:.2f} seconds")
            return extracted_pages
        except Exception as e:
            print(f"PDF extraction error after {time.time() - start_time:.2f} seconds: {e}")
            raise

class FormFieldExtractor:
    """Class for extracting specific form fields using pattern matching"""
    
    def __init__(self):
        # Define patterns for each field
        self.field_patterns = {
            "Corporate identity number (CIN) of company": [
                r'CIN[:\s]*([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
                r'Corporate Identity Number[:\s]*([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
                r'([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
            ],
            "Financial year to which financial statements relates": [
                r'Financial Year[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'FY[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'Year ended[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'for the year[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'([0-9]{4}[-/][0-9]{2,4})'
            ],
            "Product or service category code (ITC/ NPCS 4 digit code)": [
                r'ITC[:\s]*(\d{4})',
                r'NPCS[:\s]*(\d{4})',
                r'Product code[:\s]*(\d{4})',
                r'Service code[:\s]*(\d{4})',
                r'Category code[:\s]*(\d{4})'
            ],
            "Description of the product or service category": [
                r'Product category[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)',
                r'Service category[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)',
                r'Business segment[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)',
                r'Main business[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)'
            ],
            "Turnover of the product or service category (in Rupees)": [
                r'Category turnover[:\s]*(Rs\.?\s*[\d,]+)',
                r'Segment turnover[:\s]*(Rs\.?\s*[\d,]+)',
                r'Product turnover[:\s]*(Rs\.?\s*[\d,]+)',
                r'Service turnover[:\s]*(Rs\.?\s*[\d,]+)',
                r'Revenue[:\s]*(Rs\.?\s*[\d,]+)'
            ],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": [
                r'Highest.*?code[:\s]*(\d{8})',
                r'Main product code[:\s]*(\d{8})',
                r'Primary service code[:\s]*(\d{8})',
                r'ITC.*?8[:\s]*(\d{8})',
                r'NPCS.*?8[:\s]*(\d{8})'
            ],
            "Description of the product or service": [
                r'Main product[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)',
                r'Primary service[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)',
                r'Highest contributing[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)',
                r'Product description[:\s]*([A-Za-z\s,.-]+?)(?:\n|$)'
            ],
            "Turnover of highest contributing product or service (in Rupees)": [
                r'Highest.*?turnover[:\s]*([\d,]+)',
                r'Main.*?turnover[:\s]*([\d,]+)',
                r'Primary.*?revenue[:\s]*([\d,]+)',
                r'Highest.*?revenue[:\s]*([\d,]+)',
                r'Maximum.*?turnover[:\s]*([\d,]+)',
                # Additional patterns for common formats
                r'Rs\.?\s*([\d,]+)\s*(?:lakhs?|crores?)?',
                r'₹\s*([\d,]+)',
                r'(\d{1,3}(?:,\d{3})*|\d+)(?:\s*(?:lakhs?|crores?))?'
            ]
        }
    
    def extract_field_value(self, text: str, field_name: str) -> str:
        """Extract a specific field value from text using regex patterns"""
        patterns = self.field_patterns.get(field_name, [])
        
        for pattern in patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the extracted value
                    value = self._clean_extracted_value(value, field_name)
                    if value:  # Only return non-empty values
                        return value
            except Exception as e:
                print(f"Error in pattern matching for {field_name}: {e}")
                continue
        
        return "Information not found"
    
    def _clean_extracted_value(self, value: str, field_name: str) -> str:
        """Clean and format extracted values"""
        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        # Field-specific cleaning
        if "CIN" in field_name:
            # Ensure CIN format is correct
            value = re.sub(r'[^A-Z0-9]', '', value.upper())
            
        elif "financial year" in field_name.lower():
            # Standardize financial year format
            value = re.sub(r'[^\d\-/]', '', value)
            
        elif "code" in field_name.lower():
            # Extract only digits for codes
            digits = re.findall(r'\d+', value)
            if digits:
                if "4 digit" in field_name:
                    value = digits[0][:4] if len(digits[0]) >= 4 else digits[0]
                elif "8 digit" in field_name:
                    value = digits[0][:8] if len(digits[0]) >= 8 else digits[0]
                else:
                    value = digits[0]
        
        elif "turnover" in field_name.lower() or "Rupees" in field_name:
            # Clean up financial figures
            # Remove currency symbols and extract numbers
            value = re.sub(r'[^\d,]', '', value)
            # Remove commas for clean number
            if "highest contributing" in field_name.lower():
                value = value.replace(',', '')
        
        elif "description" in field_name.lower():
            # Clean up descriptions
            value = re.sub(r'[^\w\s,.-]', '', value)
            # Limit length for descriptions
            value = value[:100] if len(value) > 100 else value
        
        return value

class PDFFormExtractor:
    """Main class for extracting form data from scanned PDFs"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.field_extractor = FormFieldExtractor()
        
        # Define the specific fields to extract
        self.fields = [
            "Corporate identity number (CIN) of company",
            "Financial year to which financial statements relates",
            "Product or service category code (ITC/ NPCS 4 digit code)",
            "Description of the product or service category",
            "Turnover of the product or service category (in Rupees)",
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
            "Description of the product or service",
            "Turnover of highest contributing product or service (in Rupees)"
        ]
        
        # Pages to search for each field (can be customized based on form layout)
        self.field_pages = {
            "Corporate identity number (CIN) of company": [1],
            "Financial year to which financial statements relates": [1],
            "Product or service category code (ITC/ NPCS 4 digit code)": [1, 10],
            "Description of the product or service category": [1, 10],
            "Turnover of the product or service category (in Rupees)": [10],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": [10],
            "Description of the product or service": [10],
            "Turnover of highest contributing product or service (in Rupees)": [10]
        }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF file and extract form data"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from pages 1 and 10 only
            pages_text = self.pdf_extractor.extract_specific_pages(pdf_path, [1, 10])
            
            # Store results
            results = {}
            
            # Process each field
            for field in self.fields:
                field_start_time = time.time()
                print(f"Extracting: {field[:50]}...")
                
                # Get pages to search for this field
                search_pages = self.field_pages.get(field, [1, 10])
                
                # Combine text from relevant pages
                combined_text = ""
                for page_num in search_pages:
                    if page_num in pages_text:
                        combined_text += f"\n--- Page {page_num} ---\n"
                        combined_text += pages_text[page_num]
                
                # Extract field value
                value = self.field_extractor.extract_field_value(combined_text, field)
                results[field] = value
                
                print(f"  Found: {value}")
                print(f"  Extraction took {time.time() - field_start_time:.2f} seconds")
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {field: f"Error: {str(e)}" for field in self.fields}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """Process multiple PDF files and save results to Excel"""
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
                pdf_start_time = time.time()
                extracted_data = self.process_pdf(pdf_path)
                
                # Add filename to results
                result = {"PDF Filename": pdf_file}
                result.update(extracted_data)
                results.append(result)
                
                print(f"Processed {pdf_file} in {time.time() - pdf_start_time:.2f} seconds")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                # Add error entry
                result = {"PDF Filename": pdf_file}
                result.update({field: f"Error: {str(e)}" for field in self.fields})
                results.append(result)
        
        # Save results to Excel
        if results:
            df = pd.DataFrame(results)
            
            # Reorder columns
            columns = ["PDF Filename"] + self.fields
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            print(f"Processed {len(results)} PDFs successfully")
        else:
            print("No results to save")
        
        print(f"Total batch processing completed in {time.time() - batch_start_time:.2f} seconds")

def main():
    """Main function with command line argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Form Data Extractor for Scanned Forms")
    parser.add_argument("--pdf_dir", type=str, 
                       default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", 
                       help="Directory containing PDF files")
    parser.add_argument("--output", type=str, 
                       default="form_extraction_results.xlsx", 
                       help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, 
                       help="Process a single PDF instead of a directory")
    
    args = parser.parse_args()
    
    try:
        print("Starting PDF Form Data Extraction")
        start_time = time.time()
        
        # Create extractor
        extractor = PDFFormExtractor()
        
        if args.single_pdf:
            # Process single PDF
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
            
            pdf_file = os.path.basename(args.single_pdf)
            extracted_data = extractor.process_pdf(args.single_pdf)
            
            # Save results
            result = {"PDF Filename": pdf_file}
            result.update(extracted_data)
            
            df = pd.DataFrame([result])
            columns = ["PDF Filename"] + extractor.fields
            df = df[columns]
            df.to_excel(args.output, index=False)
            print(f"Results saved to {args.output}")
            
        else:
            # Process directory
            extractor.process_pdfs_batch(args.pdf_dir, args.output)
        
        print(f"Processing completed successfully in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
