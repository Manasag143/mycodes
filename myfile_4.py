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
                    print(f"Extracted {len(text)} characters from page {page_num}")
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
    """Class for extracting specific form fields using flexible pattern matching"""
    
    def __init__(self):
        # More flexible patterns that look for common keywords and nearby values
        self.field_keywords = {
            "Corporate identity number (CIN) of company": [
                "cin", "corporate identity", "corporate identification", 
                "company identification", "registration number"
            ],
            "Financial year to which financial statements relates": [
                "financial year", "fy", "year ended", "period ended", 
                "financial period", "accounting year", "reporting period"
            ],
            "Product or service category code (ITC/ NPCS 4 digit code)": [
                "itc", "npcs", "product code", "service code", "category code",
                "business code", "activity code", "classification code"
            ],
            "Description of the product or service category": [
                "product category", "service category", "business segment",
                "main business", "principal business", "nature of business",
                "business activity", "product description", "service description"
            ],
            "Turnover of the product or service category (in Rupees)": [
                "category turnover", "segment turnover", "product turnover",
                "service turnover", "revenue", "sales", "income"
            ],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": [
                "highest turnover", "main product", "primary product",
                "major product", "principal product", "8 digit", "eight digit"
            ],
            "Description of the product or service": [
                "main product", "primary service", "principal business",
                "highest contributing", "major business", "core business"
            ],
            "Turnover of highest contributing product or service (in Rupees)": [
                "highest turnover", "main turnover", "primary turnover",
                "principal turnover", "major turnover", "maximum turnover",
                "highest revenue", "main revenue", "primary revenue"
            ]
        }
    
    def extract_field_value(self, text: str, field_name: str) -> str:
        """Extract field value using flexible keyword-based search"""
        print(f"Searching for: {field_name}")
        
        # Get keywords for this field
        keywords = self.field_keywords.get(field_name, [])
        
        # Clean and normalize text for better matching
        normalized_text = self._normalize_text(text)
        
        # Try different extraction strategies
        strategies = [
            self._extract_by_keywords_and_patterns,
            self._extract_by_proximity_search,
            self._extract_by_line_analysis,
            self._extract_by_context_matching
        ]
        
        for strategy in strategies:
            result = strategy(normalized_text, field_name, keywords)
            if result and result != "Information not found":
                print(f"Found using strategy {strategy.__name__}: {result}")
                return result
        
        print(f"No value found for {field_name}")
        return "Information not found"
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better pattern matching"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        # Keep original text structure but clean it
        return text.strip()
    
    def _extract_by_keywords_and_patterns(self, text: str, field_name: str, keywords: List[str]) -> str:
        """Extract using keyword proximity and common patterns"""
        
        if "CIN" in field_name or "corporate identity" in field_name.lower():
            # Look for CIN patterns
            cin_patterns = [
                r'([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})',
                r'CIN[:\s]*([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})',
                r'([LU]-?\d{5}-?[A-Z]{2}-?\d{4}-?[A-Z]{3}-?\d{6})'
            ]
            for pattern in cin_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return re.sub(r'[^A-Z0-9]', '', match.group(1).upper())
        
        elif "financial year" in field_name.lower():
            # Look for year patterns
            year_patterns = [
                r'(\d{4}[-/]\d{2,4})',
                r'FY\s*(\d{4}[-/]\d{2,4})',
                r'year\s*(\d{4}[-/]\d{2,4})',
                r'(\d{4}\s*-\s*\d{2,4})'
            ]
            for pattern in year_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        elif "4 digit code" in field_name:
            # Look for 4-digit codes
            four_digit_patterns = [
                r'(?:ITC|NPCS|code)[:\s]*(\d{4})',
                r'(\d{4})\s*(?:ITC|NPCS)',
                r'\b(\d{4})\b'
            ]
            for pattern in four_digit_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0]
        
        elif "8 digit code" in field_name:
            # Look for 8-digit codes
            eight_digit_patterns = [
                r'(?:ITC|NPCS|code)[:\s]*(\d{8})',
                r'(\d{8})\s*(?:ITC|NPCS)',
                r'\b(\d{8})\b'
            ]
            for pattern in eight_digit_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return matches[0]
        
        elif "turnover" in field_name.lower() and "highest" in field_name.lower():
            # Look for numerical values that could be turnover
            turnover_patterns = [
                r'(\d{1,3}(?:,\d{3})*)',  # Numbers with commas
                r'(\d+)',  # Any digits
                r'Rs\.?\s*(\d{1,3}(?:,\d{3})*)',  # With Rs.
                r'₹\s*(\d{1,3}(?:,\d{3})*)'  # With rupee symbol
            ]
            
            for pattern in turnover_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    # Return the largest number found (likely the turnover)
                    numbers = [int(match.replace(',', '')) for match in matches if match.replace(',', '').isdigit()]
                    if numbers:
                        largest = max(numbers)
                        return str(largest)
        
        return ""
    
    def _extract_by_proximity_search(self, text: str, field_name: str, keywords: List[str]) -> str:
        """Search for values near keywords"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if any keyword is in this line
            for keyword in keywords:
                if keyword in line_lower:
                    # Look for values in this line and nearby lines
                    search_lines = []
                    
                    # Current line
                    search_lines.append(line)
                    
                    # Previous and next lines
                    if i > 0:
                        search_lines.append(lines[i-1])
                    if i < len(lines) - 1:
                        search_lines.append(lines[i+1])
                    
                    combined_context = ' '.join(search_lines)
                    
                    # Extract based on field type
                    if "CIN" in field_name:
                        cin_match = re.search(r'([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})', combined_context, re.IGNORECASE)
                        if cin_match:
                            return cin_match.group(1).upper()
                    
                    elif "financial year" in field_name.lower():
                        year_match = re.search(r'(\d{4}[-/]\d{2,4})', combined_context)
                        if year_match:
                            return year_match.group(1)
                    
                    elif "code" in field_name.lower():
                        if "4 digit" in field_name:
                            code_match = re.search(r'\b(\d{4})\b', combined_context)
                        else:
                            code_match = re.search(r'\b(\d{8})\b', combined_context)
                        if code_match:
                            return code_match.group(1)
                    
                    elif "turnover" in field_name.lower():
                        # Look for numbers
                        numbers = re.findall(r'(\d{1,3}(?:,\d{3})*|\d+)', combined_context)
                        if numbers:
                            # Return the largest number
                            nums = [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit()]
                            if nums:
                                return str(max(nums))
                    
                    elif "description" in field_name.lower():
                        # Extract text after the keyword
                        after_keyword = combined_context.split(keyword, 1)
                        if len(after_keyword) > 1:
                            desc = after_keyword[1].strip()
                            # Clean and limit description
                            desc = re.sub(r'[^\w\s,.-]', '', desc)[:100]
                            return desc.strip()
        
        return ""
    
    def _extract_by_line_analysis(self, text: str, field_name: str, keywords: List[str]) -> str:
        """Analyze each line for potential values"""
        lines = text.split('\n')
        
        if "CIN" in field_name:
            for line in lines:
                cin_match = re.search(r'([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})', line, re.IGNORECASE)
                if cin_match:
                    return cin_match.group(1).upper()
        
        elif "financial year" in field_name.lower():
            for line in lines:
                year_match = re.search(r'(\d{4}[-/]\d{2,4})', line)
                if year_match:
                    return year_match.group(1)
        
        elif "turnover" in field_name.lower() and "highest" in field_name.lower():
            # Look for lines with large numbers
            max_number = 0
            for line in lines:
                numbers = re.findall(r'(\d{6,})', line)  # Look for numbers with at least 6 digits
                for num in numbers:
                    try:
                        val = int(num.replace(',', ''))
                        if val > max_number:
                            max_number = val
                    except:
                        continue
            
            if max_number > 0:
                return str(max_number)
        
        return ""
    
    def _extract_by_context_matching(self, text: str, field_name: str, keywords: List[str]) -> str:
        """Try to extract based on document context and common form layouts"""
        
        # For numerical fields, try to find the largest reasonable number
        if "turnover" in field_name.lower() and "highest" in field_name.lower():
            # Find all large numbers in the text
            all_numbers = re.findall(r'\b(\d{6,})\b', text)
            if all_numbers:
                # Convert to integers and find the largest
                numbers = []
                for num in all_numbers:
                    try:
                        numbers.append(int(num.replace(',', '')))
                    except:
                        continue
                
                if numbers:
                    return str(max(numbers))
        
        # For code fields, look for any 4 or 8 digit sequences
        elif "4 digit code" in field_name:
            codes = re.findall(r'\b(\d{4})\b', text)
            if codes:
                return codes[0]  # Return first 4-digit code found
        
        elif "8 digit code" in field_name:
            codes = re.findall(r'\b(\d{8})\b', text)
            if codes:
                return codes[0]  # Return first 8-digit code found
        
        return ""

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
        
        # Pages to search for each field
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
            
            # Debug: Print sample text from each page
            for page_num, text in pages_text.items():
                print(f"Page {page_num} sample text (first 200 chars): {text[:200]}...")
            
            # Store results
            results = {}
            
            # Process each field
            for field in self.fields:
                field_start_time = time.time()
                print(f"\nExtracting: {field}")
                
                # Get pages to search for this field
                search_pages = self.field_pages.get(field, [1, 10])
                
                # Combine text from relevant pages
                combined_text = ""
                for page_num in search_pages:
                    if page_num in pages_text and pages_text[page_num]:
                        combined_text += f"\n--- Page {page_num} ---\n"
                        combined_text += pages_text[page_num]
                
                if not combined_text.strip():
                    print(f"  No text found in pages {search_pages}")
                    results[field] = "No text found in target pages"
                    continue
                
                # Extract field value
                value = self.field_extractor.extract_field_value(combined_text, field)
                results[field] = value
                
                print(f"  Result: {value}")
                print(f"  Extraction took {time.time() - field_start_time:.2f} seconds")
            
            print(f"\nCompleted processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
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
