import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional

class PDFTextAnalyzer:
    """Class to analyze and understand how PyMuPDF extracts text from your PDFs"""
    
    def analyze_pdf_structure(self, pdf_path: str):
        """Analyze the PDF structure and show exactly how text is extracted"""
        print(f"\n{'='*60}")
        print(f"ANALYZING PDF: {pdf_path}")
        print(f"{'='*60}")
        
        try:
            doc = fitz.open(pdf_path)
            print(f"Total pages in PDF: {len(doc)}")
            
            # Analyze pages 1 and 10
            for page_num in [1, 10]:
                if page_num <= len(doc):
                    print(f"\n{'-'*40}")
                    print(f"PAGE {page_num} ANALYSIS")
                    print(f"{'-'*40}")
                    
                    page = doc[page_num - 1]
                    
                    # Get text using different methods
                    text_default = page.get_text()
                    text_dict = page.get_text("dict")
                    text_blocks = page.get_text("blocks")
                    
                    print(f"Text length (default method): {len(text_default)} characters")
                    print(f"Number of text blocks: {len(text_blocks)}")
                    
                    # Show raw text
                    print(f"\nRAW TEXT (first 500 characters):")
                    print(repr(text_default[:500]))
                    
                    print(f"\nFORMATTED TEXT:")
                    print(text_default[:500])
                    
                    # Split by lines and analyze
                    lines = text_default.split('\n')
                    print(f"\nLINE-BY-LINE ANALYSIS (first 20 lines):")
                    for i, line in enumerate(lines[:20]):
                        if line.strip():  # Only show non-empty lines
                            print(f"Line {i+1:2d}: '{line.strip()}'")
                    
                    # Look for specific patterns we expect
                    print(f"\nPATTERN DETECTION:")
                    
                    # CIN patterns
                    cin_matches = re.findall(r'[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}', text_default, re.IGNORECASE)
                    if cin_matches:
                        print(f"CIN patterns found: {cin_matches}")
                    else:
                        # Look for partial CIN patterns
                        partial_cin = re.findall(r'[LU]\d+[A-Z]*\d*[A-Z]*\d*', text_default, re.IGNORECASE)
                        if partial_cin:
                            print(f"Partial CIN-like patterns: {partial_cin}")
                        else:
                            print("No CIN patterns found")
                    
                    # Year patterns
                    year_patterns = re.findall(r'\d{4}[-/]\d{2,4}', text_default)
                    if year_patterns:
                        print(f"Year patterns found: {year_patterns}")
                    else:
                        # Look for just years
                        years = re.findall(r'20\d{2}', text_default)
                        if years:
                            print(f"Years found: {years}")
                        else:
                            print("No year patterns found")
                    
                    # Number patterns (4 digits, 8 digits, large numbers)
                    four_digit = re.findall(r'\b\d{4}\b', text_default)
                    eight_digit = re.findall(r'\b\d{8}\b', text_default)
                    large_numbers = re.findall(r'\b\d{6,}\b', text_default)
                    
                    print(f"4-digit numbers: {four_digit[:10] if len(four_digit) > 10 else four_digit}")
                    print(f"8-digit numbers: {eight_digit[:5] if len(eight_digit) > 5 else eight_digit}")
                    print(f"Large numbers (6+ digits): {large_numbers[:10] if len(large_numbers) > 10 else large_numbers}")
                    
                    # Look for currency and financial terms
                    financial_terms = re.findall(r'(?:rs|rupees|₹|turnover|revenue|sales)[\s\.:]*(\d+(?:,\d{3})*)', text_default, re.IGNORECASE)
                    if financial_terms:
                        print(f"Financial figures: {financial_terms}")
                    
                    print(f"\nALL NUMBERS FOUND ON PAGE {page_num}:")
                    all_numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text_default)
                    print(f"Total numbers found: {len(all_numbers)}")
                    if all_numbers:
                        print(f"First 20 numbers: {all_numbers[:20]}")
                        # Show unique numbers
                        unique_numbers = list(set(all_numbers))
                        unique_numbers.sort(key=lambda x: len(x), reverse=True)
                        print(f"Longest numbers: {unique_numbers[:10]}")
                
                else:
                    print(f"Page {page_num} does not exist in this PDF")
            
            doc.close()
            
        except Exception as e:
            print(f"Error analyzing PDF: {e}")

class SmartFieldExtractor:
    """Extract fields based on actual text patterns found in PDFs"""
    
    def __init__(self):
        self.debug = True
    
    def extract_all_fields(self, text: str, page_num: int) -> Dict[str, str]:
        """Extract all fields from the given text"""
        
        if self.debug:
            print(f"\nEXTRACTING FROM PAGE {page_num}")
            print(f"Text length: {len(text)} characters")
        
        results = {}
        
        # 1. Corporate Identity Number (CIN)
        results["Corporate identity number (CIN) of company"] = self._extract_cin(text)
        
        # 2. Financial Year
        results["Financial year to which financial statements relates"] = self._extract_financial_year(text)
        
        # 3. Product/Service Category Code (4 digit)
        results["Product or service category code (ITC/ NPCS 4 digit code)"] = self._extract_4_digit_code(text)
        
        # 4. Description of product/service category
        results["Description of the product or service category"] = self._extract_category_description(text)
        
        # 5. Category turnover
        results["Turnover of the product or service category (in Rupees)"] = self._extract_category_turnover(text)
        
        # 6. Highest contributing product code (8 digit)
        results["Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)"] = self._extract_8_digit_code(text)
        
        # 7. Description of highest contributing product
        results["Description of the product or service"] = self._extract_product_description(text)
        
        # 8. Highest contributing turnover
        results["Turnover of highest contributing product or service (in Rupees)"] = self._extract_highest_turnover(text)
        
        return results
    
    def _extract_cin(self, text: str) -> str:
        """Extract CIN with multiple fallback strategies"""
        
        # Strategy 1: Perfect CIN format
        cin_match = re.search(r'([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})', text, re.IGNORECASE)
        if cin_match:
            result = cin_match.group(1).upper()
            if self.debug:
                print(f"CIN found (perfect format): {result}")
            return result
        
        # Strategy 2: CIN with separators
        cin_sep_match = re.search(r'([LU][-\s]?\d{5}[-\s]?[A-Z]{2}[-\s]?\d{4}[-\s]?[A-Z]{3}[-\s]?\d{6})', text, re.IGNORECASE)
        if cin_sep_match:
            result = re.sub(r'[^A-Z0-9]', '', cin_sep_match.group(1).upper())
            if self.debug:
                print(f"CIN found (with separators): {result}")
            return result
        
        # Strategy 3: Look for any L/U followed by numbers and letters
        partial_cin = re.search(r'([LU]\d+[A-Z]*\d+[A-Z]*\d+)', text, re.IGNORECASE)
        if partial_cin:
            result = partial_cin.group(1).upper()
            if len(result) >= 15:  # Reasonable CIN length
                if self.debug:
                    print(f"CIN found (partial): {result}")
                return result
        
        if self.debug:
            print("CIN not found")
        return "Information not found"
    
    def _extract_financial_year(self, text: str) -> str:
        """Extract financial year"""
        
        # Strategy 1: Standard FY format
        fy_match = re.search(r'(\d{4}[-/]\d{2,4})', text)
        if fy_match:
            result = fy_match.group(1)
            if self.debug:
                print(f"Financial year found: {result}")
            return result
        
        # Strategy 2: Look for consecutive years
        years = re.findall(r'20\d{2}', text)
        if len(years) >= 2:
            # Try to find consecutive years
            for i in range(len(years) - 1):
                year1 = int(years[i])
                year2 = int(years[i + 1])
                if year2 == year1 + 1:
                    result = f"{year1}-{str(year2)[2:]}"
                    if self.debug:
                        print(f"Financial year found (consecutive years): {result}")
                    return result
        
        if self.debug:
            print("Financial year not found")
        return "Information not found"
    
    def _extract_4_digit_code(self, text: str) -> str:
        """Extract 4-digit product/service code"""
        
        # Find all 4-digit numbers
        four_digits = re.findall(r'\b(\d{4})\b', text)
        
        if four_digits:
            # Filter out years and common non-code numbers
            filtered = []
            for code in four_digits:
                num = int(code)
                if not (1900 <= num <= 2030):  # Not a year
                    filtered.append(code)
            
            if filtered:
                result = filtered[0]  # Take the first valid one
                if self.debug:
                    print(f"4-digit code found: {result} (from {four_digits})")
                return result
        
        if self.debug:
            print("4-digit code not found")
        return "Information not found"
    
    def _extract_8_digit_code(self, text: str) -> str:
        """Extract 8-digit product/service code"""
        
        # Find all 8-digit numbers
        eight_digits = re.findall(r'\b(\d{8})\b', text)
        
        if eight_digits:
            result = eight_digits[0]
            if self.debug:
                print(f"8-digit code found: {result}")
            return result
        
        if self.debug:
            print("8-digit code not found")
        return "Information not found"
    
    def _extract_category_description(self, text: str) -> str:
        """Extract description of product/service category"""
        
        # Look for business-related descriptive text
        keywords = ['manufacture', 'production', 'service', 'trading', 'business', 'activity', 'operations']
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 20 and any(keyword in line.lower() for keyword in keywords):
                # Clean the line
                cleaned = re.sub(r'[^\w\s,.-]', '', line)
                if len(cleaned) > 10:
                    result = cleaned[:100]  # Limit length
                    if self.debug:
                        print(f"Category description found: {result}")
                    return result
        
        if self.debug:
            print("Category description not found")
        return "Information not found"
    
    def _extract_product_description(self, text: str) -> str:
        """Extract description of main product/service"""
        
        # Similar to category description but look for more specific terms
        keywords = ['product', 'service', 'main', 'primary', 'principal', 'major']
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 15 and any(keyword in line.lower() for keyword in keywords):
                cleaned = re.sub(r'[^\w\s,.-]', '', line)
                if len(cleaned) > 10:
                    result = cleaned[:100]
                    if self.debug:
                        print(f"Product description found: {result}")
                    return result
        
        if self.debug:
            print("Product description not found")
        return "Information not found"
    
    def _extract_category_turnover(self, text: str) -> str:
        """Extract category turnover"""
        
        # Look for medium-sized numbers (not the largest ones)
        numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*|\d{6,10})\b', text)
        
        if numbers:
            # Convert to integers for comparison
            num_values = []
            for num in numbers:
                try:
                    clean_num = num.replace(',', '')
                    if clean_num.isdigit():
                        val = int(clean_num)
                        if 100000 <= val <= 100000000:  # Reasonable range for category turnover
                            num_values.append((val, num))
                except:
                    continue
            
            if num_values:
                # Sort and take a middle value (not the largest, not the smallest)
                num_values.sort()
                if len(num_values) >= 3:
                    result = str(num_values[len(num_values)//2][0])
                else:
                    result = str(num_values[0][0])
                
                if self.debug:
                    print(f"Category turnover found: {result}")
                return result
        
        if self.debug:
            print("Category turnover not found")
        return "Information not found"
    
    def _extract_highest_turnover(self, text: str) -> str:
        """Extract highest contributing product/service turnover"""
        
        # Look for the largest numbers in the text
        numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*|\d{6,})\b', text)
        
        if numbers:
            max_val = 0
            max_num = ""
            
            for num in numbers:
                try:
                    clean_num = num.replace(',', '')
                    if clean_num.isdigit():
                        val = int(clean_num)
                        if val > max_val and val >= 1000000:  # At least 1 million
                            max_val = val
                            max_num = str(val)
                except:
                    continue
            
            if max_num:
                if self.debug:
                    print(f"Highest turnover found: {max_num}")
                return max_num
        
        if self.debug:
            print("Highest turnover not found")
        return "Information not found"

class PDFFormExtractor:
    """Main class for extracting form data from scanned PDFs"""
    
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        self.analyzer = PDFTextAnalyzer()
        self.extractor = SmartFieldExtractor()
        
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
    
    def analyze_single_pdf(self, pdf_path: str):
        """Analyze a single PDF to understand its structure"""
        self.analyzer.analyze_pdf_structure(pdf_path)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF file and extract form data"""
        print(f"\nProcessing PDF: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract text from pages 1 and 10
            all_results = {}
            
            for page_num in [1, 10]:
                if page_num <= len(doc):
                    page = doc[page_num - 1]
                    text = page.get_text()
                    
                    if text.strip():
                        page_results = self.extractor.extract_all_fields(text, page_num)
                        
                        # Merge results, preferring non-"Information not found" values
                        for field, value in page_results.items():
                            if field not in all_results or all_results[field] == "Information not found":
                                all_results[field] = value
            
            doc.close()
            
            # Ensure all fields are present
            final_results = {}
            for field in self.fields:
                final_results[field] = all_results.get(field, "Information not found")
            
            return final_results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {field: f"Error: {str(e)}" for field in self.fields}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """Process multiple PDF files and save results to Excel"""
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
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            extracted_data = self.process_pdf(pdf_path)
            
            result = {"PDF Filename": pdf_file}
            result.update(extracted_data)
            results.append(result)
        
        # Save to Excel
        if results:
            df = pd.DataFrame(results)
            columns = ["PDF Filename"] + self.fields
            df = df[columns]
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart PDF Form Data Extractor")
    parser.add_argument("--pdf_dir", type=str, 
                       default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", 
                       help="Directory containing PDF files")
    parser.add_argument("--output", type=str, 
                       default="smart_extraction_results.xlsx", 
                       help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, 
                       help="Process a single PDF")
    parser.add_argument("--analyze_only", action="store_true", 
                       help="Only analyze PDF structure, don't extract data")
    
    args = parser.parse_args()
    
    extractor = PDFFormExtractor(debug_mode=True)
    
    if args.single_pdf:
        if args.analyze_only:
            # Just analyze the structure
            extractor.analyze_single_pdf(args.single_pdf)
        else:
            # Analyze and then extract
            extractor.analyze_single_pdf(args.single_pdf)
            extracted_data = extractor.process_pdf(args.single_pdf)
            
            # Save results
            pdf_file = os.path.basename(args.single_pdf)
            result = {"PDF Filename": pdf_file}
            result.update(extracted_data)
            
            df = pd.DataFrame([result])
            df.to_excel(args.output, index=False)
            print(f"Results saved to {args.output}")
    else:
        extractor.process_pdfs_batch(args.pdf_dir, args.output)

if __name__ == "__main__":
    main()
