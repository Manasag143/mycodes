import fitz  # PyMuPDF
import pandas as pd
import re
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFDebugger:
    """Class specifically for debugging PDF content and understanding structure"""
    
    def __init__(self):
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
    
    def extract_and_analyze_pdf(self, pdf_path: str):
        """Extract and thoroughly analyze PDF content"""
        print(f"\n{'='*100}")
        print(f"COMPLETE PDF ANALYSIS: {os.path.basename(pdf_path)}")
        print(f"{'='*100}")
        
        try:
            doc = fitz.open(pdf_path)
            print(f"Total pages in PDF: {len(doc)}")
            
            # Analyze specific pages (1 and 10)
            target_pages = [1, 10]
            
            for page_num in target_pages:
                if page_num <= len(doc):
                    self._analyze_page_content(doc, page_num)
                else:
                    print(f"\nPage {page_num} does not exist in this PDF")
            
            doc.close()
            
        except Exception as e:
            print(f"Error analyzing PDF: {e}")
    
    def _analyze_page_content(self, doc, page_num: int):
        """Analyze content of a specific page"""
        print(f"\n{'-'*80}")
        print(f"PAGE {page_num} ANALYSIS")
        print(f"{'-'*80}")
        
        page = doc.load_page(page_num - 1)  # 0-based indexing
        text = page.get_text()
        
        print(f"Page {page_num} character count: {len(text)}")
        print(f"Page {page_num} line count: {len(text.split(chr(10)))}")
        
        # Show raw text structure
        print(f"\n--- RAW TEXT CONTENT (First 50 lines) ---")
        lines = text.split('\n')
        for i, line in enumerate(lines[:50]):
            if line.strip():  # Only show non-empty lines
                print(f"{i+1:3d}: {line}")
        
        if len(lines) > 50:
            print(f"\n... (Total {len(lines)} lines, showing only first 50)")
            print(f"\n--- LAST 20 LINES ---")
            for i, line in enumerate(lines[-20:], len(lines)-19):
                if line.strip():
                    print(f"{i:3d}: {line}")
        
        # Search for potential answers
        print(f"\n--- PATTERN DETECTION ON PAGE {page_num} ---")
        
        # 1. CIN patterns
        cin_patterns = re.findall(r'[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}', text)
        if cin_patterns:
            print(f"✓ CIN patterns found: {cin_patterns}")
        else:
            print("✗ No standard CIN patterns found")
            # Look for lines containing "CIN" or "Corporate Identity"
            cin_lines = [line.strip() for line in lines if 'cin' in line.lower() or 'corporate identity' in line.lower()]
            if cin_lines:
                print(f"  Lines mentioning CIN: {cin_lines[:3]}")
        
        # 2. Financial year patterns
        fy_patterns = re.findall(r'20\d{2}[-/]\d{2,4}|FY\s*20\d{2}|financial year.*?20\d{2}', text, re.IGNORECASE)
        if fy_patterns:
            print(f"✓ Financial Year patterns: {fy_patterns}")
        else:
            print("✗ No standard FY patterns found")
            fy_lines = [line.strip() for line in lines if any(word in line.lower() for word in ['financial year', 'year ended', 'fy '])]
            if fy_lines:
                print(f"  Lines mentioning Financial Year: {fy_lines[:3]}")
        
        # 3. Look for 4-digit and 8-digit codes
        four_digit_codes = re.findall(r'\b\d{4}\b', text)
        eight_digit_codes = re.findall(r'\b\d{8}\b', text)
        
        if four_digit_codes:
            print(f"✓ 4-digit codes found: {four_digit_codes[:10]}...")
        else:
            print("✗ No 4-digit codes found")
        
        if eight_digit_codes:
            print(f"✓ 8-digit codes found: {eight_digit_codes}")
        else:
            print("✗ No 8-digit codes found")
        
        # 4. Look for monetary amounts
        money_patterns = re.findall(r'(?:Rs\.?|₹|INR)[\s]*[0-9,]+\.?\d*|[0-9,]+\.?\d*\s*(?:crore|lakh|million)', text, re.IGNORECASE)
        if money_patterns:
            print(f"✓ Monetary amounts found: {money_patterns[:10]}...")
        else:
            print("✗ No clear monetary patterns found")
        
        # 5. Look for product/service related content
        product_lines = [line.strip() for line in lines if any(word in line.lower() for word in ['product', 'service', 'business', 'segment', 'category'])]
        if product_lines:
            print(f"✓ Product/Service lines found: {len(product_lines)} lines")
            print(f"  Sample lines: {product_lines[:3]}")
        else:
            print("✗ No product/service related content found")
        
        # 6. Look for tables or structured data
        table_indicators = [line.strip() for line in lines if '\t' in line or '|' in line or len(re.findall(r'\s+', line)) > 5]
        if table_indicators:
            print(f"✓ Potential table structures: {len(table_indicators)} lines")
            print(f"  Sample table lines: {table_indicators[:3]}")
        
        # 7. Look for questions themselves in the text
        print(f"\n--- SEARCHING FOR ACTUAL QUESTIONS IN TEXT ---")
        for i, question in enumerate(self.questions, 1):
            question_found = False
            question_words = question.lower().split()
            
            for line in lines:
                line_lower = line.lower()
                # Check if significant parts of the question appear in the line
                word_matches = sum(1 for word in question_words if len(word) > 3 and word in line_lower)
                if word_matches >= 2:  # If at least 2 significant words match
                    print(f"  Q{i} potential match: {line.strip()}")
                    question_found = True
                    break
            
            if not question_found:
                print(f"  Q{i}: No clear match found")
    
    def find_exact_answers(self, pdf_path: str):
        """Try to find where exactly the answers are located"""
        print(f"\n{'='*100}")
        print(f"ANSWER LOCATION ANALYSIS: {os.path.basename(pdf_path)}")
        print(f"{'='*100}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Check all pages for question content
            for page_num in range(1, min(len(doc) + 1, 15)):  # Check first 15 pages
                page = doc.load_page(page_num - 1)
                text = page.get_text()
                lines = text.split('\n')
                
                # Look for lines that might contain answers
                relevant_lines = []
                for line in lines:
                    line_clean = line.strip()
                    if len(line_clean) > 10:  # Ignore very short lines
                        # Check if line contains potential answer indicators
                        if any(indicator in line_clean.lower() for indicator in [
                            'cin', 'corporate identity', 'l1', 'l2', 'l3', 'l4', 'l5', 'u1', 'u2', 'u3',
                            'financial year', 'fy', 'year ended', '2023', '2024', '2022', '2021',
                            'code', 'itc', 'npcs', 'product', 'service', 'category',
                            'turnover', 'revenue', 'sales', 'rs.', '₹', 'crore', 'lakh',
                            'highest', 'main', 'primary', 'description'
                        ]):
                            relevant_lines.append((line_clean, line_clean.lower()))
                
                if relevant_lines:
                    print(f"\n--- PAGE {page_num} - RELEVANT LINES ---")
                    for line, line_lower in relevant_lines[:20]:  # Show first 20 relevant lines
                        print(f"  {line}")
            
            doc.close()
            
        except Exception as e:
            print(f"Error in answer location analysis: {e}")

class SimpleExtractor:
    """Simplified extractor to manually locate answers"""
    
    def __init__(self):
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
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Simple extraction with manual inspection"""
        print(f"\nExtracting from: {os.path.basename(pdf_path)}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Get text from pages 1 and 10
            page1_text = ""
            page10_text = ""
            
            if len(doc) >= 1:
                page1_text = doc.load_page(0).get_text()
            
            if len(doc) >= 10:
                page10_text = doc.load_page(9).get_text()
            
            doc.close()
            
            # Manual extraction attempts
            results = {}
            
            # Process each question with the appropriate page
            for i, question in enumerate(self.questions):
                if i < 2:  # First 2 questions from page 1
                    answer = self._extract_answer_simple(question, page1_text, 1)
                else:  # Rest from page 10
                    answer = self._extract_answer_simple(question, page10_text, 10)
                
                results[question] = answer
                print(f"  Q{i+1}: {answer}")
            
            return results
            
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            return {q: "Error" for q in self.questions}
    
    def _extract_answer_simple(self, question: str, text: str, page_num: int) -> str:
        """Simple answer extraction"""
        if not text:
            return f"Page {page_num} not found"
        
        lines = text.split('\n')
        
        # For each question type, look for the most obvious patterns
        question_lower = question.lower()
        
        if "cin" in question_lower:
            # Look for CIN pattern
            for line in lines:
                cin_match = re.search(r'[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}', line)
                if cin_match:
                    return cin_match.group(0)
            return "CIN not found in standard format"
        
        elif "financial year" in question_lower:
            # Look for year patterns
            for line in lines:
                if 'financial year' in line.lower() or 'fy' in line.lower() or 'year ended' in line.lower():
                    year_match = re.search(r'20\d{2}[-/]\d{2,4}', line)
                    if year_match:
                        return year_match.group(0)
            return "Financial year not found"
        
        elif "4 digit code" in question_lower:
            # Look for 4 digit codes
            four_digit_codes = re.findall(r'\b\d{4}\b', text)
            if four_digit_codes:
                return f"Found codes: {four_digit_codes[:5]}"  # Show first 5
            return "No 4-digit codes found"
        
        elif "8 digit code" in question_lower:
            # Look for 8 digit codes
            eight_digit_codes = re.findall(r'\b\d{8}\b', text)
            if eight_digit_codes:
                return f"Found codes: {eight_digit_codes}"
            return "No 8-digit codes found"
        
        elif "turnover" in question_lower:
            # Look for monetary amounts
            money_patterns = re.findall(r'(?:Rs\.?|₹|INR)[\s]*[0-9,]+\.?\d*|[0-9,]+\.?\d*\s*(?:crore|lakh)', text, re.IGNORECASE)
            if money_patterns:
                return f"Found amounts: {money_patterns[:3]}"  # Show first 3
            return "No turnover amounts found"
        
        elif "description" in question_lower:
            # Look for description lines
            desc_lines = [line.strip() for line in lines if 'description' in line.lower() or 'product' in line.lower() or 'service' in line.lower()]
            if desc_lines:
                return f"Found descriptions: {desc_lines[0][:100]}..."  # Show first description
            return "No descriptions found"
        
        else:
            return "Question type not recognized"

def main():
    """Main function for debugging and extraction"""
    debugger = PDFDebugger()
    extractor = SimpleExtractor()
    
    # Get PDF file path from user
    pdf_path = input("Enter the full path to your PDF file: ").strip().strip('"')
    
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return
    
    print("Choose analysis type:")
    print("1. Complete PDF structure analysis")
    print("2. Answer location analysis")
    print("3. Simple extraction attempt")
    print("4. All of the above")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice in ['1', '4']:
        debugger.extract_and_analyze_pdf(pdf_path)
    
    if choice in ['2', '4']:
        debugger.find_exact_answers(pdf_path)
    
    if choice in ['3', '4']:
        print(f"\n{'='*80}")
        print("SIMPLE EXTRACTION ATTEMPT")
        print(f"{'='*80}")
        results = extractor.extract_from_pdf(pdf_path)
        
        # Save results to Excel for inspection
        df = pd.DataFrame([{"PDF_File": os.path.basename(pdf_path), **results}])
        output_file = "debug_extraction_results.xlsx"
        df.to_excel(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
