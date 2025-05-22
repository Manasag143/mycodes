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
                
            doc.close()
            print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages
        except Exception as e:
            print(f"PDF extraction error after {time.time() - start_time:.2f} seconds: {e}")
            raise
    
    def search_keywords_in_pdf(self, pages: List[Dict[str, Any]], keywords: List[str], question: str) -> List[int]:
        """Search for keywords with optimized pattern matching"""
        start_time = time.time()
        relevant_pages = []
        
        # First try to find pages with exact question mentions
        question_keywords = [
            question,
            re.sub(r'(of company|to which|relates|code|in Rupees)', '', question).strip()
        ]
        
        # Search for the complete question or its main part first
        for page in pages:
            for q_keyword in question_keywords:
                if len(q_keyword) > 10 and q_keyword.lower() in page["text"].lower():
                    if page["page_num"] not in relevant_pages:
                        relevant_pages.append(page["page_num"])
        
        # If we found pages with the exact question, prioritize those
        if relevant_pages:
            print(f"Keyword search took {time.time() - start_time:.2f} seconds (found exact match)")
            return relevant_pages
        
        # Otherwise, fall back to simplified keyword-based search
        for keyword in keywords:
            for page in pages:
                if keyword.lower() in page["text"].lower():
                    if page["page_num"] not in relevant_pages:
                        relevant_pages.append(page["page_num"])
        
        print(f"Keyword search took {time.time() - start_time:.2f} seconds (found {len(relevant_pages)} pages)")
        return relevant_pages

class AnswerExtractor:
    """Class for extracting answers using pattern matching and rule-based extraction"""
    
    def __init__(self):
        # Define extraction patterns for different question types
        self.patterns = {
            'cin': [
                r'[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}',  # Standard CIN format
                r'CIN[:\s]+([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})',
                r'Corporate Identity Number[:\s]+([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})'
            ],
            'financial_year': [
                r'(?:FY|F\.Y\.|Financial Year)[:\s]*(\d{4}[-/]\d{2,4})',
                r'(?:year ended|period ended)[:\s]*(\d{1,2}[a-zA-Z]*\s+\w+\s+\d{4})',
                r'(\d{4}[-/]\d{2,4})',
                r'for the year (\d{4}[-/]\d{2,4})',
                r'ended (\d{1,2}\w*\s+\w+\s+\d{4})'
            ],
            'product_code_4': [
                r'(?:ITC|NPCS|Product Code|Service Code)[:\s]*(\d{4})\b',
                r'4\s*digit\s*code[:\s]*(\d{4})',
                r'\b(\d{4})\b(?=\s*(?:ITC|NPCS|code))',
                r'Category.*?(\d{4})',
                r'Code[:\s]*(\d{4})\b'
            ],
            'product_code_8': [
                r'(?:ITC|NPCS|Product Code|Service Code)[:\s]*(\d{8})\b',
                r'8\s*digit\s*code[:\s]*(\d{8})',
                r'\b(\d{8})\b(?=\s*(?:ITC|NPCS|code))',
                r'highest.*?contributing.*?(\d{8})',
                r'main.*?product.*?(\d{8})'
            ],
            'turnover': [
                r'(?:Rs\.?|₹|INR)[\s]*([0-9,]+\.?\d*)',
                r'turnover[:\s]*(?:Rs\.?|₹|INR)?[\s]*([0-9,]+\.?\d*)',
                r'revenue[:\s]*(?:Rs\.?|₹|INR)?[\s]*([0-9,]+\.?\d*)',
                r'([0-9,]+\.?\d*)\s*(?:crore|lakh|million|billion)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'
            ]
        }
    
    def extract_answer(self, question: str, context: str) -> str:
        """Extract answer using pattern matching and contextual analysis"""
        if not context or not context.strip():
            return "Information not found"
        
        # Determine question type and apply appropriate extraction
        question_lower = question.lower()
        
        if "cin" in question_lower or "corporate identity" in question_lower:
            return self._extract_cin(context)
        elif "financial year" in question_lower:
            return self._extract_financial_year(context)
        elif "4 digit code" in question_lower:
            return self._extract_product_code_4(context)
        elif "8 digit code" in question_lower:
            return self._extract_product_code_8(context)
        elif "turnover" in question_lower and "category" in question_lower:
            return self._extract_category_turnover(context)
        elif "turnover" in question_lower and "highest" in question_lower:
            return self._extract_highest_turnover(context)
        elif "description" in question_lower and "category" in question_lower:
            return self._extract_category_description(context)
        elif "description" in question_lower and ("highest" in question_lower or "service" in question_lower):
            return self._extract_service_description(context)
        else:
            return self._extract_generic(context, question)
    
    def _extract_cin(self, context: str) -> str:
        """Extract Corporate Identity Number"""
        for pattern in self.patterns['cin']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                # Return the first valid CIN found
                cin = matches[0] if isinstance(matches[0], str) else matches[0]
                if len(cin) == 21:  # Standard CIN length
                    return cin
        return "Information not found"
    
    def _extract_financial_year(self, context: str) -> str:
        """Extract financial year"""
        for pattern in self.patterns['financial_year']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                return matches[0]
        return "Information not found"
    
    def _extract_product_code_4(self, context: str) -> str:
        """Extract 4-digit product/service code"""
        # Look for context lines that mention products/services with codes
        lines = context.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['product', 'service', 'category', 'business']):
                for pattern in self.patterns['product_code_4']:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        return matches[0]
        
        # Fallback: search entire context
        for pattern in self.patterns['product_code_4']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return "Information not found"
    
    def _extract_product_code_8(self, context: str) -> str:
        """Extract 8-digit product/service code"""
        # Look for context lines that mention highest contributing or main products
        lines = context.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['highest', 'main', 'primary', 'contributing']):
                for pattern in self.patterns['product_code_8']:
                    matches = re.findall(pattern, line, re.IGNORECASE)
                    if matches:
                        return matches[0]
        
        # Fallback: search entire context
        for pattern in self.patterns['product_code_8']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return "Information not found"
    
    def _extract_category_turnover(self, context: str) -> str:
        """Extract turnover of product/service category"""
        return self._extract_turnover_amount(context, ['category', 'segment'])
    
    def _extract_highest_turnover(self, context: str) -> str:
        """Extract turnover of highest contributing product/service"""
        # Look for specific patterns related to highest turnover
        lines = context.split('\n')
        
        # First try to find lines with "highest" or similar keywords
        for line in lines:
            if any(keyword in line.lower() for keyword in ['highest', 'maximum', 'main', 'primary']):
                amount = self._extract_amount_from_line(line)
                if amount:
                    return amount
        
        # Look for table-like structures with amounts
        return self._extract_turnover_amount(context, ['highest', 'maximum', 'main', 'primary'])
    
    def _extract_turnover_amount(self, context: str, context_keywords: List[str]) -> str:
        """Extract monetary amounts with context"""
        lines = context.split('\n')
        
        # Look for lines containing both context keywords and amounts
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in context_keywords):
                amount = self._extract_amount_from_line(line)
                if amount:
                    return amount
        
        # Fallback: look for any monetary amount in context
        for pattern in self.patterns['turnover']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            if matches:
                # Clean up the amount (remove commas, etc.)
                amount = matches[0].replace(',', '')
                if amount.replace('.', '').isdigit():
                    return amount
        
        return "Information not found"
    
    def _extract_amount_from_line(self, line: str) -> Optional[str]:
        """Extract monetary amount from a single line"""
        for pattern in self.patterns['turnover']:
            matches = re.findall(pattern, line, re.IGNORECASE)
            if matches:
                amount = matches[0].replace(',', '')
                if amount.replace('.', '').isdigit():
                    return amount
        return None
    
    def _extract_category_description(self, context: str) -> str:
        """Extract description of product/service category"""
        lines = context.split('\n')
        
        # Look for lines that contain description-related keywords
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['description', 'category', 'business', 'activity']):
                # Clean the line and extract meaningful content
                cleaned = re.sub(r'^[•\-\*\d\.]+\s*', '', line.strip())
                if len(cleaned) > 10:  # Ensure it's substantial
                    return cleaned
        
        return "Information not found"
    
    def _extract_service_description(self, context: str) -> str:
        """Extract description of the product/service"""
        lines = context.split('\n')
        
        # Look for lines that contain service description
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['product', 'service', 'offering', 'business']):
                if 'description' in line_lower or 'details' in line_lower:
                    cleaned = re.sub(r'^[•\-\*\d\.]+\s*', '', line.strip())
                    if len(cleaned) > 10:
                        return cleaned
        
        return "Information not found"
    
    def _extract_generic(self, context: str, question: str) -> str:
        """Generic extraction for other question types"""
        # Extract key terms from the question
        key_terms = re.findall(r'\b\w{4,}\b', question.lower())
        
        lines = context.split('\n')
        best_match = ""
        max_score = 0
        
        for line in lines:
            line_lower = line.lower()
            score = sum(1 for term in key_terms if term in line_lower)
            
            if score > max_score and len(line.strip()) > 10:
                max_score = score
                best_match = line.strip()
        
        return best_match if best_match else "Information not found"

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.answer_extractor = AnswerExtractor()
        
        # Define the specific regulatory questions
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
        
        # Enhanced keywords for better matching
        self.question_keywords = {
            "Corporate identity number (CIN) of company": ["CIN", "corporate identity", "registration", "L", "U"],
            "Financial year to which financial statements relates": ["financial year", "FY", "year ended", "period ended", "fiscal year"],
            "Product or service category code (ITC/ NPCS 4 digit code)": ["ITC", "NPCS", "product code", "service code", "4 digit", "category code"],
            "Description of the product or service category": ["product category", "service category", "business segment", "description", "activity"],
            "Turnover of the product or service category (in Rupees)": ["category turnover", "segment turnover", "revenue", "sales", "turnover"],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": ["8 digit", "highest turnover", "main product", "primary", "contributing"],
            "Description of the product or service": ["product description", "service description", "main offering", "primary business"],
            "Turnover of highest contributing product or service (in Rupees)": ["highest turnover", "main product turnover", "primary revenue", "maximum", "contributing"]
        }
        
        # Page preferences (first 2 questions typically on page 1, rest on page 10)
        self.page_preferences = {
            0: [1, 2, 3],  # CIN - usually on first few pages
            1: [1, 2, 3],  # Financial year - usually on first few pages
            2: [10, 9, 11, 8, 12],  # Product code - usually on page 10
            3: [10, 9, 11, 8, 12],  # Product description - usually on page 10
            4: [10, 9, 11, 8, 12],  # Category turnover - usually on page 10
            5: [10, 9, 11, 8, 12],  # Highest product code - usually on page 10
            6: [10, 9, 11, 8, 12],  # Service description - usually on page 10
            7: [10, 9, 11, 8, 12],  # Highest turnover - usually on page 10
        }
    
    def process_pdf(self, pdf_path: str, debug_mode: bool = False) -> Dict[str, str]:
        """Process a PDF file and extract regulatory information"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            answers = {}
            
            if debug_mode:
                print(f"\n{'='*80}")
                print(f"DEBUG MODE - PDF: {os.path.basename(pdf_path)}")
                print(f"{'='*80}")
                print(f"Total pages in PDF: {len(pages)}")
            
            # Process each question
            for i, question in enumerate(self.questions):
                question_start = time.time()
                
                if debug_mode:
                    print(f"\n--- Question {i+1}: {question} ---")
                
                # Get keywords for this question
                keywords = self.question_keywords.get(question, [])
                
                # Search for relevant pages
                relevant_page_nums = self.pdf_extractor.search_keywords_in_pdf(pages, keywords, question)
                
                # If no relevant pages found, use page preferences
                if not relevant_page_nums:
                    preferred_pages = self.page_preferences.get(i, [1, 2, 3])
                    relevant_page_nums = [p for p in preferred_pages if p <= len(pages)]
                    if debug_mode:
                        print(f"  No keyword matches found, using preferred pages: {relevant_page_nums}")
                
                # Extract context from relevant pages
                context = ""
                for page_num in relevant_page_nums[:3]:  # Limit to first 3 relevant pages
                    page_data = next((p for p in pages if p["page_num"] == page_num), None)
                    if page_data:
                        context += f"\n--- Page {page_num} ---\n"
                        context += page_data["text"]
                
                if debug_mode:
                    print(f"  Context length: {len(context)} characters")
                    print(f"  First 200 chars: {context[:200]}...")
                
                # Extract answer
                answer = self.answer_extractor.extract_answer(question, context)
                answers[question] = answer
                
                if debug_mode:
                    print(f"  Answer: {answer}")
                    print(f"  Time taken: {time.time() - question_start:.2f} seconds")
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str, debug_mode: bool = False):
        """Process multiple PDF files and save results to Excel"""
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
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                answers = self.process_pdf(pdf_path, debug_mode)
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                result = {"PDF Filename": pdf_file}
                result.update({question: f"Error: {str(e)}" for question in self.questions})
                results.append(result)
        
        # Save results to Excel
        if results:
            df = pd.DataFrame(results)
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            self._display_summary(df)
        
        print(f"Total batch processing completed in {time.time() - batch_start_time:.2f} seconds")
    
    def _display_summary(self, df: pd.DataFrame):
        """Display extraction summary"""
        print("\n" + "="*80)
        print("EXTRACTION SUMMARY")
        print("="*80)
        print(f"Total PDFs processed: {len(df)}")
        
        for question in self.questions:
            found_count = len(df[df[question] != "Information not found"])
            success_rate = (found_count / len(df)) * 100
            print(f"{question[:50]}... : {found_count}/{len(df)} ({success_rate:.1f}%)")
        
        print("="*80)

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_extraction_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    try:
        print("Starting PDF Regulatory Information Extraction")
        pipeline = PDFRegulatoryExtractor()
        
        if args.single_pdf:
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
            
            # Process single PDF
            results = pipeline.process_pdf(args.single_pdf, args.debug)
            
            # Save results
            df_data = [{"PDF Filename": os.path.basename(args.single_pdf), **results}]
            df = pd.DataFrame(df_data)
            columns = ["PDF Filename"] + pipeline.questions
            df = df[columns]
            df.to_excel(args.output, index=False)
            print(f"Results saved to {args.output}")
            
        else:
            # Process directory
            pipeline.process_pdfs_batch(args.pdf_dir, args.output, args.debug)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
