import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ExtractionResult:
    """Data class for storing extraction results"""
    value: str
    confidence: float
    page_number: int
    context: str = ""

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
                    "text": text,
                    "original_text": text  # Keep original for context
                })
                
            doc.close()
            print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages
        except Exception as e:
            print(f"PDF extraction error after {time.time() - start_time:.2f} seconds: {e}")
            raise

class RuleBasedExtractor:
    """Rule-based extractor for regulatory information"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize regex patterns for different types of information"""
        return {
            "cin": [
                {
                    "pattern": r"(?:CIN|Corporate\s+Identity\s+Number|Corporate\s+Identification\s+Number)[\s:]*([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})",
                    "confidence": 0.95
                },
                {
                    "pattern": r"\b([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b",
                    "confidence": 0.85
                }
            ],
            "financial_year": [
                {
                    "pattern": r"(?:financial\s+year|FY|F\.Y\.|year\s+ended|period\s+ended)[\s:]*(\d{4}[-/]\d{2,4})",
                    "confidence": 0.90
                },
                {
                    "pattern": r"(?:for\s+the\s+year\s+ended|March)\s+(\d{1,2}[a-z]*\s+\w+\s+\d{4})",
                    "confidence": 0.85
                },
                {
                    "pattern": r"\b(\d{4}[-/]\d{2,4})\b",
                    "confidence": 0.70
                }
            ],
            "product_code_4digit": [
                {
                    "pattern": r"(?:ITC|NPCS|Product\s+Code|Classification\s+Code)[\s:]*(\d{4})",
                    "confidence": 0.90
                },
                {
                    "pattern": r"(?:Code|HSN)[\s:]*(\d{4})\b",
                    "confidence": 0.80
                }
            ],
            "product_code_8digit": [
                {
                    "pattern": r"(?:ITC|NPCS|Product\s+Code|Classification\s+Code)[\s:]*(\d{8})",
                    "confidence": 0.90
                },
                {
                    "pattern": r"(?:Code|HSN)[\s:]*(\d{8})\b",
                    "confidence": 0.80
                }
            ],
            "turnover": [
                {
                    "pattern": r"(?:turnover|revenue|sales)[\s:]*(?:Rs\.?|₹|INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:crores?|lakhs?|millions?|billions?)?",
                    "confidence": 0.85
                },
                {
                    "pattern": r"(?:Rs\.?|₹|INR)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
                    "confidence": 0.75
                },
                {
                    "pattern": r"\b(\d{1,3}(?:,\d{3})*)\s*(?:crores?|lakhs?)",
                    "confidence": 0.80
                }
            ],
            "description": [
                {
                    "pattern": r"(?:description|business|activity|main\s+activity|principal\s+activity)[\s:]*([A-Za-z\s,.-]+?)(?:\n|\.|;|$)",
                    "confidence": 0.80
                }
            ]
        }
    
    def extract_cin(self, pages: List[Dict[str, Any]]) -> ExtractionResult:
        """Extract Corporate Identity Number"""
        for page in pages[:5]:  # Usually in first few pages
            text = page["text"]
            for pattern_info in self.patterns["cin"]:
                matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
                for match in matches:
                    cin = match.group(1)
                    if self._validate_cin(cin):
                        context = self._extract_context(text, match.start(), match.end())
                        return ExtractionResult(
                            value=cin,
                            confidence=pattern_info["confidence"],
                            page_number=page["page_num"],
                            context=context
                        )
        
        return ExtractionResult("Information not found", 0.0, 0)
    
    def extract_financial_year(self, pages: List[Dict[str, Any]]) -> ExtractionResult:
        """Extract Financial Year"""
        for page in pages[:10]:  # Check more pages for financial year
            text = page["text"]
            for pattern_info in self.patterns["financial_year"]:
                matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
                for match in matches:
                    fy = match.group(1)
                    if self._validate_financial_year(fy):
                        context = self._extract_context(text, match.start(), match.end())
                        return ExtractionResult(
                            value=self._normalize_financial_year(fy),
                            confidence=pattern_info["confidence"],
                            page_number=page["page_num"],
                            context=context
                        )
        
        return ExtractionResult("Information not found", 0.0, 0)
    
    def extract_product_code(self, pages: List[Dict[str, Any]], digit_count: int = 4) -> ExtractionResult:
        """Extract product/service code (4 or 8 digit)"""
        pattern_key = f"product_code_{digit_count}digit"
        
        for page in pages:
            text = page["text"]
            for pattern_info in self.patterns[pattern_key]:
                matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
                for match in matches:
                    code = match.group(1)
                    if len(code) == digit_count:
                        context = self._extract_context(text, match.start(), match.end())
                        return ExtractionResult(
                            value=code,
                            confidence=pattern_info["confidence"],
                            page_number=page["page_num"],
                            context=context
                        )
        
        return ExtractionResult("Information not found", 0.0, 0)
    
    def extract_turnover(self, pages: List[Dict[str, Any]], search_terms: List[str] = None) -> ExtractionResult:
        """Extract turnover information"""
        if search_terms is None:
            search_terms = ["turnover", "revenue", "sales"]
        
        best_result = ExtractionResult("Information not found", 0.0, 0)
        
        for page in pages:
            text = page["text"]
            
            # Look for sections containing search terms
            for term in search_terms:
                term_positions = [m.start() for m in re.finditer(term, text, re.IGNORECASE)]
                
                for pos in term_positions:
                    # Extract surrounding text (200 chars before and after)
                    start = max(0, pos - 200)
                    end = min(len(text), pos + 200)
                    context_text = text[start:end]
                    
                    # Look for turnover patterns in this context
                    for pattern_info in self.patterns["turnover"]:
                        matches = re.finditer(pattern_info["pattern"], context_text, re.IGNORECASE)
                        for match in matches:
                            amount = match.group(1)
                            if self._validate_turnover(amount):
                                normalized_amount = self._normalize_turnover(amount)
                                context = self._extract_context(context_text, match.start(), match.end())
                                
                                result = ExtractionResult(
                                    value=normalized_amount,
                                    confidence=pattern_info["confidence"],
                                    page_number=page["page_num"],
                                    context=context
                                )
                                
                                if result.confidence > best_result.confidence:
                                    best_result = result
        
        return best_result
    
    def extract_description(self, pages: List[Dict[str, Any]], search_terms: List[str] = None) -> ExtractionResult:
        """Extract product/service description"""
        if search_terms is None:
            search_terms = ["description", "business", "activity", "main activity", "principal activity"]
        
        for page in pages:
            text = page["text"]
            
            for term in search_terms:
                term_positions = [m.start() for m in re.finditer(term, text, re.IGNORECASE)]
                
                for pos in term_positions:
                    # Extract surrounding text
                    start = max(0, pos - 50)
                    end = min(len(text), pos + 300)
                    context_text = text[start:end]
                    
                    for pattern_info in self.patterns["description"]:
                        matches = re.finditer(pattern_info["pattern"], context_text, re.IGNORECASE)
                        for match in matches:
                            description = match.group(1).strip()
                            if len(description) > 10 and self._validate_description(description):
                                context = self._extract_context(context_text, match.start(), match.end())
                                return ExtractionResult(
                                    value=description,
                                    confidence=pattern_info["confidence"],
                                    page_number=page["page_num"],
                                    context=context
                                )
        
        return ExtractionResult("Information not found", 0.0, 0)
    
    def _validate_cin(self, cin: str) -> bool:
        """Validate CIN format"""
        if len(cin) != 21:
            return False
        if not re.match(r'^[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}$', cin):
            return False
        return True
    
    def _validate_financial_year(self, fy: str) -> bool:
        """Validate financial year format"""
        if re.match(r'\d{4}[-/]\d{2,4}', fy):
            return True
        if re.match(r'\d{1,2}[a-z]*\s+\w+\s+\d{4}', fy):
            return True
        return False
    
    def _validate_turnover(self, amount: str) -> bool:
        """Validate turnover amount"""
        try:
            # Remove commas and convert to float
            clean_amount = amount.replace(',', '')
            float(clean_amount)
            return True
        except ValueError:
            return False
    
    def _validate_description(self, description: str) -> bool:
        """Validate description"""
        # Check if description contains meaningful content
        if len(description.split()) < 3:
            return False
        # Avoid descriptions that are just numbers or codes
        if re.match(r'^\d+$', description.strip()):
            return False
        return True
    
    def _normalize_financial_year(self, fy: str) -> str:
        """Normalize financial year format"""
        # Convert various formats to YYYY-YY
        if re.match(r'\d{4}[-/]\d{2}', fy):
            return fy.replace('/', '-')
        elif re.match(r'\d{4}[-/]\d{4}', fy):
            parts = re.split(r'[-/]', fy)
            return f"{parts[0]}-{parts[1][-2:]}"
        return fy
    
    def _normalize_turnover(self, amount: str) -> str:
        """Normalize turnover amount"""
        # Remove commas and return clean number
        return amount.replace(',', '')
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around a match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs without LLM"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.rule_extractor = RuleBasedExtractor()
        
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
        """Process a PDF file and extract regulatory information"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            # Dictionary to store answers
            answers = {}
            
            # Extract CIN
            cin_result = self.rule_extractor.extract_cin(pages)
            answers[self.questions[0]] = cin_result.value
            
            # Extract Financial Year
            fy_result = self.rule_extractor.extract_financial_year(pages)
            answers[self.questions[1]] = fy_result.value
            
            # Extract 4-digit product code
            code_4_result = self.rule_extractor.extract_product_code(pages, 4)
            answers[self.questions[2]] = code_4_result.value
            
            # Extract product category description
            desc_category_result = self.rule_extractor.extract_description(
                pages, ["product category", "service category", "business segment", "main business"]
            )
            answers[self.questions[3]] = desc_category_result.value
            
            # Extract category turnover
            turnover_category_result = self.rule_extractor.extract_turnover(
                pages, ["category turnover", "segment turnover", "segment revenue"]
            )
            answers[self.questions[4]] = turnover_category_result.value
            
            # Extract 8-digit product code
            code_8_result = self.rule_extractor.extract_product_code(pages, 8)
            answers[self.questions[5]] = code_8_result.value
            
            # Extract product description
            desc_product_result = self.rule_extractor.extract_description(
                pages, ["product description", "service description", "main product", "principal product"]
            )
            answers[self.questions[6]] = desc_product_result.value
            
            # Extract highest contributing turnover
            turnover_highest_result = self.rule_extractor.extract_turnover(
                pages, ["highest turnover", "main product turnover", "primary revenue", "highest contributing"]
            )
            answers[self.questions[7]] = turnover_highest_result.value
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
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
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                # Get answers for this PDF
                pdf_start_time = time.time()
                answers = self.process_pdf(pdf_path)
                
                # Add filename to results
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                
                results.append(result)
                print(f"Processed {pdf_file} in {time.time() - pdf_start_time:.2f} seconds")
                
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

def main():
    # Simple command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description="Rule-based PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    args = parser.parse_args()
    
    try:
        print("Starting PDF Regulatory Information Extraction (Rule-based)")
        start_time = time.time()
        
        # Create pipeline
        pipeline = PDFRegulatoryExtractor()
        
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
                
                # Print results for debugging
                print("\nExtraction Results:")
                for question, answer in answers.items():
                    print(f"{question}: {answer}")
                
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
