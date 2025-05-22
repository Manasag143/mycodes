import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import unicodedata

@dataclass
class ExtractionResult:
    """Data class for storing extraction results"""
    value: str
    confidence: float
    page_number: int
    context: str = ""
    extraction_method: str = ""

class TextProcessor:
    """Advanced text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace and normalize unicode
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    @staticmethod
    def extract_tables(text: str) -> List[List[str]]:
        """Extract table-like structures from text"""
        lines = text.split('\n')
        tables = []
        current_table = []
        
        for line in lines:
            # Check if line looks like a table row (has multiple whitespace-separated values)
            if re.search(r'\s{2,}', line) and len(line.split()) >= 2:
                # Split by multiple whitespaces
                row = re.split(r'\s{2,}', line.strip())
                current_table.append(row)
            else:
                if current_table and len(current_table) >= 2:
                    tables.append(current_table)
                current_table = []
        
        if current_table and len(current_table) >= 2:
            tables.append(current_table)
        
        return tables
    
    @staticmethod
    def find_key_value_pairs(text: str) -> Dict[str, str]:
        """Extract key-value pairs from text"""
        pairs = {}
        
        # Pattern for key: value
        pattern1 = r'([A-Za-z\s]+?):\s*([^\n:]+?)(?=\n|$|[A-Za-z\s]+:)'
        matches1 = re.finditer(pattern1, text)
        for match in matches1:
            key = match.group(1).strip()
            value = match.group(2).strip()
            if len(key) > 2 and len(value) > 0:
                pairs[key.lower()] = value
        
        # Pattern for key followed by value on next line or same line
        pattern2 = r'([A-Za-z\s]{5,}?)[\s]*\n?[\s]*([A-Z0-9][^\n]+?)(?=\n|$)'
        matches2 = re.finditer(pattern2, text)
        for match in matches2:
            key = match.group(1).strip()
            value = match.group(2).strip()
            if len(key) > 4 and len(value) > 2 and not key.lower() in pairs:
                pairs[key.lower()] = value
        
        return pairs

class AdvancedRuleBasedExtractor:
    """Enhanced rule-based extractor with multiple extraction strategies"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.patterns = self._initialize_comprehensive_patterns()
        self.keywords = self._initialize_keywords()
    
    def _initialize_comprehensive_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize comprehensive regex patterns"""
        return {
            "cin": [
                # Direct CIN patterns
                {
                    "pattern": r"(?i)(?:CIN|Corporate\s+Identity\s+Number|Corporate\s+Identification\s+Number|Registration\s+Number)[\s:]*([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})",
                    "confidence": 0.95,
                    "method": "direct_label"
                },
                {
                    "pattern": r"\b([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b",
                    "confidence": 0.90,
                    "method": "format_match"
                },
                # In parentheses or brackets
                {
                    "pattern": r"[\(\[]([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})[\)\]]",
                    "confidence": 0.85,
                    "method": "parentheses"
                }
            ],
            "financial_year": [
                # Various FY patterns
                {
                    "pattern": r"(?i)(?:financial\s+year|FY|F\.Y\.|year\s+ended|period\s+ended|reporting\s+period)[\s:]*(\d{4}[-/]\d{2,4})",
                    "confidence": 0.95,
                    "method": "direct_label"
                },
                {
                    "pattern": r"(?i)(?:for\s+the\s+year\s+ended|as\s+at|as\s+on)[\s]*(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})",
                    "confidence": 0.90,
                    "method": "date_format"
                },
                {
                    "pattern": r"(?i)(?:March|march)\s+(\d{4})",
                    "confidence": 0.85,
                    "method": "march_year"
                },
                {
                    "pattern": r"\b(\d{4}[-/]\d{2,4})\b",
                    "confidence": 0.70,
                    "method": "standalone_year"
                }
            ],
            "product_codes": [
                # 4 and 8 digit codes
                {
                    "pattern": r"(?i)(?:ITC|NPCS|HSN|Product\s+Code|Classification\s+Code|Code)[\s:]*(\d{4,8})",
                    "confidence": 0.90,
                    "method": "direct_label"
                },
                {
                    "pattern": r"(?i)(?:under\s+code|code\s+number|classification)[\s:]*(\d{4,8})",
                    "confidence": 0.85,
                    "method": "contextual"
                },
                {
                    "pattern": r"\b(\d{4})\b|\b(\d{8})\b",
                    "confidence": 0.60,
                    "method": "numeric_match"
                }
            ],
            "turnover_amounts": [
                # Comprehensive turnover patterns
                {
                    "pattern": r"(?i)(?:turnover|revenue|sales|income)[\s:]*(?:Rs\.?|₹|INR)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(crores?|lakhs?|millions?|billions?|thousand)?",
                    "confidence": 0.90,
                    "method": "labeled_amount"
                },
                {
                    "pattern": r"(?:Rs\.?|₹|INR)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(crores?|lakhs?|millions?|billions?)?",
                    "confidence": 0.80,
                    "method": "currency_amount"
                },
                {
                    "pattern": r"\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(crores?|lakhs?|millions?|billions?)",
                    "confidence": 0.75,
                    "method": "amount_unit"
                },
                {
                    "pattern": r"\b(\d{6,})\b",
                    "confidence": 0.60,
                    "method": "large_number"
                }
            ]
        }
    
    def _initialize_keywords(self) -> Dict[str, List[str]]:
        """Initialize keyword mappings for different fields"""
        return {
            "cin": ["cin", "corporate identity", "registration number", "company registration"],
            "financial_year": ["financial year", "fy", "year ended", "period ended", "reporting period", "annual report"],
            "product_category": ["product category", "service category", "business segment", "main business", "principal activity"],
            "product_description": ["description", "nature of business", "main activity", "principal business"],
            "product_codes": ["itc", "npcs", "hsn", "product code", "classification", "commodity"],
            "turnover": ["turnover", "revenue", "sales", "income", "gross revenue", "net revenue"],
            "highest_turnover": ["highest", "maximum", "main", "primary", "principal", "major"]
        }
    
    def extract_information(self, pages: List[Dict[str, Any]], info_type: str, **kwargs) -> ExtractionResult:
        """Main extraction method that tries multiple approaches"""
        
        # Strategy 1: Direct pattern matching
        result = self._extract_with_patterns(pages, info_type, **kwargs)
        if result.confidence > 0.7:
            return result
        
        # Strategy 2: Table extraction
        table_result = self._extract_from_tables(pages, info_type, **kwargs)
        if table_result.confidence > result.confidence:
            result = table_result
        
        # Strategy 3: Key-value pair extraction
        kv_result = self._extract_from_key_value_pairs(pages, info_type, **kwargs)
        if kv_result.confidence > result.confidence:
            result = kv_result
        
        # Strategy 4: Contextual search
        context_result = self._extract_with_context_search(pages, info_type, **kwargs)
        if context_result.confidence > result.confidence:
            result = context_result
        
        return result if result.confidence > 0 else ExtractionResult("Information not found", 0.0, 0)
    
    def _extract_with_patterns(self, pages: List[Dict[str, Any]], info_type: str, **kwargs) -> ExtractionResult:
        """Extract using regex patterns"""
        pattern_key = self._get_pattern_key(info_type, **kwargs)
        if pattern_key not in self.patterns:
            return ExtractionResult("Information not found", 0.0, 0)
        
        best_result = ExtractionResult("Information not found", 0.0, 0)
        
        # Determine page range based on info type
        page_range = self._get_page_range(info_type, len(pages))
        
        for page in pages[page_range[0]:page_range[1]]:
            text = self.text_processor.clean_text(page["text"])
            
            for pattern_info in self.patterns[pattern_key]:
                matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    extracted_value = self._extract_match_value(match, info_type, **kwargs)
                    if extracted_value and self._validate_extraction(extracted_value, info_type, **kwargs):
                        normalized_value = self._normalize_value(extracted_value, info_type)
                        context = self._get_context(text, match.start(), match.end())
                        
                        result = ExtractionResult(
                            value=normalized_value,
                            confidence=pattern_info["confidence"],
                            page_number=page["page_num"],
                            context=context,
                            extraction_method=f"pattern_{pattern_info['method']}"
                        )
                        
                        if result.confidence > best_result.confidence:
                            best_result = result
        
        return best_result
    
    def _extract_from_tables(self, pages: List[Dict[str, Any]], info_type: str, **kwargs) -> ExtractionResult:
        """Extract information from table structures"""
        best_result = ExtractionResult("Information not found", 0.0, 0)
        
        for page in pages:
            tables = self.text_processor.extract_tables(page["text"])
            
            for table in tables:
                result = self._search_in_table(table, info_type, page["page_num"], **kwargs)
                if result.confidence > best_result.confidence:
                    best_result = result
        
        return best_result
    
    def _extract_from_key_value_pairs(self, pages: List[Dict[str, Any]], info_type: str, **kwargs) -> ExtractionResult:
        """Extract from key-value pair structures"""
        best_result = ExtractionResult("Information not found", 0.0, 0)
        
        for page in pages:
            kv_pairs = self.text_processor.find_key_value_pairs(page["text"])
            
            for key, value in kv_pairs.items():
                if self._is_relevant_key(key, info_type):
                    if self._validate_extraction(value, info_type, **kwargs):
                        normalized_value = self._normalize_value(value, info_type)
                        
                        result = ExtractionResult(
                            value=normalized_value,
                            confidence=0.80,
                            page_number=page["page_num"],
                            context=f"{key}: {value}",
                            extraction_method="key_value_pair"
                        )
                        
                        if result.confidence > best_result.confidence:
                            best_result = result
        
        return best_result
    
    def _extract_with_context_search(self, pages: List[Dict[str, Any]], info_type: str, **kwargs) -> ExtractionResult:
        """Extract using contextual keyword search"""
        keywords = self.keywords.get(self._get_keyword_group(info_type), [])
        best_result = ExtractionResult("Information not found", 0.0, 0)
        
        for page in pages:
            text = page["text"].lower()
            
            for keyword in keywords:
                positions = [m.start() for m in re.finditer(re.escape(keyword), text)]
                
                for pos in positions:
                    # Extract surrounding context
                    start = max(0, pos - 200)
                    end = min(len(text), pos + 200)
                    context = text[start:end]
                    
                    # Apply patterns to this context
                    result = self._extract_from_context(context, info_type, page["page_num"], **kwargs)
                    if result.confidence > best_result.confidence:
                        best_result = result
        
        return best_result
    
    def _search_in_table(self, table: List[List[str]], info_type: str, page_num: int, **kwargs) -> ExtractionResult:
        """Search for information within a table structure"""
        keywords = self.keywords.get(self._get_keyword_group(info_type), [])
        
        for row_idx, row in enumerate(table):
            for col_idx, cell in enumerate(row):
                cell_lower = cell.lower()
                
                # Check if this cell contains relevant keywords
                for keyword in keywords:
                    if keyword in cell_lower:
                        # Look for value in adjacent cells
                        candidates = []
                        
                        # Same row, next column
                        if col_idx + 1 < len(row):
                            candidates.append(row[col_idx + 1])
                        
                        # Next row, same column
                        if row_idx + 1 < len(table) and col_idx < len(table[row_idx + 1]):
                            candidates.append(table[row_idx + 1][col_idx])
                        
                        # Next row, next column
                        if (row_idx + 1 < len(table) and 
                            col_idx + 1 < len(table[row_idx + 1])):
                            candidates.append(table[row_idx + 1][col_idx + 1])
                        
                        for candidate in candidates:
                            if self._validate_extraction(candidate, info_type, **kwargs):
                                normalized_value = self._normalize_value(candidate, info_type)
                                return ExtractionResult(
                                    value=normalized_value,
                                    confidence=0.75,
                                    page_number=page_num,
                                    context=f"Table: {cell} -> {candidate}",
                                    extraction_method="table_search"
                                )
        
        return ExtractionResult("Information not found", 0.0, 0)
    
    def _extract_from_context(self, context: str, info_type: str, page_num: int, **kwargs) -> ExtractionResult:
        """Extract information from a context snippet"""
        pattern_key = self._get_pattern_key(info_type, **kwargs)
        if pattern_key not in self.patterns:
            return ExtractionResult("Information not found", 0.0, 0)
        
        for pattern_info in self.patterns[pattern_key]:
            matches = re.finditer(pattern_info["pattern"], context, re.IGNORECASE)
            
            for match in matches:
                extracted_value = self._extract_match_value(match, info_type, **kwargs)
                if extracted_value and self._validate_extraction(extracted_value, info_type, **kwargs):
                    normalized_value = self._normalize_value(extracted_value, info_type)
                    
                    return ExtractionResult(
                        value=normalized_value,
                        confidence=pattern_info["confidence"] * 0.9,  # Slightly lower confidence
                        page_number=page_num,
                        context=context[:100] + "..." if len(context) > 100 else context,
                        extraction_method=f"context_{pattern_info['method']}"
                    )
        
        return ExtractionResult("Information not found", 0.0, 0)
    
    # Helper methods
    def _get_pattern_key(self, info_type: str, **kwargs) -> str:
        """Get the pattern key for a given info type"""
        if info_type == "cin":
            return "cin"
        elif info_type == "financial_year":
            return "financial_year"
        elif info_type in ["product_code_4", "product_code_8"]:
            return "product_codes"
        elif info_type == "turnover":
            return "turnover_amounts"
        else:
            return "product_codes"  # Default
    
    def _get_keyword_group(self, info_type: str) -> str:
        """Get keyword group for info type"""
        mapping = {
            "cin": "cin",
            "financial_year": "financial_year",
            "product_code_4": "product_codes",
            "product_code_8": "product_codes",
            "description_category": "product_category",
            "description_product": "product_description",
            "turnover": "turnover",
            "turnover_highest": "highest_turnover"
        }
        return mapping.get(info_type, "product_codes")
    
    def _get_page_range(self, info_type: str, total_pages: int) -> Tuple[int, int]:
        """Get the page range to search for different info types"""
        if info_type in ["cin", "financial_year"]:
            return (0, min(5, total_pages))  # First 5 pages
        elif info_type in ["product_code_4", "product_code_8", "description_category", "description_product"]:
            return (0, min(15, total_pages))  # First 15 pages
        else:  # turnover information
            return (0, total_pages)  # All pages
    
    def _extract_match_value(self, match, info_type: str, **kwargs) -> str:
        """Extract the relevant value from a regex match"""
        groups = match.groups()
        if not groups:
            return match.group(0)
        
        # For most patterns, the first group contains the value
        for group in groups:
            if group and group.strip():
                return group.strip()
        
        return match.group(0)
    
    def _validate_extraction(self, value: str, info_type: str, **kwargs) -> bool:
        """Validate extracted value based on info type"""
        if not value or len(value.strip()) == 0:
            return False
        
        value = value.strip()
        
        if info_type == "cin":
            return len(value) == 21 and re.match(r'^[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}$', value)
        
        elif info_type == "financial_year":
            return (re.match(r'\d{4}[-/]\d{2,4}', value) or 
                   re.match(r'\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}', value) or
                   re.match(r'\d{4}', value))
        
        elif info_type == "product_code_4":
            return len(value) == 4 and value.isdigit()
        
        elif info_type == "product_code_8":
            return len(value) == 8 and value.isdigit()
        
        elif info_type == "turnover":
            # Remove commas and check if it's a valid number
            clean_value = re.sub(r'[,\s]', '', value)
            try:
                float(clean_value)
                return True
            except ValueError:
                return False
        
        elif info_type in ["description_category", "description_product"]:
            return len(value.split()) >= 2 and not value.isdigit()
        
        return True
    
    def _normalize_value(self, value: str, info_type: str) -> str:
        """Normalize extracted value"""
        value = value.strip()
        
        if info_type == "financial_year":
            # Convert to YYYY-YY format
            if re.match(r'\d{4}[-/]\d{4}', value):
                parts = re.split(r'[-/]', value)
                return f"{parts[0]}-{parts[1][-2:]}"
            elif re.match(r'\d{4}', value):
                year = int(value)
                return f"{year}-{str(year+1)[-2:]}"
        
        elif info_type == "turnover":
            # Remove commas but keep the number clean
            return re.sub(r'[,\s]', '', value)
        
        return value
    
    def _is_relevant_key(self, key: str, info_type: str) -> bool:
        """Check if a key is relevant for the info type"""
        keywords = self.keywords.get(self._get_keyword_group(info_type), [])
        key_lower = key.lower()
        
        return any(keyword in key_lower for keyword in keywords)
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Get context around a match"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

class PDFExtractor:
    """Enhanced PDF text extraction"""
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with better structure preservation"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num, page in enumerate(doc):
                # Get text with layout preservation
                text = page.get_text()
                
                # Also try to get text blocks for better structure
                blocks = page.get_text("blocks")
                structured_text = ""
                for block in blocks:
                    if len(block) >= 5:  # Text block
                        structured_text += block[4] + "\n"
                
                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "structured_text": structured_text,
                    "blocks": blocks
                })
            
            doc.close()
            print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages
            
        except Exception as e:
            print(f"PDF extraction error: {e}")
            raise

class PDFRegulatoryExtractor:
    """Enhanced main extractor class"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.rule_extractor = AdvancedRuleBasedExtractor()
        
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
        """Process PDF with enhanced extraction"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            answers = {}
            
            # Extract each piece of information
            print("Extracting CIN...")
            cin_result = self.rule_extractor.extract_information(pages, "cin")
            answers[self.questions[0]] = cin_result.value
            print(f"CIN: {cin_result.value} (confidence: {cin_result.confidence:.2f})")
            
            print("Extracting Financial Year...")
            fy_result = self.rule_extractor.extract_information(pages, "financial_year")
            answers[self.questions[1]] = fy_result.value
            print(f"FY: {fy_result.value} (confidence: {fy_result.confidence:.2f})")
            
            print("Extracting 4-digit product code...")
            code4_result = self.rule_extractor.extract_information(pages, "product_code_4")
            answers[self.questions[2]] = code4_result.value
            print(f"4-digit code: {code4_result.value} (confidence: {code4_result.confidence:.2f})")
            
            print("Extracting product category description...")
            desc_cat_result = self.rule_extractor.extract_information(pages, "description_category")
            answers[self.questions[3]] = desc_cat_result.value
            print(f"Category desc: {desc_cat_result.value[:50]}... (confidence: {desc_cat_result.confidence:.2f})")
            
            print("Extracting category turnover...")
            turnover_cat_result = self.rule_extractor.extract_information(pages, "turnover")
            answers[self.questions[4]] = turnover_cat_result.value
            print(f"Category turnover: {turnover_cat_result.value} (confidence: {turnover_cat_result.confidence:.2f})")
            
            print("Extracting 8-digit product code...")
            code8_result = self.rule_extractor.extract_information(pages, "product_code_8")
            answers[self.questions[5]] = code8_result.value
            print(f"8-digit code: {code8_result.value} (confidence: {code8_result.confidence:.2f})")
            
            print("Extracting product description...")
            desc_prod_result = self.rule_extractor.extract_information(pages, "description_product")
            answers[self.questions[6]] = desc_prod_result.value
            print(f"Product desc: {desc_prod_result.value[:50]}... (confidence: {desc_prod_result.confidence:.2f})")
            
            print("Extracting highest turnover...")
            turnover_high_result = self.rule_extractor.extract_information(pages, "turnover_highest")
            answers[self.questions[7]] = turnover_high_result.value
            print(f"Highest turnover: {turnover_high_result.value} (confidence: {turnover_high_result.confidence:.2f})")
            
            print(f"Completed processing in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """Process multiple PDFs"""
        batch_start_time = time.time()
        print(f"Processing PDFs in: {pdf_dir}")
        
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
                print(f"\n--- Processing {pdf_file} ---")
                answers = self.process_pdf(pdf_path)
                
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                result = {"PDF Filename": pdf_file}
                result.update({question: f"Error: {str(e)}" for question in self.questions})
                results.append(result)
        
        if results:
            df = pd.DataFrame(results)
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            df.to_excel(output_excel, index=False)
            print(f"\nResults saved to {output_excel}")
        
        print(f"Total processing time: {time.time() - batch_start_time:.2f} seconds")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF")
    args = parser.parse_args()
    
    try:
        print("Starting Enhanced PDF Regulatory Information Extraction")
        start_time = time.time()
        
        pipeline = PDFRegulatoryExtractor()
        
        if args.single_pdf:
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
                
            # Process single PDF
            results = []
            pdf_file = os.path.basename(args.single_pdf)
            
            try:
                answers = pipeline.process_pdf(args.single_pdf)
                
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                results.append(result)
                
                # Save results
                df = pd.DataFrame(results)
                columns = ["PDF Filename"] + pipeline.questions
                df = df[columns]
                df.to_excel(args.output, index=False)
                print(f"\nResults saved to {args.output}")
                
                # Print detailed results
                print("\n" + "="*80)
                print("EXTRACTION RESULTS")
                print("="*80)
                for question, answer in answers.items():
                    print(f"\n{question}:")
                    print(f"  Answer: {answer}")
                print("="*80)
                
            except Exception as e:
                print(f"Error processing {args.single_pdf}: {e}")
        else:
            # Process directory of PDFs
            pipeline.process_pdfs_batch(args.pdf_dir, args.output)
        
        print(f"\nProcessing completed successfully in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
