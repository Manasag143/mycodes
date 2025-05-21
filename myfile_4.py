import json
import os
import time
import logging
import argparse
import pandas as pd
import fitz  # PyMuPDF
import re
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
from colorama import Fore, Style, init
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

# Initialize colorama for colored terminal output
init()

# Define the Llama URL
LLAMA_URL = "https://ue1-llm.crisil.local/llama3_3/70b/llm/"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_regulatory_extraction.log"),
        logging.StreamHandler()
    ]
)

class LLMGenerator:
    """
    Component to generate responses from hosted LLM model using Text Generative Inference.
    """
    def __init__(self, url: str = LLAMA_URL):
        self.url = url
        self.generation_kwargs = {
            "max_new_tokens": 5048,
            "return_full_text": False,
            "temperature": 0.1
        }

    def run(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        try:
            # Disable SSL verification warnings
            requests.packages.urllib3.disable_warnings()
            
            # Make the API call
            response = requests.post(self.url, verify=False, json=body)
            print(f"Request status: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"LLM API error: Status {response.status_code}"
                logging.error(error_msg)
                return f"Error: {error_msg}"
            
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                response_text = response_json[0].get('generated_text', '')
                return response_text
            else:
                error_msg = "Unexpected response format from LLM API"
                logging.error(error_msg)
                return f"Error: {error_msg}"
                
        except Exception as e:
            error_msg = f"Error in LLM call: {e}"
            logging.error(error_msg)
            return f"Error: {str(e)}"

class PDFExtractor:
    """Class for extracting text and tables from PDF files using PyMuPDF."""
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from each page of a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and text for each page
        """
        try:
            print(Fore.BLUE + f"Extracting text from {pdf_path}")
            doc = fitz.open(pdf_path)
            
            pages = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                pages.append({
                    "page_num": page_num + 1,  # 1-indexed page numbers for human readability
                    "text": text
                })
                
            print(Fore.GREEN + f"Extracted text from {len(pages)} pages")
            return pages
            
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract potential tables from PDF using heuristics.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page number and table text
        """
        try:
            print(Fore.BLUE + f"Scanning for tables in {pdf_path}")
            doc = fitz.open(pdf_path)
            
            tables = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Look for tabular structures
                # 1. Sections with multiple aligned digits (potential ITC/NPCS codes)
                digit_patterns = re.findall(r'((?:\d+[\s\t]+){2,}[^\n]+)', text)
                
                # 2. Sections with column-like structures
                column_patterns = re.findall(r'([^\n]+[\s\t]{3,}[^\n]+[\s\t]{3,}[^\n]+)', text)
                
                # 3. Look for ITC/NPCS code mentions
                code_patterns = re.findall(r'([^\n]*(ITC|NPCS|HS|NIC)[\s\t]*(code|Code)[^\n]*(?:\n[^\n]+){0,5})', text)
                
                # 4. Look for turnover mentions
                turnover_patterns = re.findall(
                    r'([^\n]*(turnover|revenue|sales)[\s\t]*(?:of|for)?[\s\t]*(?:product|service)?[^\n]*(?:\n[^\n]+){0,5})', 
                    text, 
                    re.IGNORECASE
                )
                
                # 5. Look for CIN pattern
                cin_patterns = re.findall(r'([^\n]*(CIN|Corporate Identity Number)[^\n]*(?:\n[^\n]+){0,3})', text, re.IGNORECASE)
                
                # 6. Look for financial year mentions
                fy_patterns = re.findall(
                    r'([^\n]*(financial year|FY|F\.Y\.|year ended)[^\n]*(?:\n[^\n]+){0,3})', 
                    text, 
                    re.IGNORECASE
                )
                
                # Combine all findings
                if digit_patterns or column_patterns or code_patterns or turnover_patterns or cin_patterns or fy_patterns:
                    table_text = ""
                    
                    if digit_patterns:
                        table_text += "\nPOTENTIAL CODE PATTERNS:\n" + "\n".join(digit_patterns)
                    
                    if column_patterns:
                        table_text += "\nTABULAR STRUCTURES:\n" + "\n".join(column_patterns)
                    
                    if code_patterns:
                        table_text += "\nITC/NPCS CODE MENTIONS:\n" + "\n".join([p[0] for p in code_patterns])
                    
                    if turnover_patterns:
                        table_text += "\nTURNOVER MENTIONS:\n" + "\n".join([p[0] for p in turnover_patterns])
                    
                    if cin_patterns:
                        table_text += "\nCIN MENTIONS:\n" + "\n".join([p[0] for p in cin_patterns])
                    
                    if fy_patterns:
                        table_text += "\nFINANCIAL YEAR MENTIONS:\n" + "\n".join([p[0] for p in fy_patterns])
                    
                    tables.append({
                        "page_num": page_num + 1,
                        "table_text": table_text
                    })
            
            print(Fore.GREEN + f"Found {len(tables)} potential tables/structured content blocks")
            return tables
            
        except Exception as e:
            logging.error(f"Error extracting tables from PDF {pdf_path}: {e}")
            return []
            
    def search_keywords_in_pdf(self, pages: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, List[int]]:
        """
        Search for keywords in the extracted PDF text with enhanced pattern matching.
        
        Args:
            pages: List of dictionaries containing page number and text
            keywords: List of keywords to search for
            
        Returns:
            Dictionary mapping keywords to list of page numbers where they appear
        """
        print(Fore.BLUE + f"Searching for {len(keywords)} keywords in extracted text")
        
        keyword_pages = {}
        for keyword in keywords:
            keyword_pages[keyword] = []
            
            # Create variations and patterns for more robust matching
            variations = [keyword]
            
            # For CIN search, add regex pattern for CIN format
            if keyword.lower() in ["cin", "corporate identity number"]:
                cin_pattern = r'[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}'
                variations.append(cin_pattern)
            
            # For product codes, look for digit patterns
            if "code" in keyword.lower() or "digit" in keyword.lower():
                if "4 digit" in keyword.lower():
                    variations.append(r'\b\d{4}\b')  # 4-digit code pattern
                if "8 digit" in keyword.lower():
                    variations.append(r'\b\d{8}\b')  # 8-digit code pattern
            
            # For financial years, look for year patterns
            if "financial year" in keyword.lower() or "fy" == keyword.lower():
                variations.append(r'\b(FY|F\.Y\.|Financial Year)[ :]?[0-9]{4}[-/ ][0-9]{2,4}\b')
                variations.append(r'\b[0-9]{4}[-/ ][0-9]{2,4}\b')
            
            # For turnover and financial figures
            if any(term in keyword.lower() for term in ["turnover", "revenue", "sales", "rupees"]):
                variations.append(r'(?i)(Rs\.?|₹|INR)[ ]?[0-9,.]+[ ]?(lakhs?|crores?|millions?|billions?)')
                variations.append(r'[0-9,.]+[ ]?(lakhs?|crores?|millions?|billions?)')
            
            for page in pages:
                found = False
                for var in variations:
                    # Determine if it's a regex pattern or simple string
                    if var.startswith(r'\b') or var.startswith(r'[') or var.startswith(r'(?'):
                        # It's a regex pattern
                        if re.search(var, page["text"], re.IGNORECASE | re.MULTILINE):
                            found = True
                            break
                    else:
                        # Simple string search
                        if re.search(re.escape(var), page["text"], re.IGNORECASE):
                            found = True
                            break
                
                if found and page["page_num"] not in keyword_pages[keyword]:
                    keyword_pages[keyword].append(page["page_num"])
                    
        # Print results
        for keyword, page_nums in keyword_pages.items():
            if page_nums:
                print(Fore.CYAN + f"Keyword '{keyword}' found on pages: {page_nums}")
            else:
                print(Fore.YELLOW + f"Keyword '{keyword}' not found in document")
                
        return keyword_pages

class QuestionAnswerProcessor:
    """Class for processing PDF content to answer regulatory information questions."""
    
    def __init__(self, llm_generator: LLMGenerator):
        self.llm_generator = llm_generator
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer to a question using the provided context,
        with post-processing to ensure consistent formatting.
        
        Args:
            question: The question to answer
            context: The context containing the information needed to answer the question
            
        Returns:
            The generated and post-processed answer
        """
        print(Fore.BLUE + f"Generating answer for question: {question}")
        
        # Create a prompt for Llama
        prompt = self._create_qa_prompt(question, context)
        
        # Generate the answer
        raw_answer = self.llm_generator.run(prompt)
        
        # Post-process the answer for consistent formatting
        processed_answer = self._post_process_answer(question, raw_answer)
        
        print(Fore.GREEN + f"Generated answer: {processed_answer[:100]}..." if len(processed_answer) > 100 else processed_answer)
        return processed_answer
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for the Llama model to answer a regulatory question.
        
        Args:
            question: The specific regulatory question to answer
            context: The context containing the information needed to answer the question
            
        Returns:
            The formatted prompt for the LLM
        """
        # Format the prompt for Llama
        sys_message = "You are an expert in extracting business regulatory information from financial documents with high precision and accuracy."
        
        # Create specific instructions based on question type
        specific_instructions = ""
        if "CIN" in question or "corporate identity" in question.lower():
            specific_instructions = """
- For Corporate Identity Number (CIN), look for a code that typically starts with 'L' or 'U' followed by numbers and letters
- A CIN is a 21-digit alphanumeric code (e.g., L12345MH2010PLC123456)
- It may appear in the company information section, header, footer, or directors' report
"""
        elif "financial year" in question.lower():
            specific_instructions = """
- Look for phrases like "for the year ended", "financial year", "FY", followed by dates
- Express the financial year in the format "YYYY-YY" or as stated in the document
- Check the header of financial statements or the directors' report
"""
        elif "4 digit code" in question:
            specific_instructions = """
- Look for 4-digit codes associated with product or service categories
- Check business segment reporting, notes to accounts, or product classification sections
- The code might be labeled as ITC code, NPCS code, HS code, or NIC code
- Only return the 4-digit code itself, not surrounding text
"""
        elif "8 digit code" in question:
            specific_instructions = """
- Look for 8-digit codes associated with specific products or services
- These may appear in detailed product listings or segment reporting sections
- The code might be labeled as ITC code, NPCS code, HS code, or product code
- Only return the 8-digit code itself, not surrounding text
"""
        elif "turnover" in question.lower():
            specific_instructions = """
- Look for financial figures associated with revenue, turnover, or sales
- Extract the exact amount including the unit (e.g., "Rs. 10,000,000" or "₹ 10 crores")
- Pay attention to whether amounts are in thousands, lakhs, crores, or millions
- Check income statement, segment reporting, or directors' report sections
"""
        elif "description" in question.lower():
            specific_instructions = """
- Extract the exact description as stated in the document
- Look in business segment reporting, notes to accounts, or product sections
- The description should match the associated code mentioned elsewhere
"""
        
        # Create prompt for Llama with specific format requirements
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}

You will be provided with a question and context from a PDF document. Your task is to extract the precise answer to the question from the context.

{specific_instructions}
Follow these general guidelines:
- If the answer is explicitly stated in the text, provide that exact answer including any associated codes or figures
- For financial figures, maintain the exact format (e.g., "Rs. 10,00,000" or "₹ 10 crores")
- If the answer requires combining information from different parts of the text, do so accurately
- If the answer is not in the context, respond with "Information not found in the provided context"
- Do not provide explanations, just the direct answer to the question

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Context from PDF:
{context}

Answer the question based only on the information in the context.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def _post_process_answer(self, question: str, answer: str) -> str:
        """
        Post-process the answer to format it consistently or extract specific information.
        
        Args:
            question: The original question
            answer: The raw answer from LLM
            
        Returns:
            Processed answer
        """
        # Remove any explanatory text or prefixes
        cleaned = re.sub(r'^(Answer:|The answer is:|Based on the context,)', '', answer).strip()
        
        # Handle "not found" responses consistently
        if re.search(r'(not found|not provided|not mentioned|couldn\'t find|could not find|no information)', 
                     cleaned, re.IGNORECASE):
            return "Information not found in the provided context"
        
        # Format CIN consistently
        if "CIN" in question or "corporate identity number" in question.lower():
            # Try to extract CIN format (L12345MH2010PLC123456)
            cin_match = re.search(r'[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}', cleaned)
            if cin_match:
                return cin_match.group(0)
        
        # Format financial year consistently
        if "financial year" in question.lower():
            # Look for common financial year patterns
            fy_match = re.search(r'(FY|F\.Y\.|Financial Year)[ :]?([0-9]{4}[-/ ][0-9]{2,4})', cleaned, re.IGNORECASE)
            if fy_match:
                return fy_match.group(2)
            
            # Try another pattern
            fy_match2 = re.search(r'([0-9]{4}[-/ ][0-9]{2,4})', cleaned)
            if fy_match2:
                return fy_match2.group(1)
        
        # Format product codes consistently
        if "code" in question.lower():
            # For 4-digit code
            if "4 digit" in question.lower():
                code_match = re.search(r'\b(\d{4})\b', cleaned)
                if code_match:
                    return code_match.group(1)
            
            # For 8-digit code
            if "8 digit" in question.lower():
                code_match = re.search(r'\b(\d{8})\b', cleaned)
                if code_match:
                    return code_match.group(1)
        
        # Format financial figures consistently
        if "turnover" in question.lower() or "in Rupees" in question:
            # Try to extract amount with currency and units
            turnover_match = re.search(r'(Rs\.?|₹|INR)[ ]?([0-9,.]+)[ ]?(lakhs?|crores?|millions?|billions?)?', 
                                     cleaned, re.IGNORECASE)
            if turnover_match:
                currency = turnover_match.group(1) or "Rs."
                amount = turnover_match.group(2)
                unit = turnover_match.group(3) or ""
                return f"{currency} {amount} {unit}".strip()
            
            # Try just amount and units
            amount_match = re.search(r'([0-9,.]+)[ ]?(lakhs?|crores?|millions?|billions?)', cleaned, re.IGNORECASE)
            if amount_match:
                return f"Rs. {amount_match.group(1)} {amount_match.group(2)}"
        
        # Return the cleaned answer for other questions
        return cleaned

class PDFRegulatoryExtractor:
    """
    Main class for extracting regulatory information from PDFs and answering specific questions.
    """
    def __init__(self, config_path: str = "config1.json"):
        self.config_path = config_path
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
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
        
        # Define keywords for each question to help locate relevant information
        self.question_keywords = {
            "Corporate identity number (CIN) of company": ["CIN", "corporate identity number", "company registration", "L", "U", "registration number"],
            "Financial year to which financial statements relates": ["financial year", "FY", "year ended", "period ended", "statement period", "fiscal year"],
            "Product or service category code (ITC/ NPCS 4 digit code)": ["ITC code", "NPCS code", "product code", "service code", "4 digit", "category code"],
            "Description of the product or service category": ["product category", "service category", "business segment", "category description", "main business"],
            "Turnover of the product or service category (in Rupees)": ["category turnover", "segment turnover", "business turnover", "revenue from", "sales from", "income from"],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": ["8 digit", "product code", "service code", "highest turnover", "main product", "major service"],
            "Description of the product or service": ["product description", "service description", "main offering", "primary product", "key service"],
            "Turnover of highest contributing product or service (in Rupees)": ["product turnover", "service turnover", "product revenue", "service revenue", "contributing", "highest sales"]
        }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Process a PDF file to extract regulatory information and answer all questions.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping questions to answers
        """
        print(Fore.CYAN + "=" * 80)
        print(Fore.CYAN + f"Processing PDF: {pdf_path}")
        print(Fore.CYAN + "=" * 80)
        
        try:
            # Extract regular text from PDF
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            # Extract potential tables and structured content
            tables = self.pdf_extractor.extract_tables_from_pdf(pdf_path)
            
            # Add table text to pages for combined processing
            for table in tables:
                # Add with special page numbers to distinguish them
                pages.append({
                    "page_num": 1000 + table["page_num"],  # 1000+ indicates table content
                    "text": f"TABLE/STRUCTURED CONTENT FROM PAGE {table['page_num']}:\n{table['table_text']}"
                })
            
            # Create a dictionary to store answers
            answers = {}
            
            # Process each question
            for question in self.questions:
                # Get keywords for this question
                keywords = self.question_keywords.get(question, [])
                
                # Search for pages containing these keywords
                keyword_pages = self.pdf_extractor.search_keywords_in_pdf(pages, keywords)
                
                # Combine all page numbers where keywords were found
                relevant_page_nums = set()
                for page_nums in keyword_pages.values():
                    relevant_page_nums.update(page_nums)
                
                # Always include table content for specific questions
                if any(term in question.lower() for term in ["code", "turnover", "category", "service", "CIN"]):
                    table_page_nums = [p["page_num"] for p in pages if p["page_num"] >= 1000]
                    relevant_page_nums.update(table_page_nums)
                
                # If no relevant pages found, use fallback strategy
                if not relevant_page_nums:
                    print(Fore.YELLOW + f"No relevant pages found for question: {question}")
                    
                    # Include first few pages as fallback for CIN and financial year
                    if "CIN" in question or "financial year" in question.lower():
                        relevant_page_nums = set(range(1, min(10, len(pages) + 1)))
                    else:
                        # For other questions, include all pages with table-like content
                        table_page_nums = [p["page_num"] for p in pages if p["page_num"] >= 1000]
                        if table_page_nums:
                            relevant_page_nums.update(table_page_nums)
                        else:
                            # Last resort: include first few pages
                            relevant_page_nums = set(range(1, min(5, len(pages) + 1)))
                
                # Sort page numbers for readability
                relevant_page_nums = sorted(list(relevant_page_nums))
                
                # Extract text from relevant pages
                context = ""
                for page_num in relevant_page_nums:
                    # Find the right page
                    page_indices = [i for i, p in enumerate(pages) if p["page_num"] == page_num]
                    if page_indices:
                        # Format table content differently
                        if page_num >= 1000:
                            context += f"\n--- TABLE CONTENT (Page {page_num - 1000}) ---\n"
                        else:
                            context += f"\n--- Page {page_num} ---\n"
                        context += pages[page_indices[0]]["text"]
                
                # Truncate context if too long (based on Llama's context window)
                max_context_length = 12000
                if len(context) > max_context_length:
                    print(Fore.YELLOW + f"Context too long ({len(context)} chars), truncating...")
                    context = context[:max_context_length]
                
                # Generate answer with post-processing for consistent formatting
                answer = self.qa_processor.generate_answer(question, context)
                
                # Store answer
                answers[question] = answer
            
            print(Fore.GREEN + f"Completed processing of {pdf_path}")
            return answers
            
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            # Return error message for all questions
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """
        Process multiple PDF files and save results to an Excel file.
        
        Args:
            pdf_dir: Directory containing PDF files
            output_excel: Path to output Excel file
        """
        print(Fore.CYAN + "=" * 80)
        print(Fore.CYAN + f"Processing all PDFs in directory: {pdf_dir}")
        print(Fore.CYAN + "=" * 80)
        
        # Check if directory exists
        if not os.path.isdir(pdf_dir):
            print(Fore.RED + f"Directory not found: {pdf_dir}")
            return
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(Fore.RED + f"No PDF files found in {pdf_dir}")
            return
        
        print(Fore.GREEN + f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF and collect results
        results = []
        
        # Determine number of workers based on system capabilities
        num_workers = min(os.cpu_count() or 4, 4)  # Use at most 4 workers
        print(Fore.BLUE + f"Using {num_workers} parallel workers")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all PDF processing tasks
            future_to_pdf = {
                executor.submit(self.process_pdf, os.path.join(pdf_dir, pdf_file)): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Process results as they complete
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    # Get answers for this PDF
                    answers = future.result()
                    
                    # Add filename to results
                    result = {"PDF Filename": pdf_file}
                    result.update(answers)
                    
                    results.append(result)
                    print(Fore.GREEN + f"Added results for {pdf_file}")
                    
                except Exception as e:
                    print(Fore.RED + f"Error processing {pdf_file}: {e}")
                    # Add error entry
                    result = {"PDF Filename": pdf_file}
                    result.update({question: f"Error: {str(e)}" for question in self.questions})
                    results.append(result)
        
        # Create a DataFrame and save to Excel
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Reorder columns to have PDF Filename first, then questions in specified order
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(Fore.GREEN + f"Results saved to {output_excel}")
        else:
            print(Fore.RED + "No results to save")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PDF Regulatory Information Extraction Pipeline")
    parser.add_argument("--pdf_dir", type=str, default="pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--config", type=str, default="config1.json", help="Configuration file")
    args = parser.parse_args()
    
    try:
        # Print startup message
        print(Fore.CYAN + "=" * 80)
        print(Fore.CYAN + "PDF REGULATORY INFORMATION EXTRACTION PIPELINE")
        print(Fore.CYAN + "=" * 80)
        
        # Create and run the pipeline
        pipeline = PDFRegulatoryExtractor(config_path=args.config)
        pipeline.process_pdfs_batch(args.pdf_dir, args.output)
        
        print(Fore.GREEN + "Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"Pipeline error: {e}", exc_info=True)
        print(Fore.RED + f"Pipeline error: {e
