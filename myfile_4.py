import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
import requests
import warnings
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress SSL warnings
warnings.filterwarnings('ignore')

# Define the Llama URL
LLAMA_URL = "https://ue1-llm.crisil.local/llama3_3/70b/llm/"

class LLMGenerator:
    """Component to generate responses from hosted LLM model"""
    def __init__(self, url: str = LLAMA_URL):
        self.url = url
        self.generation_kwargs = {
            "max_new_tokens": 2048,  # Reduced from 5048
            "return_full_text": False,
            "temperature": 0.1
        }
        # Use a session for connection pooling
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification

    def run(self, prompt: str) -> str:
        """Send prompt to LLM and get response with timeout"""
        start_time = time.time()
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        try:
            # Add timeout parameter to prevent hanging
            response = self.session.post(self.url, json=body, timeout=30)
            
            if response.status_code != 200:
                print(f"API Error: Status {response.status_code}")
                return f"Error: LLM API error: Status {response.status_code}"
            
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                result = response_json[0].get('generated_text', '')
                print(f"LLM API call took {time.time() - start_time:.2f} seconds")
                return result
            else:
                print(f"Unexpected API response format: {response_json}")
                return "Error: Unexpected response format from LLM API"
                
        except requests.exceptions.Timeout:
            print(f"API call timed out after {time.time() - start_time:.2f} seconds")
            return "Error: Request to LLM API timed out"
        except Exception as e:
            print(f"API call failed after {time.time() - start_time:.2f} seconds: {e}")
            return f"Error: {str(e)}"

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
    
    def search_keywords_in_pdf(self, pages: List[Dict[str, Any]], keywords: List[str], question: str) -> List[int]:
        """Search for keywords with optimized pattern matching"""
        start_time = time.time()
        relevant_pages = []
        
        # First try to find pages with exact question mentions
        question_keywords = [
            # The complete question
            question,
            # Main subject of the question (remove filler words)
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
        # Skip complex regex patterns for better performance
        for keyword in keywords:
            for page in pages:
                if keyword.lower() in page["text"].lower():
                    if page["page_num"] not in relevant_pages:
                        relevant_pages.append(page["page_num"])
        
        print(f"Keyword search took {time.time() - start_time:.2f} seconds (found {len(relevant_pages)} pages)")
        return relevant_pages

class QuestionAnswerProcessor:
    """Class for processing PDF content to answer regulatory information questions"""
    
    def __init__(self, llm_generator: LLMGenerator):
        self.llm_generator = llm_generator
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer to a question using the provided context"""
        start_time = time.time()
        
        # Create a prompt for Llama
        prompt = self._create_qa_prompt(question, context)
        
        # Generate the answer
        raw_answer = self.llm_generator.run(prompt)
        
        # Post-process the answer for consistent formatting
        processed_answer = self._post_process_answer(question, raw_answer)
        
        print(f"Answer generation for '{question[:30]}...' took {time.time() - start_time:.2f} seconds")
        return processed_answer
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the Llama model with comprehensive few-shot examples"""
        # Format the prompt for Llama with exactly 9 few-shot examples
        sys_message = "You are an expert in extracting business regulatory information from financial documents with high precision and accuracy."
        
        # Comprehensive few-shot examples for all 9 questions
        few_shot_examples = """
Examples:

Example 1:
Context: "The Corporate Identity Number (CIN) of the Company is L17110MH1973PLC019786. This annual report covers the financial year 2022-23."
Question: Corporate identity number (CIN) of company
Answer: L17110MH1973PLC019786

Example 2:
Context: "Annual Report for the Financial Year 2022-23. The board of directors present the annual report for the year ended March 31, 2023."
Question: Financial year to which financial statements relates
Answer: 2022-23

Example 3:
Context: "TATA CONSULTANCY SERVICES LIMITED Annual Report 2022-23. TCS is India's leading software services company."
Question: Name of the company
Answer: TATA CONSULTANCY SERVICES LIMITED

Example 4:
Context: "The company operates in information technology services with ITC code 6202. This represents computer programming activities."
Question: Product or service category code (ITC/ NPCS 4 digit code)
Answer: 6202

Example 5:
Context: "Primary business segment: Information Technology Services and Digital Solutions. The company provides end-to-end IT services."
Question: Description of the product or service category
Answer: Information Technology Services and Digital Solutions

Example 6:
Context: "Revenue from IT Services segment: Rs. 1,85,000 crores. This represents the turnover from our primary service category."
Question: Turnover of the product or service category (in Rupees)
Answer: 185000000000

Example 7:
Context: "Highest contributing service with detailed code 62020001 representing custom software development services."
Question: Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
Answer: 62020001

Example 8:
Context: "Primary service offering: Custom Software Development and Maintenance Services for enterprise clients globally."
Question: Description of the product or service
Answer: Custom Software Development and Maintenance Services

Example 9:
Context: "Revenue from highest contributing service: Rs. 95,500 crores from software development and maintenance services."
Question: Turnover of highest contributing product or service (in Rupees)
Answer: 95500000000
"""
        
        instructions = """
Extract the precise answer to the question from the context provided.

Guidelines:
- If the answer is explicitly stated in the text, provide that exact answer
- For CIN: Extract the complete 21-character alphanumeric code starting with 'L' or 'U'
- For company name: Provide the full official name including suffixes like 'Limited', 'Ltd.', etc.
- For financial year: Use format like '2022-23' or as stated in document
- For codes: Extract only the numeric digits (4-digit or 8-digit as specified)
- For turnover amounts: Convert to pure numbers (remove currency symbols, commas, words like 'crores')
- For descriptions: Provide the exact text as mentioned in the document
- If no relevant information is found, respond with "-"
- Provide only the direct answer without explanations or prefixes
"""
        
        # Create the complete prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}

{instructions}

{few_shot_examples}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Context from PDF (Pages 1 and 10):
{context}

Extract the answer based only on the information provided in the context.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def _post_process_answer(self, question: str, answer: str) -> str:
        """Post-process the answer to format it consistently"""
        # Remove any explanatory text or prefixes
        cleaned = re.sub(r'^(Answer:|The answer is:|Based on the context,)', '', answer).strip()
        
        # Handle "not found" responses consistently - return "-"
        if re.search(r'(not found|not provided|not mentioned|couldn\'t find|could not find|no information|not available|not specified|not given)', 
                     cleaned, re.IGNORECASE) or not cleaned:
            return "-"
        
        # Format company name consistently
        if "name of the company" in question.lower() or "company name" in question.lower():
            # Clean up common prefixes and suffixes that might be added by the model
            cleaned = re.sub(r'^(The company name is|Company name:|Name:)', '', cleaned, flags=re.IGNORECASE).strip()
            # Remove quotes if present
            cleaned = cleaned.strip('"\'')
            return cleaned
        
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
        
        # Format financial figures consistently - enhanced for highest turnover
        if "turnover" in question.lower() or "in Rupees" in question:
            # For the highest turnover question, try to extract just the number
            if "highest" in question.lower():
                # Look for pure numbers first (most common format)
                number_match = re.search(r'\b(\d{1,3}(?:,\d{3})*|\d+)\b', cleaned)
                if number_match:
                    # Remove commas and return just the number
                    return number_match.group(1).replace(',', '')
            
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
        
        # Return the cleaned answer for other questions, or "-" if empty
        return cleaned if cleaned else "-"

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self, config_path: str = "config1.json"):
        self.config_path = config_path
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
        # Define the specific regulatory questions to extract - with company name as 3rd
        self.questions = [
            "Corporate identity number (CIN) of company",
            "Financial year to which financial statements relates",
            "Name of the company",  # Added as 3rd column
            "Product or service category code (ITC/ NPCS 4 digit code)",
            "Description of the product or service category",
            "Turnover of the product or service category (in Rupees)",
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
            "Description of the product or service",
            "Turnover of highest contributing product or service (in Rupees)"
        ]
        
        # Define keywords for each question - enhanced for the last question and added company name
        self.question_keywords = {
            "Corporate identity number (CIN) of company": ["CIN", "corporate identity", "L", "U", "registration"],
            "Financial year to which financial statements relates": ["financial year", "FY", "year ended", "period ended"],
            "Name of the company": ["company", "limited", "ltd", "private limited", "pvt ltd", "corporation", "enterprises", "annual report"],  # Added keywords for company name
            "Product or service category code (ITC/ NPCS 4 digit code)": ["ITC", "NPCS", "product code", "4 digit"],
            "Description of the product or service category": ["product category", "service category", "business segment"],
            "Turnover of the product or service category (in Rupees)": ["category turnover", "segment turnover", "revenue"],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": ["8 digit", "highest turnover", "main product"],
            "Description of the product or service": ["product description", "service description", "main offering"],
            "Turnover of highest contributing product or service (in Rupees)": ["highest turnover", "main product turnover", "primary product revenue", "segment revenue", "business segment", "revenue from operations", "sale of products", "product sales"]
        }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a PDF file with performance optimization"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF - this is where bottlenecks often occur
            extract_start = time.time()
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            print(f"PDF text extraction completed in {time.time() - extract_start:.2f} seconds")
            
            # Create a dictionary to store answers
            answers = {}
            
            # Process each question
            for question in self.questions:
                question_start = time.time()
                print(f"Processing question: {question[:30]}...")
                
                # Get keywords for this question
                keywords = self.question_keywords.get(question, [])
                
                # Search for pages containing these keywords - simplified for performance
                search_start = time.time()
                relevant_page_nums = self.pdf_extractor.search_keywords_in_pdf(pages, keywords, question)
                print(f"  Page search took {time.time() - search_start:.2f} seconds")
                
                # If no relevant pages found, use pages 1 and 10 only
                if not relevant_page_nums:
                    print(f"  No relevant pages found, using pages 1 and 10")
                    # Use only page 1 and page 10 for context
                    relevant_page_nums = [1]
                    if len(pages) >= 10:
                        relevant_page_nums.append(10)
                else:
                    # Even if relevant pages found, prioritize pages 1 and 10
                    relevant_page_nums = [1]
                    if len(pages) >= 10:
                        relevant_page_nums.append(10)
                
                # Sort page numbers for readability
                relevant_page_nums = sorted(list(set(relevant_page_nums)))
                
                # Extract text from relevant pages
                context_start = time.time()
                context = ""
                for page_num in relevant_page_nums:
                    # Find the right page
                    page_indices = [i for i, p in enumerate(pages) if p["page_num"] == page_num]
                    if page_indices:
                        context += f"\n--- Page {page_num} ---\n"
                        context += pages[page_indices[0]]["text"]
                
                # Truncate context to reduce LLM processing time
                max_context_length = 4000  # Drastically reduced from 12000
                if len(context) > max_context_length:
                    context = context[:max_context_length]
                print(f"  Context preparation took {time.time() - context_start:.2f} seconds")
                
                # Generate answer - this is where most time is spent
                llm_start = time.time()
                answer = self.qa_processor.generate_answer(question, context)
                print(f"  LLM processing took {time.time() - llm_start:.2f} seconds")
                
                # Store answer
                answers[question] = answer
                
                # Log total time for this question
                print(f"Question processed in {time.time() - question_start:.2f} seconds")
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return error message for all questions
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
        
        # Use serial processing for better bottleneck identification
        # For production, you can re-enable parallel processing if needed
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
            
            # Reorder columns to have PDF Filename first, then questions in order
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
    parser = argparse.ArgumentParser(description="Optimized PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--config", type=str, default="config1.json", help="Configuration file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    args = parser.parse_args()
    
    try:
        print("Starting PDF Regulatory Information Extraction")
        start_time = time.time()
        
        # Create pipeline
        pipeline = PDFRegulatoryExtractor(config_path=args.config)
        
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
