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
        """Create a prompt for the Llama model with few-shot examples"""
        # Format the prompt for Llama
        sys_message = "You are an expert in extracting business regulatory information from financial documents with high precision and accuracy."
        
        # Create specific instructions based on question type
        specific_instructions = ""
        few_shot_examples = ""
        
        # Add question-specific instructions and examples
        if "CIN" in question or "corporate identity" in question.lower():
            specific_instructions = """
- For Corporate Identity Number (CIN), look for a code that typically starts with 'L' or 'U' followed by numbers and letters
- A CIN is a 21-digit alphanumeric code (e.g., L12345MH2010PLC123456)
"""
            few_shot_examples = """
Example:
Context: "...The Corporate Identity Number (CIN) of the Company is L17110MH1973PLC019786..."
Question: Corporate identity number (CIN) of company
Answer: L17110MH1973PLC019786
"""
        elif "name of the company" in question.lower():
            specific_instructions = """
- Look for the company name which is usually mentioned in the header, title, or early sections of the document
- Look for phrases like "Annual Report of", "Company Name:", or in the company letterhead
- The company name may include suffixes like Ltd., Limited, Pvt. Ltd., Private Limited, etc.
"""
            few_shot_examples = """
Example:
Context: "...ABC Industries Limited Annual Report 2023..."
Question: Name of the Company
Answer: ABC Industries Limited
"""
        elif "financial year" in question.lower():
            specific_instructions = """
- Look for phrases like "for the year ended", "financial year", "FY", followed by dates
- Express the financial year in DD/MM/YYYY - DD/MM/YYYY format (e.g., "01/04/2023 - 31/03/2024")
- If only year format is found (like 2023-24), convert it to the date format assuming April 1st to March 31st
"""
            few_shot_examples = """
Example:
Context: "...Annual Report for the Financial Year 2022-23..."
Question: Financial year to which financial statements relates
Answer: 01/04/2022 - 31/03/2023
"""
        elif "4 digit code" in question:
            specific_instructions = """
- Look for 4-digit codes associated with product or service categories
- The code might be labeled as ITC code, NPCS code, HS code, or NIC code
"""
            few_shot_examples = """
Example:
Context: "...The company primarily operates in the business of textile products with ITC code 5205..."
Question: Product or service category code (ITC/ NPCS 4 digit code)
Answer: 5205
"""
        elif "8 digit code" in question:
            specific_instructions = """
- Look for 8-digit codes associated with specific products or services
- The code might be labeled as ITC code, NPCS code, HS code, or product code
"""
        elif "turnover" in question.lower() and "category" in question.lower():
            specific_instructions = """
- Look for financial figures associated with revenue, turnover, or sales of the product/service category
- Extract the exact amount including the unit (e.g., "Rs. 10,000,000" or "₹ 10 crores")
"""     
            few_shot_examples = """
Example:
Context: "Turnover of highest contributing product or service (in Rupees)"
Question: Turnover of highest contributing product or service (in Rupees)
Answer: 309223000
"""
        elif "turnover" in question.lower() and "highest" in question.lower():
            specific_instructions = """
- Look for financial figures associated with the highest revenue, turnover, or sales product/service
- Extract the exact numerical amount in Rupees (e.g., "309223000")
- Look for tables, financial statements, or segment reporting sections
- The amount may be presented as plain numbers, with commas, or with currency symbols
"""
            few_shot_examples = """
Example:
Context: "...Turnover of highest contributing product or service: Rs. 30,92,23,000..."
Question: Turnover of highest contributing product or service (in Rupees)
Answer: 309223000
"""
        elif "description" in question.lower():
            specific_instructions = """
- Extract the exact description as stated in the document
- Look in business segment reporting, notes to accounts, or product sections
"""
        
        # Create simplified prompt for better performance
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}

You will be provided with a question and context from a PDF document. Your task is to extract the precise answer to the question from the context.

{specific_instructions}
Follow these general guidelines:
- If the answer is explicitly stated in the text, provide that exact answer
- If the answer is not in the context, respond with "-"
- Provide only the direct answer without explanations

{few_shot_examples}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Context from PDF:
{context}

Answer the question based only on the information in the context. If the information is not available, return "-".
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def _post_process_answer(self, question: str, answer: str) -> str:
        """Post-process the answer to format it consistently"""
        # Remove any explanatory text or prefixes
        cleaned = re.sub(r'^(Answer:|The answer is:|Based on the context,)', '', answer).strip()
        
        # Handle "not found" responses consistently - return "-"
        if re.search(r'(not found|not provided|not mentioned|couldn\'t find|could not find|no information|not available|not stated)', 
                     cleaned, re.IGNORECASE):
            return "-"
        
        # If the cleaned answer is empty or just whitespace, return "-"
        if not cleaned or cleaned.isspace():
            return "-"
        
        # Format CIN consistently
        if "CIN" in question or "corporate identity number" in question.lower():
            # Try to extract CIN format (L12345MH2010PLC123456)
            cin_match = re.search(r'[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}', cleaned)
            if cin_match:
                return cin_match.group(0)
        
        # Format company name consistently
        if "name of the company" in question.lower():
            # Clean up company name - remove extra whitespace and common prefixes
            company_name = re.sub(r'^(Company Name|Name|Company)[:.\s]*', '', cleaned, flags=re.IGNORECASE).strip()
            if company_name:
                return company_name
            return "-"
        
        # Format financial year consistently
        if "financial year" in question.lower():
            # Look for common financial year patterns
            fy_match = re.search(r'(FY|F\.Y\.|Financial Year)[ :]?([0-9]{4}[-/ ][0-9]{2,4})', cleaned, re.IGNORECASE)
            if fy_match:
                year_part = fy_match.group(2)
                return self._convert_to_date_format(year_part)
            
            # Try another pattern
            fy_match2 = re.search(r'([0-9]{4}[-/ ][0-9]{2,4})', cleaned)
            if fy_match2:
                year_part = fy_match2.group(1)
                return self._convert_to_date_format(year_part)
            
            # Try to find year ended pattern
            year_ended_match = re.search(r'year ended[:\s]+([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{4})', cleaned, re.IGNORECASE)
            if year_ended_match:
                end_date = year_ended_match.group(1)
                # Convert to DD/MM/YYYY format and calculate start date
                return self._format_year_ended_to_date_range(end_date)
        
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
    
    def _convert_to_date_format(self, year_string: str) -> str:
        """Convert year format like '2023-24' to date format '01/04/2023 - 31/03/2024'"""
        try:
            # Handle different separators
            year_string = year_string.replace('/', '-').replace(' ', '-')
            
            if '-' in year_string:
                parts = year_string.split('-')
                start_year = int(parts[0])
                
                # Handle 2-digit or 4-digit end year
                if len(parts[1]) == 2:
                    end_year = int(f"{start_year // 100}{parts[1]:0>2}")
                else:
                    end_year = int(parts[1])
                
                # Financial year typically runs April 1 to March 31
                start_date = f"01/04/{start_year}"
                end_date = f"31/03/{end_year}"
                
                return f"{start_date} - {end_date}"
            else:
                # Single year - assume current financial year
                year = int(year_string)
                start_date = f"01/04/{year}"
                end_date = f"31/03/{year + 1}"
                return f"{start_date} - {end_date}"
                
        except (ValueError, IndexError):
            return year_string  # Return original if conversion fails
    
    def _format_year_ended_to_date_range(self, end_date_str: str) -> str:
        """Convert 'year ended 31/03/2024' to '01/04/2023 - 31/03/2024' format"""
        try:
            # Parse the end date
            import datetime
            
            # Handle different date formats
            for fmt in ['%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']:
                try:
                    end_date = datetime.datetime.strptime(end_date_str.replace('-', '/'), fmt)
                    break
                except ValueError:
                    continue
            else:
                return end_date_str  # Return original if parsing fails
            
            # Calculate start date (typically one year before)
            start_date = end_date.replace(year=end_date.year - 1)
            
            # If end date is March 31, start date should be April 1
            if end_date.month == 3 and end_date.day == 31:
                start_date = start_date.replace(month=4, day=1)
            else:
                # Add one day to end date to get start date
                start_date = end_date.replace(year=end_date.year - 1) + datetime.timedelta(days=1)
            
            # Format both dates
            start_formatted = start_date.strftime('%d/%m/%Y')
            end_formatted = end_date.strftime('%d/%m/%Y')
            
            return f"{start_formatted} - {end_formatted}"
            
        except (ValueError, AttributeError):
            return end_date_str  # Return original if conversion fails

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self, config_path: str = "config1.json"):
        self.config_path = config_path
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
        # Define the specific regulatory questions to extract (with Company Name as 3rd)
        self.questions = [
            "Corporate identity number (CIN) of company",
            "Financial year to which financial statements relates",
            "Name of the Company",
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
            "Name of the Company": ["company name", "annual report", "limited", "ltd", "private limited", "pvt ltd"],
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
                
                # Always use only pages 1 and 10 for all questions
                if not relevant_page_nums:
                    print(f"  No relevant pages found, using fallback strategy (pages 1 and 10)")
                    # Use only pages 1 and 10 for all questions
                    relevant_page_nums = []
                    if len(pages) >= 1:
                        relevant_page_nums.append(1)
                    if len(pages) >= 10:
                        relevant_page_nums.append(10)
                else:
                    # Even if relevant pages found, limit to only pages 1 and 10
                    print(f"  Limiting to pages 1 and 10 only")
                    limited_pages = []
                    if 1 in relevant_page_nums and len(pages) >= 1:
                        limited_pages.append(1)
                    if 10 in relevant_page_nums and len(pages) >= 10:
                        limited_pages.append(10)
                    # If neither page 1 nor 10 were in relevant pages, still use them as fallback
                    if not limited_pages:
                        if len(pages) >= 1:
                            limited_pages.append(1)
                        if len(pages) >= 10:
                            limited_pages.append(10)
                    relevant_page_nums = limited_pages
                
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
                
                # Store answer (ensure it's not empty, return "-" if so)
                answers[question] = answer if answer and answer.strip() else "-"
                
                # Log total time for this question
                print(f"Question processed in {time.time() - question_start:.2f} seconds")
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return "-" for all questions when there's an error
            return {question: "-" for question in self.questions}
    
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
                # Add error entry with "-" for all questions
                result = {"PDF Filename": pdf_file}
                result.update({question: "-" for question in self.questions})
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
