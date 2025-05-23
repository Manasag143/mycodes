import os
import re
import time
import json
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
    
    def generate_answers(self, questions: List[str], context: str) -> Dict[str, str]:
        """Generate answers to multiple questions using the provided context and return as JSON"""
        start_time = time.time()
        
        # Create a prompt for Llama that returns JSON
        prompt = self._create_json_qa_prompt(questions, context)
        
        # Generate the answers
        raw_response = self.llm_generator.run(prompt)
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                answers = json.loads(json_str)
            else:
                # Fallback: create empty answers dict
                answers = {question: "-" for question in questions}
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {raw_response}")
            # Fallback: create empty answers dict
            answers = {question: "-" for question in questions}
        
        # Ensure all questions have answers
        for question in questions:
            if question not in answers:
                answers[question] = "-"
        
        print(f"Answer generation took {time.time() - start_time:.2f} seconds")
        return answers
    
    def _create_json_qa_prompt(self, questions: List[str], context: str) -> str:
        """Create a prompt for the Llama model that returns JSON format"""
        # Format the prompt for Llama
        sys_message = "You are an expert in extracting business regulatory information from financial documents with high precision and accuracy."
        
        # Create the questions list for JSON output
        questions_json = {f"question_{i+1}": question for i, question in enumerate(questions)}
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}

Instructions:
1. Extract information from the context delimited by ####.
2. If value not present return "-".
3. Return your response as a valid JSON object with the question as key and extracted answer as value.
4. For Corporate Identity Number (CIN), look for alphanumeric codes starting with 'L' or 'U'.
5. For financial year, look for phrases like "for the year ended", "financial year", "FY".
6. For company name, look for the official registered name of the company.
7. For product codes, look for 4-digit or 8-digit numerical codes.
8. For turnover amounts, extract numerical values in Rupees.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Questions to answer:
{json.dumps(questions_json, indent=2)}

####
{context}
####

Return a JSON object with the following structure:
{{
    "Name of the Company": "extracted_value_or_-",
    "Corporate identity number (CIN) of company": "extracted_value_or_-",
    "Financial year to which financial statements relates": "extracted_value_or_-",
    "Product or service category code (ITC/ NPCS 4 digit code)": "extracted_value_or_-",
    "Description of the product or service category": "extracted_value_or_-",
    "Turnover of the product or service category (in Rupees)": "extracted_value_or_-",
    "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": "extracted_value_or_-",
    "Description of the product or service": "extracted_value_or_-",
    "Turnover of highest contributing product or service (in Rupees)": "extracted_value_or_-"
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self, config_path: str = "config1.json"):
        self.config_path = config_path
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
        # Define the specific regulatory questions to extract (with Name of Company as 3rd column)
        self.questions = [
            "Name of the Company",
            "Corporate identity number (CIN) of company",
            "Financial year to which financial statements relates",
            "Product or service category code (ITC/ NPCS 4 digit code)",
            "Description of the product or service category",
            "Turnover of the product or service category (in Rupees)",
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
            "Description of the product or service",
            "Turnover of highest contributing product or service (in Rupees)"
        ]
        
        # Define keywords for each question
        self.question_keywords = {
            "Name of the Company": ["company name", "name of company", "registered name", "company", "limited", "ltd", "private limited"],
            "Corporate identity number (CIN) of company": ["CIN", "corporate identity", "L", "U", "registration"],
            "Financial year to which financial statements relates": ["financial year", "FY", "year ended", "period ended"],
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
            # Extract text from PDF
            extract_start = time.time()
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            print(f"PDF text extraction completed in {time.time() - extract_start:.2f} seconds")
            
            # Get all keywords for searching
            all_keywords = []
            for keywords in self.question_keywords.values():
                all_keywords.extend(keywords)
            
            # Search for relevant pages
            search_start = time.time()
            relevant_page_nums = []
            for question in self.questions:
                keywords = self.question_keywords.get(question, [])
                page_nums = self.pdf_extractor.search_keywords_in_pdf(pages, keywords, question)
                relevant_page_nums.extend(page_nums)
            
            # Remove duplicates and sort
            relevant_page_nums = sorted(list(set(relevant_page_nums)))
            
            # If no relevant pages found, use first few pages
            if not relevant_page_nums:
                relevant_page_nums = list(range(1, min(6, len(pages) + 1)))
            
            print(f"Page search took {time.time() - search_start:.2f} seconds")
            
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
            max_context_length = 6000
            if len(context) > max_context_length:
                context = context[:max_context_length]
            print(f"Context preparation took {time.time() - context_start:.2f} seconds")
            
            # Generate answers for all questions at once
            llm_start = time.time()
            answers = self.qa_processor.generate_answers(self.questions, context)
            print(f"LLM processing took {time.time() - llm_start:.2f} seconds")
            
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
            
            # Reorder columns to have PDF Filename first, then questions in specified order
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            
            # Also save as JSON for debugging
            json_output = output_excel.replace('.xlsx', '.json')
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"JSON results also saved to {json_output}")
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
                
                # Also save as JSON
                json_output = args.output.replace('.xlsx', '.json')
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"JSON results also saved to {json_output}")
                
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
