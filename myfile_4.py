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
            "max_new_tokens": 1024,  # Reduced since we're sending single page
            "return_full_text": False,
            "temperature": 0.1
        }
        self.session = requests.Session()
        self.session.verify = False

    def run(self, prompt: str) -> str:
        """Send prompt to LLM and get response with timeout"""
        start_time = time.time()
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        try:
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
                
            doc.close()
            print(f"PDF text extraction took {time.time() - start_time:.2f} seconds")
            return pages
        except Exception as e:
            print(f"PDF extraction error after {time.time() - start_time:.2f} seconds: {e}")
            raise
    
    def find_question_page(self, pages: List[Dict[str, Any]], question: str) -> int:
        """Find the page that contains the exact question"""
        start_time = time.time()
        
        # Clean the question for better matching
        question_clean = question.strip()
        question_lower = question_clean.lower()
        
        print(f"Searching for question: '{question_clean}'")
        
        # First, try to find exact match
        for page in pages:
            page_text_lower = page["text"].lower()
            if question_lower in page_text_lower:
                print(f"Found exact question match on page {page['page_num']}")
                print(f"Question search took {time.time() - start_time:.2f} seconds")
                return page["page_num"]
        
        # If no exact match, try partial matching with key terms
        key_terms = self._extract_key_terms(question)
        best_page = None
        max_matches = 0
        
        for page in pages:
            page_text_lower = page["text"].lower()
            match_count = sum(1 for term in key_terms if term.lower() in page_text_lower)
            
            if match_count > max_matches:
                max_matches = match_count
                best_page = page["page_num"]
        
        if best_page:
            print(f"Found best matching page {best_page} with {max_matches} key term matches")
        else:
            print("No matching page found, using page 1 as fallback")
            best_page = 1
            
        print(f"Question search took {time.time() - start_time:.2f} seconds")
        return best_page
    
    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from question for partial matching"""
        # Remove common words and extract meaningful terms
        common_words = {'of', 'the', 'to', 'which', 'in', 'or', 'and', 'a', 'an', 'is', 'are', 'with'}
        
        # Split question into words and filter
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [word for word in words if word not in common_words and len(word) > 2]
        
        # Add the full question as a key term too
        key_terms.append(question.lower())
        
        print(f"Key terms extracted: {key_terms}")
        return key_terms

class QuestionAnswerProcessor:
    """Class for processing PDF content to answer regulatory information questions"""
    
    def __init__(self, llm_generator: LLMGenerator):
        self.llm_generator = llm_generator
    
    def generate_answer(self, question: str, page_text: str, page_num: int) -> str:
        """Generate an answer to a question using single page context"""
        start_time = time.time()
        
        # Create a simplified prompt for single page processing
        prompt = self._create_simple_qa_prompt(question, page_text, page_num)
        
        # Generate the answer
        raw_answer = self.llm_generator.run(prompt)
        
        # Post-process the answer
        processed_answer = self._post_process_answer(question, raw_answer)
        
        print(f"Answer generation took {time.time() - start_time:.2f} seconds")
        return processed_answer
    
    def _create_simple_qa_prompt(self, question: str, page_text: str, page_num: int) -> str:
        """Create a simple, focused prompt for single page processing"""
        
        sys_message = "You are an expert at extracting specific information from financial documents."
        
        # Create a focused prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}

You will find a question and the content from a single page of a PDF document. The question likely appears somewhere in this page content.

Your task:
1. Look for the exact question in the page content
2. Find the answer that appears near or after the question
3. Extract ONLY the direct answer value
4. If you cannot find the question or answer, respond with "Not found"

Important:
- For numbers, provide only the numeric value (no currency symbols, no units)
- For codes, provide only the code
- For descriptions, provide the exact text
- Do not add explanations or extra text

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Page {page_num} Content:
{page_text}

Find the question in the content above and extract the answer that follows it.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def _post_process_answer(self, question: str, answer: str) -> str:
        """Simple post-processing for clean answers"""
        # Remove common prefixes
        cleaned = re.sub(r'^(Answer:|The answer is:|Based on|According to)', '', answer, flags=re.IGNORECASE).strip()
        
        # Handle "not found" responses
        if re.search(r'(not found|not available|cannot find|could not find)', cleaned, re.IGNORECASE):
            return "Information not found in the provided context"
        
        # For turnover questions, extract just the number
        if "turnover" in question.lower() and "rupees" in question.lower():
            # Look for numbers (with or without separators)
            number_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)', cleaned)
            if number_match:
                # Remove commas and return clean number
                return number_match.group(1).replace(',', '')
        
        # For CIN, extract the code
        if "CIN" in question or "corporate identity" in question.lower():
            cin_match = re.search(r'([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})', cleaned)
            if cin_match:
                return cin_match.group(1)
        
        # For codes, extract digits
        if "code" in question.lower():
            if "4 digit" in question.lower():
                code_match = re.search(r'(\d{4})', cleaned)
                if code_match:
                    return code_match.group(1)
            elif "8 digit" in question.lower():
                code_match = re.search(r'(\d{8})', cleaned)
                if code_match:
                    return code_match.group(1)
        
        # For financial year
        if "financial year" in question.lower():
            fy_match = re.search(r'(\d{4}[-/]\d{2,4})', cleaned)
            if fy_match:
                return fy_match.group(1)
        
        return cleaned.strip()

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self, config_path: str = "config1.json"):
        self.config_path = config_path
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
        # Exact questions as they appear in PDFs
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
        """Process a PDF file using single page per question approach"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            print(f"Extracted {len(pages)} pages from PDF")
            
            answers = {}
            
            # Process each question
            for question in self.questions:
                question_start = time.time()
                print(f"\nProcessing question: {question}")
                
                # Find the page containing this question
                target_page_num = self.pdf_extractor.find_question_page(pages, question)
                
                # Get the page content
                target_page = None
                for page in pages:
                    if page["page_num"] == target_page_num:
                        target_page = page
                        break
                
                if target_page:
                    print(f"Using page {target_page_num} for processing")
                    print(f"Page content preview: {target_page['text'][:200]}...")
                    
                    # Generate answer using single page
                    answer = self.qa_processor.generate_answer(
                        question, 
                        target_page["text"], 
                        target_page_num
                    )
                    
                    print(f"Generated answer: {answer}")
                else:
                    print(f"Could not find page for question")
                    answer = "Information not found in the provided context"
                
                answers[question] = answer
                print(f"Question processed in {time.time() - question_start:.2f} seconds")
            
            print(f"\nCompleted processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """Process multiple PDF files and save results to an Excel file"""
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
                pdf_start_time = time.time()
                print(f"\n{'='*50}")
                print(f"Processing: {pdf_file}")
                print(f"{'='*50}")
                
                answers = self.process_pdf(pdf_path)
                
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                
                results.append(result)
                print(f"\nCompleted {pdf_file} in {time.time() - pdf_start_time:.2f} seconds")
                
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
            
            # Print summary
            print(f"\nSUMMARY:")
            print(f"Total PDFs processed: {len(results)}")
            for question in self.questions:
                found_count = sum(1 for result in results if result[question] != "Information not found in the provided context" and not result[question].startswith("Error:"))
                print(f"'{question[:50]}...': {found_count}/{len(results)} found")
        else:
            print("No results to save")
        
        print(f"\nTotal batch processing completed in {time.time() - batch_start_time:.2f} seconds")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simplified PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--config", type=str, default="config1.json", help="Configuration file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    args = parser.parse_args()
    
    try:
        print("Starting Simplified PDF Regulatory Information Extraction")
        print("Strategy: Use complete question as keyword, process single page per question")
        start_time = time.time()
        
        pipeline = PDFRegulatoryExtractor(config_path=args.config)
        
        if args.single_pdf:
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
                
            results = []
            pdf_file = os.path.basename(args.single_pdf)
            
            try:
                answers = pipeline.process_pdf(args.single_pdf)
                
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                results.append(result)
                
                df = pd.DataFrame(results)
                columns = ["PDF Filename"] + pipeline.questions
                df = df[columns]
                df.to_excel(args.output, index=False)
                print(f"Results saved to {args.output}")
                
            except Exception as e:
                print(f"Error processing {args.single_pdf}: {e}")
        else:
            pipeline.process_pdfs_batch(args.pdf_dir, args.output)
        
        print(f"\nProcessing completed successfully in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
