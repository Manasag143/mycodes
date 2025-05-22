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
            "max_new_tokens": 2048,
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
    
    def search_keywords_in_pdf(self, pages: List[Dict[str, Any]], keywords: List[str], question: str) -> List[int]:
        """Enhanced keyword search with better fallback strategy"""
        start_time = time.time()
        relevant_pages = []
        
        # Create expanded keyword search based on question type
        expanded_keywords = self._expand_keywords(keywords, question)
        
        # Search for keywords across all pages
        for page in pages:
            page_text_lower = page["text"].lower()
            
            # Check if any keyword appears in this page
            for keyword in expanded_keywords:
                if keyword.lower() in page_text_lower:
                    if page["page_num"] not in relevant_pages:
                        relevant_pages.append(page["page_num"])
                        break  # Found match, move to next page
        
        # If no pages found, use intelligent fallback
        if not relevant_pages:
            relevant_pages = self._intelligent_fallback(pages, question)
        
        print(f"Keyword search took {time.time() - start_time:.2f} seconds (found {len(relevant_pages)} pages)")
        return sorted(relevant_pages)
    
    def _expand_keywords(self, keywords: List[str], question: str) -> List[str]:
        """Expand keywords based on question context"""
        expanded = list(keywords)
        
        # Add variations for turnover questions
        if "turnover" in question.lower():
            expanded.extend([
                "revenue", "sales", "income", "receipts", "earnings",
                "₹", "rs.", "rupees", "lakhs", "crores", "millions",
                "total income", "gross revenue", "net sales"
            ])
        
        # Add variations for highest contributing product
        if "highest" in question.lower() and "contributing" in question.lower():
            expanded.extend([
                "major", "main", "primary", "principal", "largest",
                "biggest", "most", "maximum", "top", "leading"
            ])
        
        # Add variations for description questions
        if "description" in question.lower():
            expanded.extend([
                "business", "activity", "operation", "segment",
                "division", "manufacturing", "trading", "services"
            ])
        
        return expanded
    
    def _intelligent_fallback(self, pages: List[Dict[str, Any]], question: str) -> List[int]:
        """Intelligent fallback strategy based on question type"""
        fallback_pages = []
        
        if "turnover" in question.lower() and "highest" in question.lower():
            # For highest turnover questions, look in financial sections
            # Check pages that contain financial terms
            financial_terms = [
                "profit", "loss", "revenue", "income", "statement",
                "financial", "balance sheet", "segment", "note"
            ]
            
            for page in pages:
                page_text_lower = page["text"].lower()
                if any(term in page_text_lower for term in financial_terms):
                    fallback_pages.append(page["page_num"])
            
            # If still no matches, use pages 1-10 (cover page + initial sections)
            if not fallback_pages:
                fallback_pages = list(range(1, min(11, len(pages) + 1)))
        
        else:
            # For other questions, use first 5 pages
            fallback_pages = list(range(1, min(6, len(pages) + 1)))
        
        return fallback_pages

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
        """Create an enhanced prompt for the Llama model"""
        sys_message = "You are an expert in extracting business regulatory information from financial documents with high precision and accuracy."
        
        # Enhanced specific instructions
        specific_instructions = ""
        few_shot_examples = ""
        
        if "turnover" in question.lower() and "highest" in question.lower():
            specific_instructions = """
- Look for financial figures associated with the highest revenue, turnover, or sales product/service
- Search for terms like "major revenue", "highest sales", "primary income", "main product revenue"
- Look in segment reporting, revenue breakdown, or notes to financial statements
- Extract the exact amount in Rupees (may be in lakhs, crores, or absolute numbers)
- If multiple products/services are mentioned, identify the one with highest contribution
"""
            few_shot_examples = """
Example 1:
Context: "The company's main business segment generated revenue of Rs. 309,223,000 during the year..."
Question: Turnover of highest contributing product or service (in Rupees)
Answer: 309223000

Example 2:
Context: "Revenue from operations: Product A - Rs. 150 crores, Product B - Rs. 200 crores..."
Question: Turnover of highest contributing product or service (in Rupees)
Answer: 20000000000

Example 3:
Context: "Segment wise revenue: Textiles Rs. 45.2 crores, Chemicals Rs. 67.8 crores..."
Question: Turnover of highest contributing product or service (in Rupees)
Answer: 6780000000
"""
        
        elif "CIN" in question or "corporate identity" in question.lower():
            specific_instructions = """
- For Corporate Identity Number (CIN), look for a code that typically starts with 'L' or 'U' followed by numbers and letters
- A CIN is a 21-digit alphanumeric code (e.g., L12345MH2010PLC123456)
"""
            few_shot_examples = """
Example 1:
Context: "...The Corporate Identity Number (CIN) of the Company is L17110MH1973PLC019786..."
Question: Corporate identity number (CIN) of company
Answer: L17110MH1973PLC019786
"""
        
        elif "financial year" in question.lower():
            specific_instructions = """
- Look for phrases like "for the year ended", "financial year", "FY", followed by dates
- Express the financial year in the format "YYYY-YY" or as stated in the document
"""
            few_shot_examples = """
Example 1:
Context: "...Annual Report for the Financial Year 2022-23..."
Question: Financial year to which financial statements relates
Answer: 2022-23
"""
        
        elif "8 digit code" in question:
            specific_instructions = """
- Look for 8-digit codes associated with specific products or services
- The code might be labeled as ITC code, NPCS code, HS code, or product code
- Often found in product classification or export-import documentation
"""
            few_shot_examples = """
Example 1:
Context: "The company's primary product falls under ITC code 52051200..."
Question: Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
Answer: 52051200
"""
        
        elif "4 digit code" in question:
            specific_instructions = """
- Look for 4-digit codes associated with product or service categories
- The code might be labeled as ITC code, NPCS code, HS code, or NIC code
"""
            few_shot_examples = """
Example 1:
Context: "...The company primarily operates in the business of textile products with ITC code 5205..."
Question: Product or service category code (ITC/ NPCS 4 digit code)
Answer: 5205
"""
        
        elif "description" in question.lower():
            specific_instructions = """
- Extract the exact description as stated in the document
- Look in business segment reporting, notes to accounts, or product sections
- For highest contributing product, look for the main business description
"""
            few_shot_examples = """
Example 1:
Context: "The company is primarily engaged in manufacturing and trading of cotton yarn and fabrics..."
Question: Description of the product or service
Answer: Manufacturing and trading of cotton yarn and fabrics
"""
        
        elif "turnover" in question.lower() and "category" in question.lower():
            specific_instructions = """
- Look for financial figures associated with revenue, turnover, or sales of the product/service category
- Extract the exact amount including the unit (e.g., "Rs. 10,000,000" or "₹ 10 crores")
"""
        
        # Create enhanced prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}

You will be provided with a question and context from a PDF document. Your task is to extract the precise answer to the question from the context.

{specific_instructions}

Follow these general guidelines:
- Look carefully through ALL the provided context
- If the answer is explicitly stated in the text, provide that exact answer
- For numerical values, provide the number without currency symbols unless specifically asked
- If the answer is not in the context, respond with "Information not found in the provided context"
- Provide only the direct answer without explanations or additional text

{few_shot_examples}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Context from PDF:
{context}

Answer the question based only on the information in the context. Look carefully through the entire context before concluding the information is not available.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def _post_process_answer(self, question: str, answer: str) -> str:
        """Enhanced post-processing for better answer extraction"""
        # Remove any explanatory text or prefixes
        cleaned = re.sub(r'^(Answer:|The answer is:|Based on the context,|According to the context,)', '', answer).strip()
        
        # Handle "not found" responses consistently
        if re.search(r'(not found|not provided|not mentioned|couldn\'t find|could not find|no information)', 
                     cleaned, re.IGNORECASE):
            return "Information not found in the provided context"
        
        # Enhanced processing for turnover questions
        if "turnover" in question.lower() and ("highest" in question.lower() or "contributing" in question.lower()):
            # Look for various number formats
            number_patterns = [
                r'(?:Rs\.?\s*|₹\s*)?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crores?|lakhs?)?',
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:crores?|lakhs?)',
                r'(\d+(?:,\d+)*)'
            ]
            
            for pattern in number_patterns:
                match = re.search(pattern, cleaned, re.IGNORECASE)
                if match:
                    number = match.group(1).replace(',', '')
                    # Convert to base number if needed
                    if 'crore' in cleaned.lower():
                        number = str(int(float(number) * 10000000))
                    elif 'lakh' in cleaned.lower():
                        number = str(int(float(number) * 100000))
                    return number
        
        # Format CIN consistently
        if "CIN" in question or "corporate identity number" in question.lower():
            cin_match = re.search(r'[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}', cleaned)
            if cin_match:
                return cin_match.group(0)
        
        # Format financial year consistently
        if "financial year" in question.lower():
            fy_match = re.search(r'(FY|F\.Y\.|Financial Year)[ :]?([0-9]{4}[-/ ][0-9]{2,4})', cleaned, re.IGNORECASE)
            if fy_match:
                return fy_match.group(2)
            
            fy_match2 = re.search(r'([0-9]{4}[-/ ][0-9]{2,4})', cleaned)
            if fy_match2:
                return fy_match2.group(1)
        
        # Format product codes consistently
        if "code" in question.lower():
            if "4 digit" in question.lower():
                code_match = re.search(r'\b(\d{4})\b', cleaned)
                if code_match:
                    return code_match.group(1)
            
            if "8 digit" in question.lower():
                code_match = re.search(r'\b(\d{8})\b', cleaned)
                if code_match:
                    return code_match.group(1)
        
        return cleaned

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self, config_path: str = "config1.json"):
        self.config_path = config_path
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
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
            "Corporate identity number (CIN) of company": ["CIN", "corporate identity", "L", "U", "registration", "incorporation"],
            "Financial year to which financial statements relates": ["financial year", "FY", "year ended", "period ended", "F.Y."],
            "Product or service category code (ITC/ NPCS 4 digit code)": ["ITC", "NPCS", "product code", "4 digit", "classification"],
            "Description of the product or service category": ["product category", "service category", "business segment", "nature of business"],
            "Turnover of the product or service category (in Rupees)": ["category turnover", "segment turnover", "revenue", "sales"],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": ["8 digit", "highest turnover", "main product", "primary", "major"],
            "Description of the product or service": ["product description", "service description", "main offering", "primary business"],
            "Turnover of highest contributing product or service (in Rupees)": ["highest turnover", "main revenue", "primary income", "major sales", "principal earnings"]
        }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a PDF file with enhanced context handling"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            extract_start = time.time()
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            print(f"PDF text extraction completed in {time.time() - extract_start:.2f} seconds")
            
            answers = {}
            
            # Process each question
            for question in self.questions:
                question_start = time.time()
                print(f"Processing question: {question}")
                
                keywords = self.question_keywords.get(question, [])
                
                # Enhanced page search
                search_start = time.time()
                relevant_page_nums = self.pdf_extractor.search_keywords_in_pdf(pages, keywords, question)
                print(f"  Page search took {time.time() - search_start:.2f} seconds")
                print(f"  Found relevant pages: {relevant_page_nums}")
                
                # Build context with more pages for problematic questions
                context_start = time.time()
                context = ""
                
                # For the last question (highest turnover), use more comprehensive context
                if "highest contributing" in question.lower() and "turnover" in question.lower():
                    # Use more pages and broader search
                    if len(relevant_page_nums) < 3:
                        # Add more pages from the document
                        additional_pages = list(range(1, min(15, len(pages) + 1)))
                        relevant_page_nums = sorted(list(set(relevant_page_nums + additional_pages)))
                        print(f"  Expanded search for turnover question to pages: {relevant_page_nums}")
                
                for page_num in relevant_page_nums:
                    page_indices = [i for i, p in enumerate(pages) if p["page_num"] == page_num]
                    if page_indices:
                        context += f"\n--- Page {page_num} ---\n"
                        context += pages[page_indices[0]]["text"]
                
                # Increase context length for turnover questions
                max_context_length = 8000 if "turnover" in question.lower() else 4000
                if len(context) > max_context_length:
                    context = context[:max_context_length]
                
                print(f"  Context preparation took {time.time() - context_start:.2f} seconds")
                print(f"  Context length: {len(context)} characters")
                
                # Generate answer
                llm_start = time.time()
                answer = self.qa_processor.generate_answer(question, context)
                print(f"  LLM processing took {time.time() - llm_start:.2f} seconds")
                print(f"  Answer: {answer}")
                
                answers[question] = answer
                print(f"Question processed in {time.time() - question_start:.2f} seconds")
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
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
                answers = self.process_pdf(pdf_path)
                
                result = {"PDF Filename": pdf_file}
                result.update(answers)
                
                results.append(result)
                print(f"Processed {pdf_file} in {time.time() - pdf_start_time:.2f} seconds")
                
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
            print(f"Results saved to {output_excel}")
        else:
            print("No results to save")
        
        print(f"Total batch processing completed in {time.time() - batch_start_time:.2f} seconds")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--config", type=str, default="config1.json", help="Configuration file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    args = parser.parse_args()
    
    try:
        print("Starting Enhanced PDF Regulatory Information Extraction")
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
        
        print(f"Processing completed successfully in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
