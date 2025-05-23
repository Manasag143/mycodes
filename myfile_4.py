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
            "max_new_tokens": 1024,  # Reduced for JSON responses
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
    
    def get_specific_pages(self, pages: List[Dict[str, Any]], page_numbers: List[int]) -> str:
        """Get text from specific pages"""
        context = ""
        for page_num in page_numbers:
            # Find the right page
            page_indices = [i for i, p in enumerate(pages) if p["page_num"] == page_num]
            if page_indices:
                context += f"\n--- Page {page_num} ---\n"
                context += pages[page_indices[0]]["text"]
        return context

class QuestionAnswerProcessor:
    """Class for processing PDF content to answer regulatory information questions"""
    
    def __init__(self, llm_generator: LLMGenerator):
        self.llm_generator = llm_generator
    
    def generate_json_answer(self, context: str, question_type: str) -> Dict[str, str]:
        """Generate JSON answer for regulatory questions"""
        start_time = time.time()
        
        # Create a prompt for JSON extraction
        prompt = self._create_json_prompt(context, question_type)
        
        # Generate the answer
        raw_answer = self.llm_generator.run(prompt)
        
        # Parse JSON response
        parsed_answer = self._parse_json_response(raw_answer, question_type)
        
        print(f"JSON answer generation took {time.time() - start_time:.2f} seconds")
        return parsed_answer
    
    def _create_json_prompt(self, context: str, question_type: str) -> str:
        """Create a simplified JSON extraction prompt"""
        
        if question_type == "basic_info":
            # For page 1 - basic company info
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert at extracting company information from regulatory documents.<|eot_id|><|start_header_id|>user<|end_header_id|>

Instructions:
1. Extract info from the context delimited by ####.
2. If value not present return "-".
3. No explanation needed, only the output JSON.

Extract below attributes in JSON format from the context:
- Company_Name: Extract the full name of the company
- Company_CIN: Extract CIN Number (format: L/U followed by numbers and letters)
- Financial_Year_Date: Extract the financial year period (e.g., "2023-24")

####
{context}
####

Output JSON:
{{"Company_Name": "", "Company_CIN": "", "Financial_Year_Date": ""}}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        else:
            # For page 10 - product/service info
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert at extracting product and service information from regulatory documents.<|eot_id|><|start_header_id|>user<|end_header_id|>

Instructions:
1. Extract info from the context delimited by ####.
2. If value not present return "-".
3. No explanation needed, only the output JSON.

Extract below attributes in JSON format from the context:
- Product_Service_Category_Code_4digit: 4-digit ITC/NPCS code
- Product_Service_Category_Description: Description of the category
- Product_Service_Category_Turnover: Turnover amount in Rupees
- Highest_Product_Service_Code_8digit: 8-digit code for highest contributing product
- Highest_Product_Service_Description: Description of highest contributing product/service  
- Highest_Product_Service_Turnover: Turnover of highest contributing product/service in Rupees

####
{context}
####

Output JSON:
{{"Product_Service_Category_Code_4digit": "", "Product_Service_Category_Description": "", "Product_Service_Category_Turnover": "", "Highest_Product_Service_Code_8digit": "", "Highest_Product_Service_Description": "", "Highest_Product_Service_Turnover": ""}}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return prompt
    
    def _parse_json_response(self, response: str, question_type: str) -> Dict[str, str]:
        """Parse JSON response from LLM"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            else:
                print(f"Could not find JSON in response: {response[:200]}...")
                # Return default structure based on question type
                if question_type == "basic_info":
                    return {"Company_Name": "-", "Company_CIN": "-", "Financial_Year_Date": "-"}
                else:
                    return {
                        "Product_Service_Category_Code_4digit": "-",
                        "Product_Service_Category_Description": "-", 
                        "Product_Service_Category_Turnover": "-",
                        "Highest_Product_Service_Code_8digit": "-",
                        "Highest_Product_Service_Description": "-",
                        "Highest_Product_Service_Turnover": "-"
                    }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response[:200]}...")
            # Return default structure
            if question_type == "basic_info":
                return {"Company_Name": "-", "Company_CIN": "-", "Financial_Year_Date": "-"}
            else:
                return {
                    "Product_Service_Category_Code_4digit": "-",
                    "Product_Service_Category_Description": "-", 
                    "Product_Service_Category_Turnover": "-",
                    "Highest_Product_Service_Code_8digit": "-",
                    "Highest_Product_Service_Description": "-",
                    "Highest_Product_Service_Turnover": "-"
                }

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self):
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
        # Define the output columns for Excel
        self.output_columns = [
            "PDF Filename",
            "Company_Name", 
            "Company_CIN",
            "Financial_Year_Date",
            "Product_Service_Category_Code_4digit",
            "Product_Service_Category_Description",
            "Product_Service_Category_Turnover", 
            "Highest_Product_Service_Code_8digit",
            "Highest_Product_Service_Description",
            "Highest_Product_Service_Turnover"
        ]
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a PDF file with optimized page-specific extraction"""
        total_start_time = time.time()
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            extract_start = time.time()
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            print(f"PDF text extraction completed in {time.time() - extract_start:.2f} seconds")
            
            # Initialize results dictionary
            results = {}
            
            # Process Page 1 for basic company info
            print("Processing Page 1 for basic company information...")
            page1_context = self.pdf_extractor.get_specific_pages(pages, [1])
            basic_info = self.qa_processor.generate_json_answer(page1_context, "basic_info")
            results.update(basic_info)
            
            # Process Page 10 for product/service info
            print("Processing Page 10 for product/service information...")
            page10_context = self.pdf_extractor.get_specific_pages(pages, [10])
            
            # If page 10 doesn't exist or is empty, try nearby pages
            if not page10_context.strip() or len(page10_context.strip()) < 100:
                print("Page 10 seems empty, trying pages 9-12...")
                page10_context = self.pdf_extractor.get_specific_pages(pages, [9, 10, 11, 12])
            
            product_info = self.qa_processor.generate_json_answer(page10_context, "product_info")
            results.update(product_info)
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return default structure with errors
            return {
                "Company_Name": f"Error: {str(e)}",
                "Company_CIN": f"Error: {str(e)}",
                "Financial_Year_Date": f"Error: {str(e)}",
                "Product_Service_Category_Code_4digit": f"Error: {str(e)}",
                "Product_Service_Category_Description": f"Error: {str(e)}",
                "Product_Service_Category_Turnover": f"Error: {str(e)}",
                "Highest_Product_Service_Code_8digit": f"Error: {str(e)}",
                "Highest_Product_Service_Description": f"Error: {str(e)}",
                "Highest_Product_Service_Turnover": f"Error: {str(e)}"
            }
    
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
                error_dict = {col: f"Error: {str(e)}" for col in self.output_columns[1:]}
                result.update(error_dict)
                results.append(result)
        
        # Create a DataFrame and save to Excel
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Reorder columns
            df = df[self.output_columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            
            # Print sample results for verification
            print("\nSample results:")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"\nPDF {i+1}: {result['PDF Filename']}")
                print(f"Company Name: {result.get('Company_Name', 'N/A')}")
                print(f"CIN: {result.get('Company_CIN', 'N/A')}")
                print(f"Financial Year: {result.get('Financial_Year_Date', 'N/A')}")
        else:
            print("No results to save")
        
        print(f"Total batch processing completed in {time.time() - batch_start_time:.2f} seconds")

def main():
    # Simple command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    args = parser.parse_args()
    
    try:
        print("Starting Enhanced PDF Regulatory Information Extraction")
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
                df = df[pipeline.output_columns]
                df.to_excel(args.output, index=False)
                print(f"Results saved to {args.output}")
                
                # Print results for verification
                print("\nExtracted Information:")
                for key, value in answers.items():
                    print(f"{key}: {value}")
                
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
