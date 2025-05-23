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
    
    def __init__(self):
        # Define keywords for each KPI question for better page search
        self.kpi_keywords = {
            "Corporate identity number (CIN) of company": ["CIN", "corporate identity", "corporate identity number", "L17110", "U17110", "registration number"],
            "Financial year to which financial statements relates": ["financial year", "FY", "year ended", "period ended", "financial statements", "2023-24", "2022-23"],
            "Name of the Company": ["company name", "name of the company", "registered name", "corporate name"],
            "Product or service category code (ITC/ NPCS 4 digit code)": ["ITC", "NPCS", "product code", "service code", "4 digit", "category code"],
            "Description of the product or service category": ["product category", "service category", "description", "business segment", "category description"],
            "Turnover of the product or service category (in Rupees)": ["category turnover", "segment turnover", "revenue", "sales", "turnover", "rupees"],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": ["8 digit", "highest turnover", "main product", "contributing product", "highest contributing"],
            "Description of the product or service": ["product description", "service description", "main offering", "primary product", "highest contributing description"],
            "Turnover of highest contributing product or service (in Rupees)": ["highest turnover", "main product turnover", "primary product revenue", "highest contributing turnover", "maximum revenue"]
        }
    
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
    
    def search_relevant_pages(self, pages: List[Dict[str, Any]], kpi_type: str) -> List[int]:
        """Search for pages containing relevant keywords for specific KPI type"""
        relevant_pages = []
        
        if kpi_type == "basic_info":
            # For basic info, check first few pages
            keywords = []
            keywords.extend(self.kpi_keywords["Corporate identity number (CIN) of company"])
            keywords.extend(self.kpi_keywords["Financial year to which financial statements relates"])
            keywords.extend(self.kpi_keywords["Name of the Company"])
        else:
            # For product info, use product-related keywords
            keywords = []
            for key in ["Product or service category code (ITC/ NPCS 4 digit code)",
                       "Description of the product or service category",
                       "Turnover of the product or service category (in Rupees)",
                       "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
                       "Description of the product or service",
                       "Turnover of highest contributing product or service (in Rupees)"]:
                keywords.extend(self.kpi_keywords[key])
        
        # Search for keywords in pages
        for page in pages:
            for keyword in keywords:
                if keyword.lower() in page["text"].lower():
                    if page["page_num"] not in relevant_pages:
                        relevant_pages.append(page["page_num"])
        
        return sorted(relevant_pages)

class QuestionAnswerProcessor:
    """Class for processing PDF content to answer regulatory information questions"""
    
    def __init__(self, llm_generator: LLMGenerator):
        self.llm_generator = llm_generator
        
        # Define the exact KPI questions as column headers
        self.kpi_questions = [
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
    
    def generate_json_answer(self, context: str, question_type: str) -> Dict[str, str]:
        """Generate JSON answer for regulatory questions with examples"""
        start_time = time.time()
        
        # Create a prompt for JSON extraction
        prompt = self._create_json_prompt_with_examples(context, question_type)
        
        # Generate the answer
        raw_answer = self.llm_generator.run(prompt)
        
        # Parse JSON response
        parsed_answer = self._parse_json_response(raw_answer, question_type)
        
        print(f"JSON answer generation took {time.time() - start_time:.2f} seconds")
        return parsed_answer
    
    def _create_json_prompt_with_examples(self, context: str, question_type: str) -> str:
        """Create a clean JSON extraction prompt with 2 examples and proper structure"""
        
        if question_type == "basic_info":
            # For page 1 - basic company info
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert at extracting regulatory information from financial documents.<|eot_id|><|start_header_id|>user<|end_header_id|>

**Instructions:**
1. Extract information from the context (Page 1 content) delimited by ####.
2. If value not present return "-".
3. No explanation needed, only output the JSON.

**KPIs to Extract:**
- Corporate identity number (CIN) of company
- Financial year to which financial statements relates  
- Name of the Company

**Few-shot Examples:**

Example 1:
Context: "ABC Industries Limited Corporate Identity Number (CIN): L17110MH1995PLC087531 Annual Report for the Financial Year 2023-24"
{{"Corporate identity number (CIN) of company": "L17110MH1995PLC087531", "Financial year to which financial statements relates": "2023-24", "Name of the Company": "ABC Industries Limited"}}

Example 2:
Context: "XYZ Corporation CIN: U72200DL2010PTC123456 For the year ended 31st March, 2022"
{{"Corporate identity number (CIN) of company": "U72200DL2010PTC123456", "Financial year to which financial statements relates": "2021-22", "Name of the Company": "XYZ Corporation"}}

**Context from Page 1:**
####
{context}
####

**Output JSON:**<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        else:
            # For page 10 - product/service info
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert at extracting regulatory information from financial documents.<|eot_id|><|start_header_id|>user<|end_header_id|>

**Instructions:**
1. Extract information from the context (Page 10 content) delimited by ####.
2. If value not present return "-".
3. No explanation needed, only output the JSON.

**KPIs to Extract:**
- Product or service category code (ITC/ NPCS 4 digit code)
- Description of the product or service category
- Turnover of the product or service category (in Rupees)
- Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
- Description of the product or service
- Turnover of highest contributing product or service (in Rupees)

**Few-shot Examples:**

Example 1:
Context: "Product Category: Textiles ITC Code: 5205 Description: Cotton yarn and fabrics Category Turnover: Rs. 15,00,00,000 Highest Contributing Product: Code 52050100 Description: Cotton yarn, single Turnover: Rs. 10,50,00,000"
{{"Product or service category code (ITC/ NPCS 4 digit code)": "5205", "Description of the product or service category": "Cotton yarn and fabrics", "Turnover of the product or service category (in Rupees)": "150000000", "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": "52050100", "Description of the product or service": "Cotton yarn, single", "Turnover of highest contributing product or service (in Rupees)": "105000000"}}

Example 2:
Context: "Business Segment: Chemicals NPCS Code: 2011 Segment Description: Basic industrial chemicals Revenue: 2500000000 Main Product Code: 20110015 Product Description: Industrial grade chemicals Primary Product Revenue: 1800000000"
{{"Product or service category code (ITC/ NPCS 4 digit code)": "2011", "Description of the product or service category": "Basic industrial chemicals", "Turnover of the product or service category (in Rupees)": "2500000000", "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": "20110015", "Description of the product or service": "Industrial grade chemicals", "Turnover of highest contributing product or service (in Rupees)": "1800000000"}}

**Context from Page 10:**
####
{context}
####

**Output JSON:**<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

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
                    return {
                        "Corporate identity number (CIN) of company": "-", 
                        "Financial year to which financial statements relates": "-", 
                        "Name of the Company": "-"
                    }
                else:
                    return {
                        "Product or service category code (ITC/ NPCS 4 digit code)": "-",
                        "Description of the product or service category": "-", 
                        "Turnover of the product or service category (in Rupees)": "-",
                        "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": "-",
                        "Description of the product or service": "-",
                        "Turnover of highest contributing product or service (in Rupees)": "-"
                    }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response[:200]}...")
            # Return default structure
            if question_type == "basic_info":
                return {
                    "Corporate identity number (CIN) of company": "-", 
                    "Financial year to which financial statements relates": "-", 
                    "Name of the Company": "-"
                }
            else:
                return {
                    "Product or service category code (ITC/ NPCS 4 digit code)": "-",
                    "Description of the product or service category": "-", 
                    "Turnover of the product or service category (in Rupees)": "-",
                    "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": "-",
                    "Description of the product or service": "-",
                    "Turnover of highest contributing product or service (in Rupees)": "-"
                }

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self):
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
        # Define the exact KPI questions as output columns
        self.kpi_questions = [
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
        
        # Define the output columns for Excel (PDF Filename + KPI questions)
        self.output_columns = ["PDF Filename"] + self.kpi_questions
    
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
            
            # Process Page 1 for basic company info (or search relevant pages)
            print("Processing for basic company information...")
            basic_pages = self.pdf_extractor.search_relevant_pages(pages, "basic_info")
            if not basic_pages:
                basic_pages = [1, 2, 3]  # Fallback to first 3 pages
            
            page1_context = self.pdf_extractor.get_specific_pages(pages, basic_pages[:3])
            basic_info = self.qa_processor.generate_json_answer(page1_context, "basic_info")
            results.update(basic_info)
            
            # Process Page 10 for product/service info (or search relevant pages)
            print("Processing for product/service information...")
            product_pages = self.pdf_extractor.search_relevant_pages(pages, "product_info")
            if not product_pages:
                product_pages = [10, 9, 11, 12]  # Fallback to around page 10
            
            page10_context = self.pdf_extractor.get_specific_pages(pages, product_pages[:4])
            
            # If context is still empty or too short, try more pages
            if not page10_context.strip() or len(page10_context.strip()) < 100:
                print("Expanding search to more pages...")
                extended_pages = list(range(8, min(len(pages) + 1, 15)))
                page10_context = self.pdf_extractor.get_specific_pages(pages, extended_pages)
            
            product_info = self.qa_processor.generate_json_answer(page10_context, "product_info")
            results.update(product_info)
            
            print(f"Completed processing {pdf_path} in {time.time() - total_start_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return default structure with errors
            error_results = {}
            for question in self.kpi_questions:
                error_results[question] = f"Error: {str(e)}"
            return error_results
    
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
                error_dict = {question: f"Error: {str(e)}" for question in self.kpi_questions}
                result.update(error_dict)
                results.append(result)
        
        # Create a DataFrame and save to Excel
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Reorder columns to match exact KPI structure
            df = df[self.output_columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            
            # Print sample results for verification
            print("\nSample results:")
            for i, result in enumerate(results[:2]):  # Show first 2 results
                print(f"\nPDF {i+1}: {result['PDF Filename']}")
                print(f"CIN: {result.get('Corporate identity number (CIN) of company', 'N/A')}")
                print(f"Financial Year: {result.get('Financial year to which financial statements relates', 'N/A')}")
                print(f"Company Name: {result.get('Name of the Company', 'N/A')}")
                print(f"Product Code (4-digit): {result.get('Product or service category code (ITC/ NPCS 4 digit code)', 'N/A')}")
        else:
            print("No results to save")
        
        print(f"Total batch processing completed in {time.time() - batch_start_time:.2f} seconds")

def main():
    # Simple command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced PDF Regulatory Information Extraction with KPI Questions")
    parser.add_argument("--pdf_dir", type=str, default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_kpi_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    args = parser.parse_args()
    
    try:
        print("Starting Enhanced PDF Regulatory Information Extraction with KPI Questions")
        print("KPI Questions being extracted:")
        pipeline = PDFRegulatoryExtractor()
        for i, question in enumerate(pipeline.kpi_questions, 1):
            print(f"{i}. {question}")
        print()
        
        start_time = time.time()
        
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
                for question in pipeline.kpi_questions:
                    print(f"{question}: {answers.get(question, 'N/A')}")
                
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
