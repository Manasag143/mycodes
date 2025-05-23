import os
import json
import time
import pandas as pd
import fitz  # PyMuPDF
import requests
import warnings
from typing import Dict, List, Any

# Suppress SSL warnings
warnings.filterwarnings('ignore')

# Define the Llama URL
LLAMA_URL = "https://ue1-llm.crisil.local/llama3_3/70b/llm/"

class StreamlinedKPIExtractor:
    """Streamlined PDF KPI extraction with single API call approach"""
    
    def __init__(self, llm_url: str = LLAMA_URL):
        self.llm_url = llm_url
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification
        
        # Define the 9 KPIs
        self.kpis = [
            "Corporate identity number (CIN) of company",
            "Financial year to which financial statements relates", 
            "Product or service category code (ITC/ NPCS 4 digit code)",
            "Description of the product or service category",
            "Turnover of the product or service category (in Rupees)",
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
            "Description of the product or service",
            "Turnover of highest contributing product or service (in Rupees)",
            "Main business activity description"
        ]
        
        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": 3048,
            "return_full_text": False,
            "temperature": 0.1
        }
    
    def extract_pages_1_and_10(self, pdf_path: str) -> str:
        """Extract text from page 1 and page 10 of PDF"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            context = ""
            
            # Extract page 1
            if len(doc) >= 1:
                page1 = doc[0]
                context += "#### PAGE 1 ####\n"
                context += page1.get_text()
                context += "\n\n"
            
            # Extract page 10 (if exists)
            if len(doc) >= 10:
                page10 = doc[9]  # 0-indexed
                context += "#### PAGE 10 ####\n"
                context += page10.get_text()
            
            doc.close()
            print(f"PDF extraction took {time.time() - start_time:.2f} seconds")
            return context
            
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return ""
    
    def create_extraction_prompt(self, context: str) -> str:
        """Create the structured prompt for all 9 KPIs extraction"""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert in extracting regulatory and financial information from PDF documents. You must extract specific KPI values with high precision and return them in exact JSON format.

CRITICAL INSTRUCTIONS:
- Extract ONLY the requested information
- If any information is not found in the provided context, return "-" for that field
- Return ONLY the JSON object, no additional text or explanations
- Do not add any commentary or notes
- For codes, extract only the numeric digits
- For financial figures, extract only the number without currency symbols
- For CIN, extract the complete alphanumeric code
- For financial year, use format like "2023-24" or "2023"

<|eot_id|><|start_header_id|>user<|end_header_id|>

Extract the following 9 KPI values from the provided PDF context and return them in JSON format:

1. Corporate identity number (CIN) of company
2. Financial year to which financial statements relates
3. Product or service category code (ITC/ NPCS 4 digit code) 
4. Description of the product or service category
5. Turnover of the product or service category (in Rupees)
6. Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
7. Description of the product or service
8. Turnover of highest contributing product or service (in Rupees)
9. Main business activity description

Return the response in this exact JSON format:
{{
    "cin": "value_or_-",
    "financial_year": "value_or_-", 
    "product_category_code": "value_or_-",
    "product_category_description": "value_or_-",
    "product_category_turnover": "value_or_-",
    "highest_product_code": "value_or_-",
    "highest_product_description": "value_or_-", 
    "highest_product_turnover": "value_or_-",
    "main_business_activity": "value_or_-"
}}

PDF CONTEXT:
{context}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return prompt
    
    def call_llm_api(self, prompt: str) -> Dict[str, str]:
        """Make single API call to extract all KPIs"""
        start_time = time.time()
        
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        
        try:
            response = self.session.post(self.llm_url, json=body, timeout=60)
            
            if response.status_code != 200:
                print(f"API Error: Status {response.status_code}")
                return self._get_empty_result()
            
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                result_text = response_json[0].get('generated_text', '')
                print(f"LLM API call took {time.time() - start_time:.2f} seconds")
                
                # Parse JSON from response
                return self._parse_json_response(result_text)
            else:
                print(f"Unexpected API response format")
                return self._get_empty_result()
                
        except requests.exceptions.Timeout:
            print(f"API call timed out after {time.time() - start_time:.2f} seconds")
            return self._get_empty_result()
        except Exception as e:
            print(f"API call failed: {e}")
            return self._get_empty_result()
    
    def _parse_json_response(self, response_text: str) -> Dict[str, str]:
        """Parse JSON response from LLM"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Ensure all expected keys are present
                expected_keys = [
                    "cin", "financial_year", "product_category_code",
                    "product_category_description", "product_category_turnover",
                    "highest_product_code", "highest_product_description", 
                    "highest_product_turnover", "main_business_activity"
                ]
                
                for key in expected_keys:
                    if key not in result:
                        result[key] = "-"
                
                return result
            else:
                print("No valid JSON found in response")
                return self._get_empty_result()
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return self._get_empty_result()
    
    def _get_empty_result(self) -> Dict[str, str]:
        """Return empty result with all fields set to '-'"""
        return {
            "cin": "-",
            "financial_year": "-", 
            "product_category_code": "-",
            "product_category_description": "-",
            "product_category_turnover": "-",
            "highest_product_code": "-",
            "highest_product_description": "-", 
            "highest_product_turnover": "-",
            "main_business_activity": "-"
        }
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF and extract all 9 KPIs"""
        start_time = time.time()
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        
        try:
            # Step 1: Extract pages 1 and 10
            context = self.extract_pages_1_and_10(pdf_path)
            
            if not context.strip():
                print("No content extracted from PDF")
                return self._get_empty_result()
            
            # Step 2: Create prompt with all 9 KPIs
            prompt = self.create_extraction_prompt(context)
            
            # Step 3: Single API call to get all results
            results = self.call_llm_api(prompt)
            
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return self._get_empty_result()
    
    def process_pdf_directory(self, pdf_dir: str, output_excel: str):
        """Process all PDFs in directory and create Excel output"""
        start_time = time.time()
        print(f"Processing all PDFs in directory: {pdf_dir}")
        
        # Check if directory exists
        if not os.path.isdir(pdf_dir):
            print(f"Directory not found: {pdf_dir}")
            return
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        results = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            
            try:
                # Extract KPIs
                kpi_results = self.process_single_pdf(pdf_path)
                
                # Create row for Excel
                row = {
                    "PDF_Name": pdf_file,
                    "Corporate_Identity_Number_CIN": kpi_results["cin"],
                    "Financial_Year": kpi_results["financial_year"],
                    "Product_Category_Code_4digit": kpi_results["product_category_code"],
                    "Product_Category_Description": kpi_results["product_category_description"],
                    "Product_Category_Turnover": kpi_results["product_category_turnover"],
                    "Highest_Product_Code_8digit": kpi_results["highest_product_code"],
                    "Highest_Product_Description": kpi_results["highest_product_description"],
                    "Highest_Product_Turnover": kpi_results["highest_product_turnover"],
                    "Main_Business_Activity": kpi_results["main_business_activity"]
                }
                
                results.append(row)
                print(f"✓ Completed: {pdf_file}")
                
            except Exception as e:
                print(f"✗ Error processing {pdf_file}: {e}")
                # Add error row
                error_row = {
                    "PDF_Name": pdf_file,
                    "Corporate_Identity_Number_CIN": "ERROR",
                    "Financial_Year": "ERROR",
                    "Product_Category_Code_4digit": "ERROR",
                    "Product_Category_Description": "ERROR",
                    "Product_Category_Turnover": "ERROR",
                    "Highest_Product_Code_8digit": "ERROR",
                    "Highest_Product_Description": "ERROR",
                    "Highest_Product_Turnover": "ERROR",
                    "Main_Business_Activity": "ERROR"
                }
                results.append(error_row)
        
        # Create Excel file
        if results:
            df = pd.DataFrame(results)
            df.to_excel(output_excel, index=False)
            print(f"\n✓ Results saved to: {output_excel}")
            print(f"✓ Total processing time: {time.time() - start_time:.2f} seconds")
            print(f"✓ Processed {len(results)} files")
        else:
            print("No results to save")

def main():
    """Main function to run the KPI extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Streamlined PDF KPI Extraction")
    parser.add_argument("--pdf_dir", type=str, 
                       default="C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's", 
                       help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="kpi_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, help="Process single PDF file")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = StreamlinedKPIExtractor()
    
    if args.single_pdf:
        # Process single PDF
        if not os.path.isfile(args.single_pdf):
            print(f"PDF file not found: {args.single_pdf}")
            return
        
        results = extractor.process_single_pdf(args.single_pdf)
        print(f"\nExtracted KPIs:")
        for key, value in results.items():
            print(f"{key}: {value}")
    else:
        # Process directory
        extractor.process_pdf_directory(args.pdf_dir, args.output)
        
    print("\n" + "="*50)
    print("KPI EXTRACTION COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()
