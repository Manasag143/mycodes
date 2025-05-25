import os
import json
import time
import pandas as pd
import fitz  # PyMuPDF
import requests
import warnings
from typing import Dict, List, Any
import re

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
            "Name of the Company",
            "Product or service category code (ITC/ NPCS 4 digit code)",
            "Description of the product or service category",
            "Turnover of the product or service category (in Rupees)",
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
            "Description of the product or service",
            "Turnover of highest contributing product or service (in Rupees)"
        ]
        
        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": 3048,
            "return_full_text": False,
            "temperature": 0.1
        }
    
    def extract_text_with_structure(self, page):
        """Extract text with better structure preservation using text blocks"""
        try:
            # Get text blocks with coordinates
            blocks = page.get_text("dict")
            text_lines = []
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        if line_text.strip():
                            text_lines.append({
                                'text': line_text.strip(),
                                'bbox': line["bbox"]
                            })
            
            # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
            text_lines.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
            
            # Group lines that are at similar y-coordinates (same row)
            grouped_lines = []
            current_group = []
            current_y = None
            tolerance = 5  # pixels tolerance for same row
            
            for line in text_lines:
                y_coord = line['bbox'][1]
                if current_y is None or abs(y_coord - current_y) <= tolerance:
                    current_group.append(line)
                    current_y = y_coord if current_y is None else current_y
                else:
                    if current_group:
                        # Sort current group by x-coordinate
                        current_group.sort(key=lambda x: x['bbox'][0])
                        grouped_lines.append(current_group)
                    current_group = [line]
                    current_y = y_coord
            
            # Add the last group
            if current_group:
                current_group.sort(key=lambda x: x['bbox'][0])
                grouped_lines.append(current_group)
            
            # Join text from grouped lines
            structured_text = ""
            for group in grouped_lines:
                line_text = " | ".join([item['text'] for item in group])
                structured_text += line_text + "\n"
            
            return structured_text
            
        except Exception as e:
            print(f"Error in structured extraction, falling back to basic: {e}")
            return page.get_text()
    
    def extract_key_pages(self, pdf_path: str) -> str:
        """Extract text from page 1, page 10, and page 11 of PDF with better structure"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            context = ""
            
            # Extract page 1
            if len(doc) >= 1:
                page1 = doc[0]
                context += "#### PAGE 1 ####\n"
                context += self.extract_text_with_structure(page1)
                context += "\n\n"
            
            # Extract page 10 (if exists) up to Segment III
            if len(doc) >= 10:
                page10 = doc[9]  # 0-indexed
                context += "#### PAGE 10 ####\n"
                page10_text = self.extract_text_with_structure(page10)
                # Stop at Segment III if found
                segment_iii_match = re.search(r'Segment\s+III', page10_text, re.IGNORECASE)
                if segment_iii_match:
                    page10_text = page10_text[:segment_iii_match.start()]
                context += page10_text
                context += "\n\n"
            
            # Extract page 11 (if exists) up to Segment III
            if len(doc) >= 11:
                page11 = doc[10]  # 0-indexed
                context += "#### PAGE 11 ####\n"
                page11_text = self.extract_text_with_structure(page11)
                # Stop at Segment III if found
                segment_iii_match = re.search(r'Segment\s+III', page11_text, re.IGNORECASE)
                if segment_iii_match:
                    page11_text = page11_text[:segment_iii_match.start()]
                context += page11_text
            
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
- If there are multiple product/service categories, create separate JSON objects for each category
- Return an array of JSON objects if multiple categories exist, otherwise return a single JSON object
- Do not add any commentary or notes
- For codes, extract only the numeric digits
- For CIN, extract the complete alphanumeric code
- For financial year, use format like "2023-24" or "2023"
- For company name, extract the complete official name as stated
- For turnover amounts: Look for revenue, sales, turnover figures in segment reporting, financial statements, or notes
- For turnover values: Return only the numerical amount without currency symbols (Rs, ₹, INR) or units (crores, lakhs)
- Product category turnover and highest product turnover might be the same if company has single main segment
- Look in pages for terms like: "Revenue from operations", "Segment revenue", "Turnover", "Sales", "Income from operations"

<|eot_id|><|start_header_id|>user<|end_header_id|>

Extract the following 9 KPI values from the provided PDF context and return them in JSON format:

1. Corporate identity number (CIN) of company
2. Financial year to which financial statements relates
3. Name of the Company
4. Product or service category code (ITC/ NPCS 4 digit code) 
5. Description of the product or service category
6. Turnover of the product or service category (in Rupees)
7. Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
8. Description of the product or service
9. Turnover of highest contributing product or service (in Rupees)

If there are multiple product/service categories, return an array of JSON objects:
[
    {{
        "cin": "value_or_-",
        "financial_year": "value_or_-",
        "company_name": "value_or_-",
        "product_category_code": "value_or_-",
        "product_category_description": "value_or_-",
        "product_category_turnover": "value_or_-",
        "highest_product_code": "value_or_-",
        "highest_product_description": "value_or_-", 
        "highest_product_turnover": "value_or_-"
    }},
    {{
        "cin": "value_or_-",
        "financial_year": "value_or_-",
        "company_name": "value_or_-",
        "product_category_code": "value_or_-",
        "product_category_description": "value_or_-",
        "product_category_turnover": "value_or_-",
        "highest_product_code": "value_or_-",
        "highest_product_description": "value_or_-", 
        "highest_product_turnover": "value_or_-"
    }}
]

If there's only one category, return a single JSON object:
{{
    "cin": "value_or_-",
    "financial_year": "value_or_-",
    "company_name": "value_or_-",
    "product_category_code": "value_or_-",
    "product_category_description": "value_or_-",
    "product_category_turnover": "value_or_-",
    "highest_product_code": "value_or_-",
    "highest_product_description": "value_or_-", 
    "highest_product_turnover": "value_or_-"
}}

PDF CONTEXT:
{context}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        return prompt
    
    def call_llm_api(self, prompt: str) -> List[Dict[str, str]]:
        """Make single API call to extract all KPIs - returns list of results"""
        start_time = time.time()
        
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        
        try:
            response = self.session.post(self.llm_url, json=body, timeout=60)
            
            if response.status_code != 200:
                print(f"API Error: Status {response.status_code}")
                return [self._get_empty_result()]
            
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                result_text = response_json[0].get('generated_text', '')
                print(f"LLM API call took {time.time() - start_time:.2f} seconds")
                
                # Parse JSON from response
                return self._parse_json_response(result_text)
            else:
                print(f"Unexpected API response format")
                return [self._get_empty_result()]
                
        except requests.exceptions.Timeout:
            print(f"API call timed out after {time.time() - start_time:.2f} seconds")
            return [self._get_empty_result()]
        except Exception as e:
            print(f"API call failed: {e}")
            return [self._get_empty_result()]
    
    def _parse_json_response(self, response_text: str) -> List[Dict[str, str]]:
        """Parse JSON response from LLM - handles both single objects and arrays"""
        try:
            # Find JSON content in response
            json_matches = []
            
            # First try to find array pattern
            array_pattern = r'\[[\s\S]*?\]'
            array_match = re.search(array_pattern, response_text)
            
            if array_match:
                try:
                    json_str = array_match.group(0)
                    result = json.loads(json_str)
                    if isinstance(result, list):
                        # Validate and clean each object in the array
                        cleaned_results = []
                        for item in result:
                            cleaned_item = self._validate_and_clean_result(item)
                            cleaned_results.append(cleaned_item)
                        return cleaned_results
                except json.JSONDecodeError:
                    pass
            
            # If no array found, look for single object
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                if isinstance(result, dict):
                    cleaned_result = self._validate_and_clean_result(result)
                    return [cleaned_result]
                elif isinstance(result, list):
                    cleaned_results = []
                    for item in result:
                        cleaned_item = self._validate_and_clean_result(item)
                        cleaned_results.append(cleaned_item)
                    return cleaned_results
            
            print("No valid JSON found in response")
            return [self._get_empty_result()]
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return [self._get_empty_result()]
    
    def _validate_and_clean_result(self, result: Dict) -> Dict[str, str]:
        """Validate and clean a single result object"""
        expected_keys = [
            "cin", "financial_year", "company_name", "product_category_code",
            "product_category_description", "product_category_turnover",
            "highest_product_code", "highest_product_description", 
            "highest_product_turnover"
        ]
        
        cleaned_result = {}
        for key in expected_keys:
            if key in result:
                cleaned_result[key] = str(result[key]) if result[key] is not None else "-"
            else:
                cleaned_result[key] = "-"
        
        return cleaned_result
    
    def _get_empty_result(self) -> Dict[str, str]:
        """Return empty result with all fields set to '-'"""
        return {
            "cin": "-",
            "financial_year": "-",
            "company_name": "-",
            "product_category_code": "-",
            "product_category_description": "-",
            "product_category_turnover": "-",
            "highest_product_code": "-",
            "highest_product_description": "-", 
            "highest_product_turnover": "-"
        }
    
    def check_if_different_kpi_sets(self, results: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        """Group results by different KPI sets (last 6 fields)"""
        if len(results) <= 1:
            return [results]
        
        grouped_results = []
        
        for result in results:
            # Extract last 6 KPI fields for comparison
            kpi_signature = (
                result["product_category_code"],
                result["product_category_description"],
                result["product_category_turnover"],
                result["highest_product_code"],
                result["highest_product_description"],
                result["highest_product_turnover"]
            )
            
            # Check if this signature already exists in any group
            found_group = False
            for group in grouped_results:
                if group:  # Check if group is not empty
                    existing_signature = (
                        group[0]["product_category_code"],
                        group[0]["product_category_description"],
                        group[0]["product_category_turnover"],
                        group[0]["highest_product_code"],
                        group[0]["highest_product_description"],
                        group[0]["highest_product_turnover"]
                    )
                    
                    if kpi_signature == existing_signature:
                        group.append(result)
                        found_group = True
                        break
            
            if not found_group:
                grouped_results.append([result])
        
        return grouped_results
    
    def process_single_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Process a single PDF and extract all 9 KPIs - returns list of results"""
        start_time = time.time()
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        
        try:
            # Step 1: Extract pages 1, 10, and 11
            context = self.extract_key_pages(pdf_path)
            
            if not context.strip():
                print("No content extracted from PDF")
                return [self._get_empty_result()]
            
            # Step 2: Create prompt with all 9 KPIs
            prompt = self.create_extraction_prompt(context)
            
            # Step 3: Single API call to get all results
            results = self.call_llm_api(prompt)
            
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            print(f"Found {len(results)} result(s)")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return [self._get_empty_result()]
    
    def find_all_pdfs(self, root_dir: str) -> List[tuple]:
        """Find all PDF files in nested folder structure"""
        pdf_files = []
        
        print(f"Scanning for PDFs in nested folders...")
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            
            # If it's a directory, look for PDFs inside it
            if os.path.isdir(item_path):
                folder_pdfs = [f for f in os.listdir(item_path) if f.lower().endswith('.pdf')]
                
                if folder_pdfs:
                    for pdf_file in folder_pdfs:
                        pdf_full_path = os.path.join(item_path, pdf_file)
                        # Store tuple of (full_path, relative_path_for_display, folder_name)
                        pdf_files.append((pdf_full_path, f"{item}/{pdf_file}", item))
                        
                    print(f"Found {len(folder_pdfs)} PDF(s) in folder: {item}")
                else:
                    print(f"No PDFs found in folder: {item}")
            
            # Also check for PDFs directly in root directory
            elif item.lower().endswith('.pdf'):
                pdf_full_path = os.path.join(root_dir, item)
                pdf_files.append((pdf_full_path, item, "root"))
        
        return pdf_files

    def process_pdf_directory(self, pdf_dir: str, output_excel: str):
        """Process all PDFs in nested directory structure and create Excel output with batch saving every 50 PDFs"""
        start_time = time.time()
        print(f"Processing all PDFs in directory structure: {pdf_dir}")
        
        # Check if directory exists
        if not os.path.isdir(pdf_dir):
            print(f"Directory not found: {pdf_dir}")
            return
        
        # Get all PDF files from nested folders
        pdf_info_list = self.find_all_pdfs(pdf_dir)
        
        if not pdf_info_list:
            print(f"No PDF files found in {pdf_dir} or its subfolders")
            return
        
        
        print(f"Found {len(pdf_info_list)} PDF files across all subfolders")
        print(f"Will save results every 50 PDFs to prevent data loss")
        
        # Process PDFs in batches of 50
        batch_size = 50
        all_results = []
        
        for i in range(0, len(pdf_info_list), batch_size):
            batch_info = pdf_info_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(pdf_info_list) + batch_size - 1) // batch_size
            
            print(f"\n" + "="*60)
            print(f"PROCESSING BATCH {batch_num}/{total_batches}")
            print(f"PDFs {i+1}-{min(i+batch_size, len(pdf_info_list))} of {len(pdf_info_list)}")
            print("="*60)
            
            batch_results = []
            batch_start_time = time.time()
            
            # Process each PDF in current batch
            for pdf_full_path, pdf_display_path, folder_name in batch_info:
                
                try:
                    # Extract KPIs - now returns list of results
                    kpi_results_list = self.process_single_pdf(pdf_full_path)
                    
                    # Group results by different KPI sets
                    grouped_results = self.check_if_different_kpi_sets(kpi_results_list)
                    
                    # Create rows for Excel - one row per unique KPI set
                    for group in grouped_results:
                        if group:  # Check if group is not empty
                            # Use the first result in group as representative
                            kpi_results = group[0]
                            
                            row = {
                                "PDF_Name": os.path.basename(pdf_full_path),  # Just filename
                                "PDF_Path": pdf_display_path,  # Full relative path
                                "Folder_Name": folder_name,  # Parent folder name
                                "Corporate_Identity_Number_CIN": kpi_results["cin"],
                                "Financial_Year": kpi_results["financial_year"],
                                "Company_Name": kpi_results["company_name"],
                                "Product_Category_Code_4digit": kpi_results["product_category_code"],
                                "Product_Category_Description": kpi_results["product_category_description"],
                                "Product_Category_Turnover": kpi_results["product_category_turnover"],
                                "Highest_Product_Code_8digit": kpi_results["highest_product_code"],
                                "Highest_Product_Description": kpi_results["highest_product_description"],
                                "Highest_Product_Turnover": kpi_results["highest_product_turnover"]
                            }
                            
                            batch_results.append(row)
                            all_results.append(row)
                    
                    print(f"✓ Completed: {pdf_display_path} ({len(grouped_results)} unique KPI set(s))")
                    
                except Exception as e:
                    print(f"✗ Error processing {pdf_display_path}: {e}")
                    # Add error row
                    error_row = {
                        "PDF_Name": os.path.basename(pdf_full_path),
                        "PDF_Path": pdf_display_path,
                        "Folder_Name": folder_name,
                        "Corporate_Identity_Number_CIN": "ERROR",
                        "Financial_Year": "ERROR",
                        "Company_Name": "ERROR",
                        "Product_Category_Code_4digit": "ERROR",
                        "Product_Category_Description": "ERROR",
                        "Product_Category_Turnover": "ERROR",
                        "Highest_Product_Code_8digit": "ERROR",
                        "Highest_Product_Description": "ERROR",
                        "Highest_Product_Turnover": "ERROR"
                    }
                    batch_results.append(error_row)
                    all_results.append(error_row)
            
            # Save batch results to Excel
            if batch_results:
                # Save current batch only
                base_name = output_excel.replace('.xlsx', '')
                batch_filename = f"{base_name}_batch_{batch_num}.xlsx"
                
                batch_df = pd.DataFrame(batch_results)
                batch_df.to_excel(batch_filename, index=False)
                
                # Save cumulative results (all processed so far) - this is the main file
                cumulative_df = pd.DataFrame(all_results)
                cumulative_df.to_excel(output_excel, index=False)
                
                batch_time = time.time() - batch_start_time
                print(f"\n📊 BATCH {batch_num} COMPLETED:")
                print(f"   ✓ Processed: {len(batch_info)} PDFs")
                print(f"   ✓ Generated: {len(batch_results)} rows")
                print(f"   ✓ Time taken: {batch_time:.2f} seconds")
                print(f"   ✓ Batch saved: {batch_filename}")
                print(f"   ✓ Main file updated: {output_excel}")
                print(f"   ✓ Total rows so far: {len(all_results)}")
        
        # Final summary
        if all_results:
            total_time = time.time() - start_time
            print(f"\n" + "="*60)
            print("🎉 ALL PROCESSING COMPLETED!")
            print("="*60)
            print(f"✓ Total PDFs processed: {len(pdf_info_list)}")
            print(f"✓ Total rows generated: {len(all_results)}")
            print(f"✓ Total time taken: {total_time:.2f} seconds")
            print(f"✓ Average time per PDF: {total_time/len(pdf_info_list):.2f} seconds")
            print(f"✓ Final results saved: {output_excel}")
            print(f"✓ Individual batch files: {(len(pdf_info_list) + batch_size - 1) // batch_size}")
            print("="*60)
        else:
            print("No results to save")

def main():
    """Main function to run the KPI extractor"""
    print("="*60)
    print("          PDF KPI EXTRACTION PIPELINE")
    print("="*60)    
    # Default settings
    pdf_dir = "C:\\Users\\c-ManasA\\OneDrive - crisil.com\\Desktop\\New folder\\pdf's"
    output_file = "kpi_results.xlsx"
    
    # Create extractor and process directory
    extractor = StreamlinedKPIExtractor()
    extractor.process_pdf_directory(pdf_dir, output_file)
        
    print("\n" + "="*60)
    print("          KPI EXTRACTION COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
