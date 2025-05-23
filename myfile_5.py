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
    
    def extract_key_pages_smart(self, pdf_path: str) -> str:
        """Extract text from page 1 and intelligently select between page 10/11 based on keywords"""
        start_time = time.time()
        try:
            doc = fitz.open(pdf_path)
            context = ""
            
            # Extract page 1 (always include)
            if len(doc) >= 1:
                page1 = doc[0]
                context += "#### PAGE 1 ####\n"
                context += page1.get_text()
                context += "\n\n"
            
            # Define financial keywords for smart page selection
            financial_keywords = [
                "segment revenue", "turnover", "revenue from operations", 
                "business segment", "ITC code", "NPCS code", "product wise",
                "segment wise", "category wise", "revenue breakdown",
                "sales breakdown", "income from operations", "segment information",
                "product category", "service category", "main products",
                "principal products", "segment results", "revenue split"
            ]
            
            # Analyze page 10 and 11 for keyword density
            page10_text = ""
            page11_text = ""
            page10_score = 0
            page11_score = 0
            
            # Extract and score page 10
            if len(doc) >= 10:
                page10_text = doc[9].get_text().lower()  # 0-indexed
                page10_score = sum(1 for keyword in financial_keywords if keyword in page10_text)
            
            # Extract and score page 11
            if len(doc) >= 11:
                page11_text = doc[10].get_text().lower()  # 0-indexed
                page11_score = sum(1 for keyword in financial_keywords if keyword in page11_text)
            
            # Smart page selection logic
            if page10_score > 0 and page11_score > 0:
                # Both pages have keywords
                if abs(page10_score - page11_score) <= 2:
                    # Scores are close, use both pages
                    if len(doc) >= 10:
                        context += "#### PAGE 10 ####\n"
                        context += doc[9].get_text()
                        context += "\n\n"
                    if len(doc) >= 11:
                        context += "#### PAGE 11 ####\n"
                        context += doc[10].get_text()
                    print(f"  Using both pages 10 & 11 (scores: {page10_score}, {page11_score})")
                elif page10_score > page11_score:
                    # Page 10 has higher score
                    context += "#### PAGE 10 ####\n"
                    context += doc[9].get_text()
                    print(f"  Using page 10 (score: {page10_score} vs {page11_score})")
                else:
                    # Page 11 has higher score
                    context += "#### PAGE 11 ####\n"
                    context += doc[10].get_text()
                    print(f"  Using page 11 (score: {page11_score} vs {page10_score})")
            elif page10_score > 0:
                # Only page 10 has keywords
                context += "#### PAGE 10 ####\n"
                context += doc[9].get_text()
                print(f"  Using page 10 (score: {page10_score})")
            elif page11_score > 0:
                # Only page 11 has keywords
                context += "#### PAGE 11 ####\n"
                context += doc[10].get_text()
                print(f"  Using page 11 (score: {page11_score})")
            else:
                # Neither page has keywords, use both as fallback
                if len(doc) >= 10:
                    context += "#### PAGE 10 ####\n"
                    context += doc[9].get_text()
                    context += "\n\n"
                if len(doc) >= 11:
                    context += "#### PAGE 11 ####\n"
                    context += doc[10].get_text()
                print(f"  Using both pages 10 & 11 (no keywords found)")
            
            doc.close()
            print(f"PDF extraction took {time.time() - start_time:.2f} seconds")
            return context
            
        except Exception as e:
            print(f"Error extracting PDF {pdf_path}: {e}")
            return ""
    
    def create_extraction_prompt(self, context: str) -> str:
        """Create the structured prompt for all 9 KPIs extraction"""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert in extracting regulatory and financial information from PDF documents with high precision and accuracy.

CRITICAL INSTRUCTIONS:
- Extract ONLY the requested information
- If any information is not found in the provided context, return "-" for that field
- Return ONLY the JSON object, no additional text or explanations
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
6. Turnover of the product or service category (in Rupees) - Look for segment-wise revenue/turnover figures
7. Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
8. Description of the product or service
9. Turnover of highest contributing product or service (in Rupees) - Look for highest revenue segment or main business turnover

IMPORTANT FOR TURNOVER EXTRACTION:
- Search for financial data in segment reporting, revenue breakdowns, notes to accounts
- Look for keywords: "Revenue from operations", "Segment revenue", "Turnover", "Net sales", "Income"
- Extract numerical values from tables showing segment-wise or product-wise revenue
- If segment data not available, use total revenue for highest contributing product
- Return only numbers (e.g., "245000000" not "Rs. 245 crores")

MULTIPLE ENTITIES HANDLING:
- If PDF contains data for multiple companies/entities, provide separate JSON for each
- Separate multiple JSONs using semicolon (;) between them
- Example: {{"cin":"L123..."}} ; {{"cin":"L456..."}}

Return the response in this exact JSON format:
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
        """Make single API call to extract all KPIs - returns list to handle multiple JSONs"""
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
                
                # Parse JSON from response (handles multiple JSONs)
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
        """Parse JSON response from LLM - handles multiple JSONs separated by semicolon"""
        try:
            # First, check if response contains semicolon-separated JSONs
            if ';' in response_text and '{' in response_text:
                # Split by semicolon and process each JSON
                json_parts = response_text.split(';')
                results = []
                
                for part in json_parts:
                    part = part.strip()
                    if '{' in part and '}' in part:
                        start_idx = part.find('{')
                        end_idx = part.rfind('}') + 1
                        json_str = part[start_idx:end_idx]
                        
                        try:
                            result = json.loads(json_str)
                            # Ensure all expected keys are present
                            validated_result = self._validate_json_keys(result)
                            results.append(validated_result)
                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON parts
                
                if results:
                    print(f"  Found {len(results)} JSON objects in response")
                    return results
            
            # Single JSON handling (original logic)
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                validated_result = self._validate_json_keys(result)
                return [validated_result]  # Return as list for consistency
            else:
                print("No valid JSON found in response")
                return [self._get_empty_result()]
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return [self._get_empty_result()]
    
    def _validate_json_keys(self, result: Dict[str, str]) -> Dict[str, str]:
        """Validate and ensure all expected keys are present in JSON"""
        expected_keys = [
            "cin", "financial_year", "company_name", "product_category_code",
            "product_category_description", "product_category_turnover",
            "highest_product_code", "highest_product_description", 
            "highest_product_turnover"
        ]
        
        validated_result = {}
        for key in expected_keys:
            validated_result[key] = result.get(key, "-")
        
        return validated_result
    
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
    
    def process_single_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """Process a single PDF and extract all 9 KPIs - returns list to handle multiple entities"""
        start_time = time.time()
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        
        try:
            # Step 1: Extract pages 1 and smart selection of 10/11
            context = self.extract_key_pages_smart(pdf_path)
            
            if not context.strip():
                print("No content extracted from PDF")
                return [self._get_empty_result()]
            
            # Step 2: Create prompt with all 9 KPIs
            prompt = self.create_extraction_prompt(context)
            
            # Step 3: Single API call to get all results (may return multiple JSONs)
            results = self.call_llm_api(prompt)
            
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return [self._get_empty_result()]
    
    def process_pdf_directory(self, pdf_dir: str, output_excel: str):
        """Process all PDFs in directory and create Excel output with batch saving every 50 PDFs"""
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
        print(f"Will save results every 50 PDFs to prevent data loss")
        
        # Process PDFs in batches of 50
        batch_size = 50
        all_results = []
        
        for i in range(0, len(pdf_files), batch_size):
            batch_files = pdf_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(pdf_files) + batch_size - 1) // batch_size
            
            print(f"\n" + "="*60)
            print(f"PROCESSING BATCH {batch_num}/{total_batches}")
            print(f"PDFs {i+1}-{min(i+batch_size, len(pdf_files))} of {len(pdf_files)}")
            print("="*60)
            
            batch_results = []
            batch_start_time = time.time()
            
            # Process each PDF in current batch
            for pdf_file in batch_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                
                try:
                    # Extract KPIs (may return multiple entities)
                    kpi_results_list = self.process_single_pdf(pdf_path)
                    
                    # Handle multiple entities from same PDF
                    for idx, kpi_results in enumerate(kpi_results_list):
                        # Create row for Excel
                        pdf_display_name = pdf_file if idx == 0 else f"{pdf_file}_entity_{idx+1}"
                        
                        row = {
                            "PDF_Name": pdf_display_name,
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
                    
                    if len(kpi_results_list) > 1:
                        print(f"✓ Completed: {pdf_file} (found {len(kpi_results_list)} entities)")
                    else:
                        print(f"✓ Completed: {pdf_file}")
                    
                except Exception as e:
                    print(f"✗ Error processing {pdf_file}: {e}")
                    # Add error row
                    error_row = {
                        "PDF_Name": pdf_file,
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
                print(f"   ✓ Processed: {len(batch_results)} PDFs")
                print(f"   ✓ Time taken: {batch_time:.2f} seconds")
                print(f"   ✓ Batch saved: {batch_filename}")
                print(f"   ✓ Main file updated: {output_excel}")
                print(f"   ✓ Total processed so far: {len(all_results)} PDFs")
        
        # Final summary (no need to create another file as main file is already updated)
        if all_results:
            total_time = time.time() - start_time
            print(f"\n" + "="*60)
            print("🎉 ALL PROCESSING COMPLETED!")
            print("="*60)
            print(f"✓ Total PDFs processed: {len(all_results)}")
            print(f"✓ Total time taken: {total_time:.2f} seconds")
            print(f"✓ Average time per PDF: {total_time/len(all_results):.2f} seconds")
            print(f"✓ Final results saved: {output_excel}")
            print(f"✓ Individual batch files: {(len(pdf_files) + batch_size - 1) // batch_size}")
            print("="*60)
        else:
            print("No results to save")

def main():
    """Main function to run the KPI extractor"""
    print("="*60)
    print("          PDF KPI EXTRACTION PIPELINE")
    print("="*60)
    print("📄 Smart page extraction: 1 + best of (10/11)")
    print("🤖 Processing 9 KPIs in single API call")
    print("🏢 Handling multiple entities per PDF")
    print("💾 Auto-saving every 50 PDFs for safety")
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
