import os
import re
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
            "max_new_tokens": 5048,
            "return_full_text": False,
            "temperature": 0.1
        }

    def run(self, prompt: str) -> str:
        """Send prompt to LLM and get response"""
        body = {
            "inputs": prompt,
            "parameters": {**self.generation_kwargs}
        }
        try:
            response = requests.post(self.url, verify=False, json=body)
            
            if response.status_code != 200:
                return f"Error: LLM API error: Status {response.status_code}"
            
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                return response_json[0].get('generated_text', '')
            else:
                return "Error: Unexpected response format from LLM API"
                
        except Exception as e:
            return f"Error: {str(e)}"

class PDFExtractor:
    """Class for extracting text and tables from PDF files"""
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from each page of a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                pages.append({
                    "page_num": page_num + 1,
                    "text": text
                })
            return pages
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            raise
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract potential tables from PDF using heuristics"""
        try:
            doc = fitz.open(pdf_path)
            tables = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Look for tabular structures
                digit_patterns = re.findall(r'((?:\d+[\s\t]+){2,}[^\n]+)', text)
                column_patterns = re.findall(r'([^\n]+[\s\t]{3,}[^\n]+[\s\t]{3,}[^\n]+)', text)
                code_patterns = re.findall(r'([^\n]*(ITC|NPCS|HS|NIC)[\s\t]*(code|Code)[^\n]*(?:\n[^\n]+){0,5})', text)
                turnover_patterns = re.findall(
                    r'([^\n]*(turnover|revenue|sales)[\s\t]*(?:of|for)?[\s\t]*(?:product|service)?[^\n]*(?:\n[^\n]+){0,5})', 
                    text, re.IGNORECASE
                )
                cin_patterns = re.findall(r'([^\n]*(CIN|Corporate Identity Number)[^\n]*(?:\n[^\n]+){0,3})', text, re.IGNORECASE)
                fy_patterns = re.findall(
                    r'([^\n]*(financial year|FY|F\.Y\.|year ended)[^\n]*(?:\n[^\n]+){0,3})', 
                    text, re.IGNORECASE
                )
                
                # Combine all findings
                if digit_patterns or column_patterns or code_patterns or turnover_patterns or cin_patterns or fy_patterns:
                    table_text = ""
                    
                    if digit_patterns:
                        table_text += "\nPOTENTIAL CODE PATTERNS:\n" + "\n".join(digit_patterns)
                    if column_patterns:
                        table_text += "\nTABULAR STRUCTURES:\n" + "\n".join(column_patterns)
                    if code_patterns:
                        table_text += "\nITC/NPCS CODE MENTIONS:\n" + "\n".join([p[0] for p in code_patterns])
                    if turnover_patterns:
                        table_text += "\nTURNOVER MENTIONS:\n" + "\n".join([p[0] for p in turnover_patterns])
                    if cin_patterns:
                        table_text += "\nCIN MENTIONS:\n" + "\n".join([p[0] for p in cin_patterns])
                    if fy_patterns:
                        table_text += "\nFINANCIAL YEAR MENTIONS:\n" + "\n".join([p[0] for p in fy_patterns])
                    
                    tables.append({
                        "page_num": page_num + 1,
                        "table_text": table_text
                    })
            
            return tables
        except Exception as e:
            print(f"Error extracting tables from PDF {pdf_path}: {e}")
            return []
            
    def search_keywords_in_pdf(self, pages: List[Dict[str, Any]], keywords: List[str], question: str) -> List[int]:
        """
        Search for keywords in the extracted PDF text with enhanced pattern matching.
        Now also uses the complete question as a search pattern.
        
        Args:
            pages: List of dictionaries containing page number and text
            keywords: List of keywords to search for
            question: The complete question text
            
        Returns:
            List of page numbers where matches were found
        """
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
            return relevant_pages
        
        # Otherwise, fall back to the keyword-based search
        for keyword in keywords:
            # Create variations and patterns for robust matching
            variations = [keyword]
            
            # For CIN search, add regex pattern for CIN format
            if keyword.lower() in ["cin", "corporate identity number"]:
                variations.append(r'[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}')
            
            # For product codes, look for digit patterns
            if "code" in keyword.lower() or "digit" in keyword.lower():
                if "4 digit" in keyword.lower():
                    variations.append(r'\b\d{4}\b')
                if "8 digit" in keyword.lower():
                    variations.append(r'\b\d{8}\b')
            
            # For financial years, look for year patterns
            if "financial year" in keyword.lower() or "fy" == keyword.lower():
                variations.append(r'\b(FY|F\.Y\.|Financial Year)[ :]?[0-9]{4}[-/ ][0-9]{2,4}\b')
                variations.append(r'\b[0-9]{4}[-/ ][0-9]{2,4}\b')
            
            # For turnover and financial figures
            if any(term in keyword.lower() for term in ["turnover", "revenue", "sales", "rupees"]):
                variations.append(r'(?i)(Rs\.?|₹|INR)[ ]?[0-9,.]+[ ]?(lakhs?|crores?|millions?|billions?)')
                variations.append(r'[0-9,.]+[ ]?(lakhs?|crores?|millions?|billions?)')
            
            for page in pages:
                found = False
                for var in variations:
                    # Determine if it's a regex pattern or simple string
                    if var.startswith(r'\b') or var.startswith(r'[') or var.startswith(r'(?'):
                        # It's a regex pattern
                        if re.search(var, page["text"], re.IGNORECASE | re.MULTILINE):
                            found = True
                            break
                    else:
                        # Simple string search
                        if re.search(re.escape(var), page["text"], re.IGNORECASE):
                            found = True
                            break
                
                if found and page["page_num"] not in relevant_pages:
                    relevant_pages.append(page["page_num"])
                    
        return relevant_pages

class QuestionAnswerProcessor:
    """Class for processing PDF content to answer regulatory information questions"""
    
    def __init__(self, llm_generator: LLMGenerator):
        self.llm_generator = llm_generator
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate an answer to a question using the provided context"""
        # Create a prompt for Llama
        prompt = self._create_qa_prompt(question, context)
        
        # Generate the answer
        raw_answer = self.llm_generator.run(prompt)
        
        # Post-process the answer for consistent formatting
        processed_answer = self._post_process_answer(question, raw_answer)
        
        return processed_answer
    
    def _create_qa_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the Llama model to answer a regulatory question with few-shot examples"""
        # Format the prompt for Llama
        sys_message = "You are an expert in extracting business regulatory information from financial documents with high precision and accuracy."
        
        # Create specific instructions based on question type
        specific_instructions = ""
        
        # Few-shot examples based on question type
        few_shot_examples = ""
        
        if "CIN" in question or "corporate identity" in question.lower():
            specific_instructions = """
- For Corporate Identity Number (CIN), look for a code that typically starts with 'L' or 'U' followed by numbers and letters
- A CIN is a 21-digit alphanumeric code (e.g., L12345MH2010PLC123456)
- It may appear in the company information section, header, footer, or directors' report
"""
            few_shot_examples = """
Example 1:
Context: "...The Corporate Identity Number (CIN) of the Company is L17110MH1973PLC019786..."
Question: Corporate identity number (CIN) of company
Answer: L17110MH1973PLC019786

Example 2:
Context: "...Corporate Information: CIN - U72200KA2007PTC041760..."
Question: Corporate identity number (CIN) of company
Answer: U72200KA2007PTC041760
"""
        elif "financial year" in question.lower():
            specific_instructions = """
- Look for phrases like "for the year ended", "financial year", "FY", followed by dates
- Express the financial year in the format "YYYY-YY" or as stated in the document
- Check the header of financial statements or the directors' report
"""
            few_shot_examples = """
Example 1:
Context: "...Annual Report for the Financial Year 2022-23..."
Question: Financial year to which financial statements relates
Answer: 2022-23

Example 2:
Context: "...Statement of Profit and Loss for the year ended March 31, 2023..."
Question: Financial year to which financial statements relates
Answer: 2022-23
"""
        elif "4 digit code" in question:
            specific_instructions = """
- Look for 4-digit codes associated with product or service categories
- Check business segment reporting, notes to accounts, or product classification sections
- The code might be labeled as ITC code, NPCS code, HS code, or NIC code
- Only return the 4-digit code itself, not surrounding text
"""
            few_shot_examples = """
Example 1:
Context: "...The company primarily operates in the business of textile products with ITC code 5205..."
Question: Product or service category code (ITC/ NPCS 4 digit code)
Answer: 5205

Example 2:
Context: "...Main product category as per NIC code 2008: Information Technology services..."
Question: Product or service category code (ITC/ NPCS 4 digit code)
Answer: 6201
"""
        elif "8 digit code" in question:
            specific_instructions = """
- Look for 8-digit codes associated with specific products or services
- These may appear in detailed product listings or segment reporting sections
- The code might be labeled as ITC code, NPCS code, HS code, or product code
- Only return the 8-digit code itself, not surrounding text
"""
            few_shot_examples = """
Example 1:
Context: "...Highest revenue generating product: Cotton yarn with ITC code 52051110..."
Question: Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
Answer: 52051110

Example 2:
Context: "...The specific software product with NPCS code 99831326 contributes the highest revenue..."
Question: Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)
Answer: 99831326
"""
        elif "turnover" in question.lower() and "category" in question.lower():
            specific_instructions = """
- Look for financial figures associated with revenue, turnover, or sales of the product/service category
- Extract the exact amount including the unit (e.g., "Rs. 10,000,000" or "₹ 10 crores")
- Pay attention to whether amounts are in thousands, lakhs, crores, or millions
- Check income statement, segment reporting, or directors' report sections
"""
            few_shot_examples = """
Example 1:
Context: "...The textile segment generated a total turnover of Rs. 450.75 crores during the year..."
Question: Turnover of the product or service category (in Rupees)
Answer: Rs. 450.75 crores

Example 2:
Context: "...Category revenue: IT services - ₹ 2,356.8 lakhs..."
Question: Turnover of the product or service category (in Rupees)
Answer: ₹ 2,356.8 lakhs
"""
        elif "turnover" in question.lower() and "highest" in question.lower():
            specific_instructions = """
- Look for financial figures associated with the highest revenue, turnover, or sales product/service
- Extract the exact amount including the unit (e.g., "Rs. 10,000,000" or "₹ 10 crores")
- Pay attention to whether amounts are in thousands, lakhs, crores, or millions
- Check product-wise breakdowns in financial statements or reports
"""
            few_shot_examples = """
Example 1:
Context: "...Cotton yarn (code 52051110) contributed Rs. 275.36 crores, being our top-selling product..."
Question: Turnover of highest contributing product or service (in Rupees)
Answer: Rs. 275.36 crores

Example 2:
Context: "...Revenue distribution by product: Enterprise Software (99831326): ₹ 1,452.6 lakhs (highest), Mobile Apps: ₹ 904.2 lakhs..."
Question: Turnover of highest contributing product or service (in Rupees)
Answer: ₹ 1,452.6 lakhs
"""
        elif "description" in question.lower() and "category" in question.lower():
            specific_instructions = """
- Extract the exact description of the product or service category as stated in the document
- Look in business segment reporting, notes to accounts, or product sections
- The description should match the associated 4-digit code mentioned elsewhere
"""
            few_shot_examples = """
Example 1:
Context: "...The company operates in Textile Manufacturing (ITC code 5205) which includes production of cotton and synthetic fabrics..."
Question: Description of the product or service category
Answer: Textile Manufacturing

Example 2:
Context: "...Business segments: Software Development Services (NIC code 6201) - Includes custom application development..."
Question: Description of the product or service category
Answer: Software Development Services
"""
        elif "description" in question.lower() and "service" in question.lower():
            specific_instructions = """
- Extract the exact description of the specific product or service as stated in the document
- Look for descriptions associated with the highest revenue product or 8-digit code
- Check product listings, catalog descriptions, or sales breakdowns
"""
            few_shot_examples = """
Example 1:
Context: "...Our premium combed cotton yarn (code 52051110) remains our flagship product, contributing the highest to our revenue..."
Question: Description of the product or service
Answer: Premium combed cotton yarn

Example 2:
Context: "...Enterprise Resource Planning Software (99831326) continues to be our leading service offering with highest sales..."
Question: Description of the product or service
Answer: Enterprise Resource Planning Software
"""
        
        # Create prompt for Llama with specific format requirements and few-shot examples
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_message}

You will be provided with a question and context from a PDF document. Your task is to extract the precise answer to the question from the context.

{specific_instructions}
Follow these general guidelines:
- If the answer is explicitly stated in the text, provide that exact answer including any associated codes or figures
- For financial figures, maintain the exact format (e.g., "Rs. 10,00,000" or "₹ 10 crores")
- If the answer requires combining information from different parts of the text, do so accurately
- If the answer is not in the context, respond with "Information not found in the provided context"
- Do not provide explanations, just the direct answer to the question

{few_shot_examples}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Context from PDF:
{context}

Answer the question based only on the information in the context.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        
        return prompt
    
    def _post_process_answer(self, question: str, answer: str) -> str:
        """Post-process the answer to format it consistently"""
        # Remove any explanatory text or prefixes
        cleaned = re.sub(r'^(Answer:|The answer is:|Based on the context,)', '', answer).strip()
        
        # Handle "not found" responses consistently
        if re.search(r'(not found|not provided|not mentioned|couldn\'t find|could not find|no information)', 
                     cleaned, re.IGNORECASE):
            return "Information not found in the provided context"
        
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
        
        # Format financial figures consistently
        if "turnover" in question.lower() or "in Rupees" in question:
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
        
        # Return the cleaned answer for other questions
        return cleaned

class PDFRegulatoryExtractor:
    """Main class for extracting regulatory information from PDFs"""
    def __init__(self, config_path: str = "config1.json"):
        self.config_path = config_path
        self.llm_generator = LLMGenerator()
        self.pdf_extractor = PDFExtractor()
        self.qa_processor = QuestionAnswerProcessor(self.llm_generator)
        
        # Define the specific regulatory questions to extract
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
        
        # Define keywords for each question to help locate relevant information
        self.question_keywords = {
            "Corporate identity number (CIN) of company": ["CIN", "corporate identity number", "company registration", "L", "U", "registration number"],
            "Financial year to which financial statements relates": ["financial year", "FY", "year ended", "period ended", "statement period", "fiscal year"],
            "Product or service category code (ITC/ NPCS 4 digit code)": ["ITC code", "NPCS code", "product code", "service code", "4 digit", "category code"],
            "Description of the product or service category": ["product category", "service category", "business segment", "category description", "main business"],
            "Turnover of the product or service category (in Rupees)": ["category turnover", "segment turnover", "business turnover", "revenue from", "sales from", "income from"],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": ["8 digit", "product code", "service code", "highest turnover", "main product", "major service"],
            "Description of the product or service": ["product description", "service description", "main offering", "primary product", "key service"],
            "Turnover of highest contributing product or service (in Rupees)": ["product turnover", "service turnover", "product revenue", "service revenue", "contributing", "highest sales"]
        }
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a PDF file to extract regulatory information and answer all questions"""
        print(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract regular text from PDF
            pages = self.pdf_extractor.extract_text_from_pdf(pdf_path)
            
            # Extract potential tables and structured content
            tables = self.pdf_extractor.extract_tables_from_pdf(pdf_path)
            
            # Add table text to pages for combined processing
            for table in tables:
                # Add with special page numbers to distinguish them
                pages.append({
                    "page_num": 1000 + table["page_num"],  # 1000+ indicates table content
                    "text": f"TABLE/STRUCTURED CONTENT FROM PAGE {table['page_num']}:\n{table['table_text']}"
                })
            
            # Create a dictionary to store answers
            answers = {}
            
            # Process each question
            for question in self.questions:
                # Get keywords for this question
                keywords = self.question_keywords.get(question, [])
                
                # Search for pages containing these keywords - now passing the full question
                relevant_page_nums = self.pdf_extractor.search_keywords_in_pdf(pages, keywords, question)
                
                # Always include table content for specific questions
                if len(relevant_page_nums) == 0 and any(term in question.lower() for term in ["code", "turnover", "category", "service", "CIN"]):
                    table_page_nums = [p["page_num"] for p in pages if p["page_num"] >= 1000]
                    relevant_page_nums.extend(table_page_nums)
                
                # If no relevant pages found, use fallback strategy
                if not relevant_page_nums:
                    print(f"No relevant pages found for question: {question}")
                    
                    # Include first few pages as fallback for CIN and financial year
                    if "CIN" in question or "financial year" in question.lower():
                        relevant_page_nums = list(range(1, min(10, len(pages) + 1)))
                    else:
                        # For other questions, include all pages with table-like content
                        table_page_nums = [p["page_num"] for p in pages if p["page_num"] >= 1000]
                        if table_page_nums:
                            relevant_page_nums.extend(table_page_nums)
                        else:
                            # Last resort: include first few pages
                            relevant_page_nums = list(range(1, min(5, len(pages) + 1)))
                
                # Sort page numbers for readability
                relevant_page_nums = sorted(list(set(relevant_page_nums)))
                
                # Extract text from relevant pages
                context = ""
                for page_num in relevant_page_nums:
                    # Find the right page
                    page_indices = [i for i, p in enumerate(pages) if p["page_num"] == page_num]
                    if page_indices:
                        # Format table content differently
                        if page_num >= 1000:
                            context += f"\n--- TABLE CONTENT (Page {page_num - 1000}) ---\n"
                        else:
                            context += f"\n--- Page {page_num} ---\n"
                        context += pages[page_indices[0]]["text"]
                
                # Truncate context if too long (based on Llama's context window)
                max_context_length = 12000
                if len(context) > max_context_length:
                    context = context[:max_context_length]
                
                # Generate answer with post-processing for consistent formatting
                answer = self.qa_processor.generate_answer(question, context)
                
                # Store answer
                answers[question] = answer
            
            print(f"Completed processing of {pdf_path}")
            return answers
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            # Return error message for all questions
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch(self, pdf_dir: str, output_excel: str):
        """Process multiple PDF files and save results to an Excel file"""
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
        
        # Determine number of workers
        num_workers = min(os.cpu_count() or 4, 4)  # Use at most 4 workers
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all PDF processing tasks
            future_to_pdf = {
                executor.submit(self.process_pdf, os.path.join(pdf_dir, pdf_file)): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Process results as they complete
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    # Get answers for this PDF
                    answers = future.result()
                    
                    # Add filename to results
                    result = {"PDF Filename": pdf_file}
                    result.update(answers)
                    
                    results.append(result)
                    print(f"Added results for {pdf_file}")
                    
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
            
            # Reorder columns to have PDF Filename first, then questions
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
        else:
            print("No results to save")

def main():
    # Simple command line argument handling
    import argparse
    parser = argparse.ArgumentParser(description="PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, default="pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_results.xlsx", help="Output Excel file")
    parser.add_argument("--config", type=str, default="config1.json", help="Configuration file")
    args = parser.parse_args()
    
    try:
        print("Starting PDF Regulatory Information Extraction")
        
        # Create and run the pipeline
        pipeline = PDFRegulatoryExtractor(config_path=args.config)
        pipeline.process_pdfs_batch(args.pdf_dir, args.output)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
