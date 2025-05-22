import os
import re
import time
import pandas as pd
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

class RuleBasedExtractor:
    """Rule-based extractor for consistent PDF forms"""
    
    def __init__(self):
        # Define extraction patterns for each question
        self.patterns = {
            "Corporate identity number (CIN) of company": [
                r'(?:CIN|Corporate Identity Number)[:\s]*([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
                r'([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})',
                r'(?:Registration No|Reg\. No)[:\s]*([LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6})'
            ],
            "Financial year to which financial statements relates": [
                r'(?:Financial Year|FY|Year ended|Period ended)[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'(?:for the year ended|year ending)[:\s]*([0-9]{4}[-/][0-9]{2,4})',
                r'([0-9]{4}[-/][0-9]{2,4})',
                r'(?:FY|F\.Y\.)[:\s]*([0-9]{4}[-/][0-9]{2,4})'
            ],
            "Product or service category code (ITC/ NPCS 4 digit code)": [
                r'(?:ITC|NPCS)[:\s]*([0-9]{4})',
                r'(?:Product Code|Service Code)[:\s]*([0-9]{4})',
                r'(?:Category Code)[:\s]*([0-9]{4})',
                r'([0-9]{4})'  # Fallback for any 4-digit number
            ],
            "Description of the product or service category": [
                r'(?:Product Category|Service Category|Description)[:\s]*([A-Za-z\s,&-]+)',
                r'(?:Business Segment|Primary Business)[:\s]*([A-Za-z\s,&-]+)',
                r'(?:Nature of Business)[:\s]*([A-Za-z\s,&-]+)'
            ],
            "Turnover of the product or service category (in Rupees)": [
                r'(?:Category Turnover|Segment Turnover)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+\.?[0-9]*)',
                r'(?:Revenue)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+\.?[0-9]*)',
                r'(?:Turnover)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+\.?[0-9]*)'
            ],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": [
                r'(?:Highest.*?ITC|Main.*?ITC|Primary.*?ITC)[:\s]*([0-9]{8})',
                r'(?:ITC|NPCS)[:\s]*([0-9]{8})',
                r'(?:Product Code|Service Code)[:\s]*([0-9]{8})',
                r'([0-9]{8})'  # Fallback for any 8-digit number
            ],
            "Description of the product or service": [
                r'(?:Product Description|Service Description|Main Product)[:\s]*([A-Za-z\s,&.-]+)',
                r'(?:Primary Product|Main Service)[:\s]*([A-Za-z\s,&.-]+)',
                r'(?:Description)[:\s]*([A-Za-z\s,&.-]+)'
            ],
            "Turnover of highest contributing product or service (in Rupees)": [
                r'(?:Highest.*?Turnover|Main.*?Turnover|Primary.*?Turnover)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+\.?[0-9]*)',
                r'(?:Revenue from operations|Sale of products)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+\.?[0-9]*)',
                r'(?:Product sales|Primary revenue)[:\s]*(?:Rs\.?|₹|INR)?\s*([0-9,]+\.?[0-9]*)',
                # Look for tables with revenue/turnover data
                r'([0-9,]+\.?[0-9]*)',  # Generic number pattern as fallback
            ]
        }
        
        # Define which pages to search for each question
        self.page_mapping = {
            "Corporate identity number (CIN) of company": [1],
            "Financial year to which financial statements relates": [1],
            "Product or service category code (ITC/ NPCS 4 digit code)": [10, 9, 11, 8, 12],
            "Description of the product or service category": [10, 9, 11, 8, 12],
            "Turnover of the product or service category (in Rupees)": [10, 9, 11, 8, 12],
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)": [10, 9, 11, 8, 12],
            "Description of the product or service": [10, 9, 11, 8, 12],
            "Turnover of highest contributing product or service (in Rupees)": [10, 9, 11, 8, 12]
        }
    
    def extract_with_patterns(self, text: str, patterns: List[str], question: str) -> Optional[str]:
        """Extract information using regex patterns"""
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Return the first match, cleaned up
                    result = matches[0].strip()
                    
                    # Post-process based on question type
                    if "CIN" in question:
                        # Validate CIN format
                        if re.match(r'^[LU][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}$', result):
                            return result
                    
                    elif "financial year" in question.lower():
                        # Clean up financial year format
                        result = re.sub(r'[^\d\-/]', '', result)
                        if len(result) >= 7:  # At least YYYY-YY
                            return result
                    
                    elif "4 digit code" in question.lower():
                        # Ensure it's exactly 4 digits
                        if re.match(r'^\d{4}$', result):
                            return result
                    
                    elif "8 digit code" in question.lower():
                        # Ensure it's exactly 8 digits
                        if re.match(r'^\d{8}$', result):
                            return result
                    
                    elif "turnover" in question.lower() and "Rupees" in question:
                        # Clean up numeric values
                        clean_number = re.sub(r'[^\d.,]', '', result)
                        clean_number = clean_number.replace(',', '')
                        if clean_number and clean_number.replace('.', '').isdigit():
                            return clean_number
                    
                    elif "description" in question.lower():
                        # Clean up description text
                        result = re.sub(r'[^\w\s,&.-]', '', result)
                        if len(result.strip()) > 3:  # Ensure meaningful description
                            return result.strip()
                    
                    else:
                        return result
                        
            except Exception as e:
                continue
        
        return None
    
    def extract_from_table(self, text: str, question: str) -> Optional[str]:
        """Extract information from table structures"""
        if "turnover" in question.lower() and "highest" in question.lower():
            # Look for table patterns with revenue/turnover data
            lines = text.split('\n')
            
            # Look for lines that might contain revenue data
            for i, line in enumerate(lines):
                if re.search(r'(?:revenue|turnover|sales)', line, re.IGNORECASE):
                    # Check surrounding lines for numbers
                    for j in range(max(0, i-2), min(len(lines), i+3)):
                        numbers = re.findall(r'\b(\d{6,})\b', lines[j])
                        if numbers:
                            return numbers[0]  # Return the largest looking number
            
            # Fallback: look for any large numbers that might be revenue
            all_numbers = re.findall(r'\b(\d{6,})\b', text)
            if all_numbers:
                # Return the largest number found (likely to be revenue)
                return max(all_numbers, key=lambda x: int(x))
        
        return None

class PDFRegulatoryExtractorFast:
    """Fast rule-based PDF extractor for consistent forms"""
    
    def __init__(self):
        self.extractor = RuleBasedExtractor()
        
        # Define the questions
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
    
    def extract_text_from_specific_pages(self, pdf_path: str, page_numbers: List[int]) -> str:
        """Extract text from specific pages only"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in page_numbers:
                if page_num <= len(doc):
                    page = doc[page_num - 1]  # fitz uses 0-based indexing
                    page_text = page.get_text()
                    text += f"\n--- Page {page_num} ---\n" + page_text
            
            doc.close()
            return text
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Process a single PDF file"""
        start_time = time.time()
        print(f"Processing PDF: {os.path.basename(pdf_path)}")
        
        results = {}
        
        try:
            # Process each question
            for question in self.questions:
                # Get the pages to search for this question
                pages_to_search = self.extractor.page_mapping.get(question, [1])
                
                # Extract text from relevant pages only
                text = self.extract_text_from_specific_pages(pdf_path, pages_to_search)
                
                if not text:
                    results[question] = "Error: Could not extract text from PDF"
                    continue
                
                # Try to extract using patterns
                patterns = self.extractor.patterns.get(question, [])
                extracted_value = self.extractor.extract_with_patterns(text, patterns, question)
                
                # If pattern matching failed, try table extraction for turnover questions
                if not extracted_value and "turnover" in question.lower():
                    extracted_value = self.extractor.extract_from_table(text, question)
                
                # Store result
                if extracted_value:
                    results[question] = extracted_value
                else:
                    results[question] = "Information not found"
            
            print(f"Processed {os.path.basename(pdf_path)} in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {question: f"Error: {str(e)}" for question in self.questions}
    
    def process_pdfs_batch_parallel(self, pdf_dir: str, output_excel: str, max_workers: int = 4):
        """Process multiple PDFs in parallel"""
        batch_start_time = time.time()
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
        print(f"Using {max_workers} parallel workers")
        
        results = []
        
        # Process PDFs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PDF processing tasks
            future_to_pdf = {
                executor.submit(self.process_pdf, os.path.join(pdf_dir, pdf_file)): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_pdf):
                pdf_file = future_to_pdf[future]
                try:
                    pdf_results = future.result()
                    
                    # Add filename to results
                    result = {"PDF Filename": pdf_file}
                    result.update(pdf_results)
                    results.append(result)
                    
                    print(f"Completed: {pdf_file}")
                    
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")
                    # Add error entry
                    result = {"PDF Filename": pdf_file}
                    result.update({question: f"Error: {str(e)}" for question in self.questions})
                    results.append(result)
        
        # Save results to Excel
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Reorder columns
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")
            
            # Print summary statistics
            print(f"\nSummary:")
            print(f"Total PDFs processed: {len(results)}")
            
            for question in self.questions:
                found_count = sum(1 for r in results if r[question] not in ["Information not found", "Error: Could not extract text from PDF"])
                print(f"{question[:50]}...: {found_count}/{len(results)} found")
        
        else:
            print("No results to save")
        
        total_time = time.time() - batch_start_time
        print(f"\nTotal processing completed in {total_time:.2f} seconds")
        print(f"Average time per PDF: {total_time/len(pdf_files):.2f} seconds")
    
    def process_pdfs_batch_serial(self, pdf_dir: str, output_excel: str):
        """Process PDFs one by one (for debugging)"""
        batch_start_time = time.time()
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
        
        results = []
        
        # Process each PDF
        for i, pdf_file in enumerate(pdf_files, 1):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"\nProcessing {i}/{len(pdf_files)}: {pdf_file}")
            
            try:
                pdf_results = self.process_pdf(pdf_path)
                
                # Add filename to results
                result = {"PDF Filename": pdf_file}
                result.update(pdf_results)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                # Add error entry
                result = {"PDF Filename": pdf_file}
                result.update({question: f"Error: {str(e)}" for question in self.questions})
                results.append(result)
        
        # Save results to Excel
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Reorder columns
            columns = ["PDF Filename"] + self.questions
            df = df[columns]
            
            # Save to Excel
            df.to_excel(output_excel, index=False)
            print(f"\nResults saved to {output_excel}")
            
            # Print summary statistics
            print(f"\nSummary:")
            print(f"Total PDFs processed: {len(results)}")
            
            for question in self.questions:
                found_count = sum(1 for r in results if r[question] not in ["Information not found", "Error: Could not extract text from PDF"])
                print(f"{question[:50]}...: {found_count}/{len(results)} found")
        
        else:
            print("No results to save")
        
        total_time = time.time() - batch_start_time
        print(f"\nTotal processing completed in {total_time:.2f} seconds")
        print(f"Average time per PDF: {total_time/len(pdf_files):.2f} seconds")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fast Rule-Based PDF Regulatory Information Extraction")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="regulatory_info_fast_results.xlsx", help="Output Excel file")
    parser.add_argument("--single_pdf", type=str, default=None, help="Process a single PDF instead of a directory")
    parser.add_argument("--parallel", action="store_true", help="Use parallel processing (default: serial)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()
    
    try:
        print("Starting Fast PDF Regulatory Information Extraction")
        start_time = time.time()
        
        # Create pipeline
        pipeline = PDFRegulatoryExtractorFast()
        
        # Process single PDF or directory
        if args.single_pdf:
            if not os.path.isfile(args.single_pdf):
                print(f"PDF file not found: {args.single_pdf}")
                return
            
            # Process single PDF
            pdf_file = os.path.basename(args.single_pdf)
            
            try:
                # Process the PDF
                answers = pipeline.process_pdf(args.single_pdf)
                
                # Create results
                results = [{"PDF Filename": pdf_file, **answers}]
                
                # Save to Excel
                df = pd.DataFrame(results)
                columns = ["PDF Filename"] + pipeline.questions
                df = df[columns]
                df.to_excel(args.output, index=False)
                print(f"Results saved to {args.output}")
                
                # Print results
                print("\nExtracted Information:")
                for question, answer in answers.items():
                    print(f"{question}: {answer}")
                
            except Exception as e:
                print(f"Error processing {args.single_pdf}: {e}")
        
        else:
            # Process all PDFs in directory
            if args.parallel:
                pipeline.process_pdfs_batch_parallel(args.pdf_dir, args.output, args.workers)
            else:
                pipeline.process_pdfs_batch_serial(args.pdf_dir, args.output)
        
        print(f"\nProcessing completed successfully in {time.time() - start_time:.2f} seconds!")
        
    except Exception as e:
        print(f"Pipeline error: {e}")

if __name__ == "__main__":
    main()
