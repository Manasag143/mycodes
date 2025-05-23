import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json

# Import Docling components
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

class DoclingPDFExtractor:
    """Basic PDF data extractor using Docling"""
    
    def __init__(self):
        # Configure pipeline options for better PDF processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned documents
        pipeline_options.do_table_structure = True  # Extract table structures
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Initialize document converter
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pipeline_options,
            }
        )
        
        # Define fields we want to extract
        self.target_fields = [
            "Corporate identity number (CIN) of company",
            "Financial year to which financial statements relates", 
            "Product or service category code (ITC/ NPCS 4 digit code)",
            "Description of the product or service category",
            "Turnover of the product or service category (in Rupees)",
            "Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)",
            "Description of the product or service",
            "Turnover of highest contributing product or service (in Rupees)"
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured data from a single PDF"""
        print(f"Processing: {pdf_path}")
        
        try:
            # Convert PDF to Docling document
            result = self.converter.convert(pdf_path)
            doc = result.document
            
            # Get document structure
            print(f"Document has {len(doc.pages)} pages")
            
            # Extract different types of content
            extraction_result = {
                'text_content': self._extract_text_content(doc),
                'tables': self._extract_tables(doc),
                'structured_data': self._extract_structured_data(doc),
                'form_fields': self._extract_form_fields(doc)
            }
            
            return extraction_result
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {'error': str(e)}
    
    def _extract_text_content(self, doc) -> Dict[str, str]:
        """Extract text content from the document"""
        text_data = {}
        
        # Get full document text
        full_text = doc.export_to_markdown()
        text_data['full_document'] = full_text
        
        # Extract text from specific pages (1 and 10)
        for page_num in [1, 10]:
            if page_num <= len(doc.pages):
                page = doc.pages[page_num - 1]
                page_text = ""
                
                # Extract text from all elements on the page
                for element in page.body.elements:
                    if hasattr(element, 'text'):
                        page_text += element.text + "\n"
                
                text_data[f'page_{page_num}'] = page_text
        
        return text_data
    
    def _extract_tables(self, doc) -> List[Dict]:
        """Extract table data from the document"""
        tables_data = []
        
        for page_num, page in enumerate(doc.pages, 1):
            for element in page.body.elements:
                if element.category == "table":
                    try:
                        # Convert table to pandas DataFrame
                        table_df = element.export_to_dataframe()
                        
                        table_info = {
                            'page': page_num,
                            'table_data': table_df.to_dict('records'),
                            'columns': list(table_df.columns),
                            'shape': table_df.shape
                        }
                        tables_data.append(table_info)
                        
                    except Exception as e:
                        print(f"Error extracting table on page {page_num}: {e}")
        
        return tables_data
    
    def _extract_structured_data(self, doc) -> Dict[str, Any]:
        """Extract structured data elements"""
        structured = {
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'key_value_pairs': []
        }
        
        for page_num, page in enumerate(doc.pages, 1):
            for element in page.body.elements:
                element_data = {
                    'page': page_num,
                    'text': getattr(element, 'text', ''),
                    'category': element.category
                }
                
                if element.category == "title" or element.category == "section-header":
                    structured['headings'].append(element_data)
                elif element.category == "paragraph":
                    structured['paragraphs'].append(element_data)
                elif element.category == "list":
                    structured['lists'].append(element_data)
        
        return structured
    
    def _extract_form_fields(self, doc) -> Dict[str, str]:
        """Extract form-like data by looking for key-value patterns"""
        form_data = {}
        
        # Get all text content
        all_text = ""
        for page in doc.pages:
            for element in page.body.elements:
                if hasattr(element, 'text'):
                    all_text += element.text + "\n"
        
        # Look for our target fields using simple pattern matching
        form_data.update(self._extract_target_fields(all_text))
        
        return form_data
    
    def _extract_target_fields(self, text: str) -> Dict[str, str]:
        """Extract specific target fields from text"""
        import re
        
        results = {}
        
        # CIN extraction
        cin_match = re.search(r'([LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})', text, re.IGNORECASE)
        if cin_match:
            results["Corporate identity number (CIN) of company"] = cin_match.group(1).upper()
        else:
            results["Corporate identity number (CIN) of company"] = "Not found"
        
        # Financial year extraction
        fy_match = re.search(r'(\d{4}[-/]\d{2,4})', text)
        if fy_match:
            results["Financial year to which financial statements relates"] = fy_match.group(1)
        else:
            results["Financial year to which financial statements relates"] = "Not found"
        
        # 4-digit code extraction
        four_digit_codes = re.findall(r'\b(\d{4})\b', text)
        # Filter out years
        filtered_codes = [code for code in four_digit_codes if not (1900 <= int(code) <= 2030)]
        if filtered_codes:
            results["Product or service category code (ITC/ NPCS 4 digit code)"] = filtered_codes[0]
        else:
            results["Product or service category code (ITC/ NPCS 4 digit code)"] = "Not found"
        
        # 8-digit code extraction
        eight_digit_codes = re.findall(r'\b(\d{8})\b', text)
        if eight_digit_codes:
            results["Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)"] = eight_digit_codes[0]
        else:
            results["Highest turnover contributing product or service code (ITC/ NPCS 8 digit code)"] = "Not found"
        
        # Turnover extraction (find largest numbers)
        large_numbers = re.findall(r'\b(\d{6,})\b', text)
        if large_numbers:
            # Convert to integers and find the largest
            numbers = [int(num) for num in large_numbers]
            largest = max(numbers)
            results["Turnover of highest contributing product or service (in Rupees)"] = str(largest)
            
            # For category turnover, use a smaller number if available
            if len(numbers) > 1:
                numbers.sort()
                results["Turnover of the product or service category (in Rupees)"] = str(numbers[-2])
            else:
                results["Turnover of the product or service category (in Rupees)"] = str(largest)
        else:
            results["Turnover of highest contributing product or service (in Rupees)"] = "Not found"
            results["Turnover of the product or service category (in Rupees)"] = "Not found"
        
        # Description extraction (simplified)
        results["Description of the product or service category"] = "Text analysis needed"
        results["Description of the product or service"] = "Text analysis needed"
        
        return results
    
    def save_detailed_analysis(self, pdf_path: str, output_dir: str = "docling_output"):
        """Save detailed analysis of PDF structure"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract all data
        extraction_result = self.extract_from_pdf(pdf_path)
        
        # Save raw extraction to JSON
        json_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_raw_extraction.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_result, f, indent=2, ensure_ascii=False, default=str)
        
        # Save text content to file
        if 'text_content' in extraction_result:
            text_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_text_content.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                for key, content in extraction_result['text_content'].items():
                    f.write(f"\n{'='*50}\n")
                    f.write(f"{key.upper()}\n")
                    f.write(f"{'='*50}\n")
                    f.write(content)
                    f.write("\n")
        
        # Save tables to Excel if any
        if 'tables' in extraction_result and extraction_result['tables']:
            excel_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_tables.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for i, table in enumerate(extraction_result['tables']):
                    df = pd.DataFrame(table['table_data'])
                    sheet_name = f"Table_Page_{table['page']}_{i+1}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Detailed analysis saved to {output_dir}")
        return extraction_result
    
    def process_multiple_pdfs(self, pdf_directory: str, output_excel: str):
        """Process multiple PDFs and save results to Excel"""
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_directory}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        all_results = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            
            try:
                # Extract data
                extraction_result = self.extract_from_pdf(pdf_path)
                
                # Get form fields
                if 'form_fields' in extraction_result:
                    result_row = {"PDF Filename": pdf_file}
                    result_row.update(extraction_result['form_fields'])
                    all_results.append(result_row)
                else:
                    # Create error row
                    result_row = {"PDF Filename": pdf_file}
                    for field in self.target_fields:
                        result_row[field] = "Extraction failed"
                    all_results.append(result_row)
                
                print(f"Processed: {pdf_file}")
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                # Create error row
                result_row = {"PDF Filename": pdf_file}
                for field in self.target_fields:
                    result_row[field] = f"Error: {str(e)}"
                all_results.append(result_row)
        
        # Save to Excel
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_excel(output_excel, index=False)
            print(f"Results saved to {output_excel}")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docling PDF Data Extractor")
    parser.add_argument("--pdf_path", type=str, help="Single PDF file to process")
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDF files")
    parser.add_argument("--output", type=str, default="docling_results.xlsx", help="Output Excel file")
    parser.add_argument("--analyze", action="store_true", help="Save detailed analysis")
    parser.add_argument("--output_dir", type=str, default="docling_output", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = DoclingPDFExtractor()
    
    if args.pdf_path:
        # Process single PDF
        if args.analyze:
            # Save detailed analysis
            extraction_result = extractor.save_detailed_analysis(args.pdf_path, args.output_dir)
            print("Detailed analysis complete!")
        else:
            # Simple extraction
            result = extractor.extract_from_pdf(args.pdf_path)
            
            # Save form fields to Excel
            if 'form_fields' in result:
                df = pd.DataFrame([{"PDF Filename": Path(args.pdf_path).name, **result['form_fields']}])
                df.to_excel(args.output, index=False)
                print(f"Results saved to {args.output}")
    
    elif args.pdf_dir:
        # Process multiple PDFs
        extractor.process_multiple_pdfs(args.pdf_dir, args.output)
    
    else:
        print("Please specify either --pdf_path or --pdf_dir")

if __name__ == "__main__":
    main()

# Example usage:
"""
# Install Docling first:
# pip install docling

# Process single PDF with detailed analysis:
# python docling_extractor.py --pdf_path "sample.pdf" --analyze --output_dir "analysis_output"

# Process single PDF for data extraction:
# python docling_extractor.py --pdf_path "sample.pdf" --output "results.xlsx"

# Process multiple PDFs:
# python docling_extractor.py --pdf_dir "path/to/pdfs" --output "batch_results.xlsx"
"""
