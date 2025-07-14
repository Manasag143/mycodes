import os
import time
import pandas as pd
import fitz  # PyMuPDF
import requests
import warnings
import logging
import json
import ast
from datetime import datetime

# Suppress SSL warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HostedLLM:
    """Custom LLM class for hosted Llama model"""
   
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
   
    def _call(self, prompt: str) -> str:
        """Make API call to hosted LLM"""
        try:
            prompt_template = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
           
            payload = json.dumps({
                "provider": "tgi",
                "deployment": "Llama 3.3 v1",
                "spec_version": 1,
                "input_text": prompt_template,
                "params": {"temperature": 0.1}
            })
           
            headers = {
                'token': '0e53d2cf9f724a94a6ca0a9c880fdee7',
                'Content-Type': 'application/json'
            }
           
            response = requests.post(
                url="https://llmgateway.crisil.local/api/v1/llm",
                headers=headers,
                data=payload,
                verify=False
            )
           
            if response.status_code != 200:
                return f"LLM Call Failed: HTTP {response.status_code} - {response.text}"
           
            response_v = ast.literal_eval(response.text)
            resp_o = response_v['output']
            output = str(resp_o).replace(prompt_template, "")
            return output.strip()
           
        except Exception as e:
            return f"LLM Call Failed - Error: {e}"

class PDFExtractor:
    """Class for extracting text from PDF files"""
   
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
           
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
           
            doc.close()
            return all_text
           
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise

class FinancialRedFlagAnalyzer:
    """Main pipeline class for financial red flag analysis"""
   
    def __init__(self, llm_endpoint: str = "https://llmgateway.crisil.local/api/v1/llm"):
        """Initialize the analyzer"""
        self.llm = HostedLLM(endpoint=llm_endpoint)
        self.pdf_extractor = PDFExtractor()
        
        # Financial analyst prompt with all keywords
        self.analyst_prompt = """You are an experienced financial analyst tasked with analyzing Earnings call transcripts for a company and identify potential causes for concern: red-flags.

INSTRUCTIONS:
1. Search for all keywords mentioned in the reference section in the document
2. Do not miss any keyword mentioned in reference
3. While searching the keywords keep in mind their definitions as well which are mentioned in reference section
4. A keyword will appear more than once in the document and you need to look at each instance and flag each occurrence as a new point only if the context is different. Do not duplicate the occurrence of the same context in terms of the keyword
5. If multiple keywords occur in the same context, do not generate multiple outputs for them instead identify them as keyword1/keyword2/keyword3/keyword4 etc
6. You will highlight a keyword only if you find it in the document in a negative cause for concern as a red flag, do not highlight positive flags
7. The output should be crisp and clear, avoid duplication of any keyword for the same context

For each red flag:
OUTPUT FORMAT:
1. The potential red flag you observed - the actual key word you found in the document
2. For the same red flag state the original quote or text that caused you to see the red flag. Also mention the page number where this statement was present

REFERENCE KEYWORDS:
1. Attrition: Refers to the increasing or high loss of employees, customers, or revenue due to various reasons such as resignation, retirement, or competition, which can negatively impact a company's financial performance.
2. Adverse: Describes an unfavourable or negative situation, event, or trend that can impact a company's financial performance, such as adverse market conditions or regulatory changes.
3. Cautious outlook: Indicates a company's conservative or pessimistic view of its future financial performance, often due to uncertainty or potential risks.
4. Challenging environment: Refers to a difficult or competitive market situation that can impact a company's financial performance, such as intense competition, regulatory challenges, or economic downturns.
5. Competition intensifying: Describes an increase in competition in a market or industry, which can lead to decreased market share, revenue, or profitability for a company.
6. Corporate governance: Refers to the system of rules, practices, and processes by which a company is directed and controlled, including issues related to board composition, executive compensation, and audit committee independence.
7. Cost inflation: Describes an increase in costs, such as labor, materials, or overheads, which can negatively impact a company's profitability.
8. Customer confidence: Refers to the level of trust and faith that customers have in a company's products or services, which can impact sales and revenue.
9. Debt reduction is lower than expected: Indicates that a company's efforts to reduce its debt have not been as successful as anticipated, which can impact its financial flexibility and credit rating.
10. Debt repayment challenges: Describes difficulties a company faces in repaying its debt, which can lead to default, restructuring, or other negative consequences.
11. Decelerate: Refers to a slowdown in a company's growth rate, which can be due to various factors such as market saturation, competition, or economic downturns.
12. Decline: Describes a decrease in a company's financial performance, such as revenue, profitability, or market share.
13. Delay: Refers to a postponement or deferral of a project, investment, or other business initiative, which can impact a company's financial performance.
14. Difficulties: Describes challenges or problems a company faces in its operations, such as supply chain disruptions, regulatory issues, or talent acquisition.
15. Disappointing results: Refers to financial performance that falls short of expectations, which can lead to a decline in stock price, investor confidence, or credit rating.
16. Elusive: Refers to a goal, target, or objective that is difficult to achieve or maintain, such as profitability, market share, or growth.
17. Group company exposure: Describes a company's financial exposure to its subsidiaries, affiliates, or joint ventures, which can impact its consolidated financial performance.
18. Guidance revision: Refers to a change in a company's financial guidance, such as revenue or earnings estimates, which can impact investor expectations and stock price.
19. Impairment charges: Refers to non-cash charges taken by a company to reflect the decline in value of its assets, such as goodwill, property, or equipment.
20. Increase provisions: Describes an increase in a company's provisions for bad debts, warranties, or other contingent liabilities, which can impact its profitability.
21. Increasing working capital: Describes an increase in a company's working capital requirements, such as accounts receivable, inventory, or accounts payable, which can impact its liquidity and cash flow.
22. Inventory levels gone up: Refers to an increase in a company's inventory levels, which can indicate slower sales, overproduction, or supply chain disruptions.
23. Liquidity concerns: Describes a company's difficulties in meeting its short-term financial obligations, such as paying debts or meeting working capital requirements.
24. Lost market share: Refers to a decline in a company's market share, which can be due to various factors such as competition, pricing, or product quality.
25. Management exits: Describes the departure of key executives or managers from a company, which can impact its leadership, strategy, and financial performance.
26. Margin pressure: Describes a decline in a company's profit margins, which can be due to various factors such as competition, pricing pressure, or cost inflation.
27. New management: Refers to the appointment of new executives or managers to a company's leadership team, which can impact its strategy, culture, and financial performance.
28. No confidence: Describes a lack of trust or faith in a company's management, strategy, or financial performance, which can impact investor confidence and stock price.
29. One-off expenses: Refers to non-recurring expenses or charges taken by a company, such as restructuring costs, impairment charges, or litigation expenses.
30. One-time write-offs: Refers to non-recurring write-offs or charges taken by a company, such as asset impairments, inventory write-offs, or accounts receivable write-offs.
31. Operational issues: Describes challenges or problems a company faces in its operations, such as supply chain disruptions, quality control issues, or talent acquisition.
32. Regulatory uncertainty: Describes uncertainty or ambiguity related to regulatory requirements, laws, or policies that can impact a company's operations, financial performance, or compliance.
33. Related party transaction: Refers to a transaction between a company and its related parties, such as subsidiaries, affiliates, or joint ventures, which can impact its financial performance and transparency.
34. Restructuring efforts: Refers to a company's plans or actions to reorganize its operations, finances, or management structure to improve its performance, efficiency, or competitiveness.
35. Scale down: Describes a company's decision to reduce its operations, investments, or workforce to conserve resources, cut costs, or adapt to changing market conditions.
36. Service issue: Refers to problems or difficulties a company faces in delivering its products or services, which can impact customer satisfaction, revenue, or reputation.
37. Shortage: Describes a situation where a company faces a lack of supply, resources, or personnel, which can impact its operations, production, or delivery of products or services.
38. Stress: Refers to a company's financial difficulties or challenges, such as debt, cash flow problems, or operational issues, which can impact its ability to meet its financial obligations.
39. Supply chain disruptions: Refers to interruptions or problems in a company's supply chain, which can impact its ability to produce, deliver, or distribute its products or services.
40. Warranty cost: Refers to the expenses or provisions a company makes for warranties or guarantees provided to its customers, which can impact its profitability or cash flow.
41. Breach: Describes a company's failure to comply with laws, regulations, or contractual obligations, which can impact its reputation, financial performance, or relationships with stakeholders.
42. Misappropriation of funds: Describes the unauthorized or improper use of a company's funds, assets, or resources, which can impact its financial performance, reputation, or relationships with stakeholders.
43. Increase in borrowing cost: Refers to a rise in the cost of borrowing for a company, which can impact its interest expenses, cash flow, or financial flexibility.
44. One time reversal: Describes a non-recurring or one-time adjustment to a company's financial statements, which can impact its profitability, revenue, or expenses.
45. Bloated balance sheet: Refers to a company's balance sheet that is overly leveraged, inefficient, or burdened with debt, which can impact its financial flexibility, credit rating, or ability to invest in growth opportunities.
46. Debt high: Describes a company's high level of debt or leverage, which can impact its financial flexibility, credit rating, or ability to invest in growth opportunities.
47. Reversal: Refers to a change in a company's financial performance, such as a decline in revenue or profitability, which can be due to various factors such as competition, market conditions, or internal issues.
48. Debtors increasing or going up: Refers to an increase in a company's accounts receivable or debtors, which can impact its working capital, cash flow, or liquidity.
49. Receivables increase: Describes an increase in a company's accounts receivable, which can impact its working capital, cash flow, or liquidity.
50. Challenges in collections: Refers to difficulties a company faces in collecting its accounts receivable or debtors, which can impact its cash flow, liquidity, or financial performance.
51. Slow down on disbursement: A reduction in the rate at which loans or funds are disbursed, often due to economic uncertainty, regulatory restrictions, or risk aversion.
52. Write-offs: The process of removing a debt or asset from a company's balance sheet because it is deemed uncollectible or worthless. Write-offs can result in a loss for the company.
53. Increase of provisioning: An increase in the amount of money set aside by a financial institution to cover potential losses on loans or assets, often due to a rise in credit risk or expected defaults.
54. Delinquency increase: A rise in the number of borrowers who are late or behind on their loan payments, often indicating a deterioration in credit quality.
55. GNPA increasing: An increase in Gross Non-Performing Assets (GNPA), which refers to the total value of loans that are overdue or in default, without considering provisions made for potential losses.
56. Misappropriation of funds: The unauthorized or improper use of funds, often by an individual or organization, for personal gain or other unauthorized purposes.
57. Increase in credit cost: A rise in the cost of borrowing or lending, often due to changes in interest rates, credit spreads, or other market conditions.
58. Slippages: The reclassification of loans from a performing to a non-performing category, often due to a borrower's failure to meet repayment obligations.
59. High credit deposit ratio: A situation where a bank's credit growth exceeds its deposit growth, potentially leading to liquidity risks or funding constraints.
60. CAR decreasing: A decline in the Capital Adequacy Ratio (CAR), which measures a bank's capital as a percentage of its risk-weighted assets. A decreasing CAR indicates a reduction in a bank's capital buffer.
61. Provision coverage falling: A decline in the provision coverage ratio, indicating that the provisions made for potential losses are decreasing relative to the growth in non-performing assets.
62. Low Profitability: A state where a business, project, or investment generates revenue, but the net income or return on investment (ROI) is significantly lower than expected, industry average, or benchmark. This can be due to various factors such as high operating costs, intense competition, inefficient operations, or poor market conditions.
63. Falling Net Interest Margin (NIM): A decrease in the difference between the interest income earned by a financial institution and the interest expense paid on deposits and other borrowings due to changes in interest or deposit rate, reduced profitability etc.
64. Negative Capital Employed: Statements that indicate a company's liabilities exceed its assets, or its return on capital employed is negative, such as high debt levels, significant losses, or insufficient cash flow.
65. Fall in networth: A decrease in net worth over a specific period.

Now analyze the earnings call transcript and identify ALL red flags following the format above."""

    def analyze_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Complete 4-iteration analysis pipeline for financial red flags"""
        print(f"Starting complete 4-iteration analysis of: {pdf_path}")
        
        # Extract text from PDF
        print("Extracting text from PDF...")
        document_text = self.pdf_extractor.extract_text_from_pdf(pdf_path)
        
        # ITERATION 1: Initial red flag identification
        print("Running Iteration 1 - Initial Red Flag Analysis...")
        iteration1_prompt = f"{self.analyst_prompt}\n\nDOCUMENT TO ANALYZE:\n{document_text}\n\nAnalysis:"
        
        start_time = time.time()
        iteration1_response = self.llm._call(iteration1_prompt)
        iteration1_time = time.time() - start_time
        print(f"Iteration 1 completed in {iteration1_time:.2f} seconds")
        
        # ITERATION 2: Deduplication and cleanup
        print("Running Iteration 2 - Deduplication and Cleanup...")
        iteration2_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{document_text}

Previous Analysis: {iteration1_response}

Based on the above analysis and the original context, please answer: Remove the duplicates from the above context. Also if the Original Quote and Keyword identifies is same remove them. 

Ensure that:
1. No duplicate red flags are present
2. Similar or redundant information is consolidated
3. All unique red flags are retained
4. The output maintains the structure and clarity of the original analysis
5. Keep the same format: number, red flag keyword, and original quote with page numbers

Answer:"""
        
        start_time = time.time()
        iteration2_response = self.llm._call(iteration2_prompt)
        iteration2_time = time.time() - start_time
        print(f"Iteration 2 completed in {iteration2_time:.2f} seconds")
        
        # ITERATION 3: Categorization of red flags
        print("Running Iteration 3 - Categorization...")
        iteration3_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{document_text}

Previous Analysis: {iteration2_response}

Based on the above analysis and the original context, please answer: Based on the deduplicated red flags identified in the previous analysis, I need help categorizing these red flags into the following 8 categories:

1. Balance Sheet Issues: Red flags related to the company's assets, liabilities, equity, and overall financial position.
2. P&L (Income Statement) Issues: Red flags related to the company's revenues, expenses, profits, and overall financial performance.
3. Management Issues: Red flags related to the company's leadership, governance, and decision-making processes.
4. Regulatory Issues: Red flags related to the company's compliance with laws, regulations, and industry standards.
5. Industry and Market Issues: Red flags related to the company's position within its industry, market trends, and competitive landscape.
6. Operational Issues: Red flags related to the company's internal processes, systems, and infrastructure.
7. Financial Reporting and Transparency Issues: Red flags related to the company's financial reporting, disclosure, and transparency practices.
8. Strategic Issues: Red flags related to the company's overall strategy, vision, and direction.

**Task**: Take each red flag from the deduplicated list in the previous analysis and assign it to the most appropriate category above.

**Rules for categorization**:
- If a red flag could fit into multiple categories, assign it to the one that seems most closely related
- Do not leave any flag unclassified
- Do not assign any flag to multiple categories
- Ensure clear classification that groups similar issues together

**Output Format**:
### 1. Balance Sheet Issues
- [Red flag 1 from previous analysis]
- [Red flag 2 from previous analysis]

### 2. P&L (Income Statement) Issues
- [Red flag 1 from previous analysis]
- [Red flag 2 from previous analysis]

Continue this format for all 8 categories, ensuring every red flag from the previous analysis is categorized.

Answer:"""
        
        start_time = time.time()
        iteration3_response = self.llm._call(iteration3_prompt)
        iteration3_time = time.time() - start_time
        print(f"Iteration 3 completed in {iteration3_time:.2f} seconds")
        
        # ITERATION 4: Detailed summary generation
        print("Running Iteration 4 - Summary Generation...")
        iteration4_prompt = f"""You must answer the question strictly based on the below given context.

Context:
{document_text}

Previous Analysis: {iteration3_response}

Based on the above analysis and the original context, please answer: Based on the categorized red flags from the previous analysis, provide a comprehensive and detailed summary of each category of red flags. Follow these guidelines:

1. **Retain all information**: Ensure that no details are omitted or lost during the summarization process
2. **Maintain a neutral tone**: Present the summary in a factual and objective manner, avoiding any emotional or biased language
3. **Focus on factual content**: Base the summary solely on the information associated with each red flag, without introducing external opinions or assumptions
4. **Include all red flags**: Incorporate every red flag within the category into the summary, without exception
5. **Balance detail and concision**: Provide a summary that is both thorough and concise, avoiding unnecessary elaboration while still conveying all essential information
6. **Incorporate quantitative data**: Wherever possible, include quantitative data and statistics to support the summary and provide additional context
7. **Category-specific content**: Ensure that the summary is generated based solely on the content present within each category
8. **Summary should be factual**: Avoid any subjective interpretations or opinions

Format the output exactly like this example:
### 1. Balance Sheet Issues
[Detailed paragraph summary of all balance sheet related red flags with specific data points and factual information]

### 2. P&L (Income Statement) Issues  
[Detailed paragraph summary of all P&L related red flags with specific data points and factual information]

Continue this format for all 8 categories. Each summary should be a comprehensive paragraph that captures all the relevant red flags and their details within that category.

Answer:"""
        
        start_time = time.time()
        iteration4_response = self.llm._call(iteration4_prompt)
        iteration4_time = time.time() - start_time
        print(f"Iteration 4 completed in {iteration4_time:.2f} seconds")
        
        # Create comprehensive results DataFrame
        total_time = iteration1_time + iteration2_time + iteration3_time + iteration4_time
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Parse initial red flags for count
        red_flags = self._parse_response(iteration1_response)
        
        results_data = {
            'iteration': [1, 2, 3, 4],
            'stage': [
                'Initial Red Flag Analysis',
                'Deduplication and Cleanup', 
                'Categorization',
                'Summary Generation'
            ],
            'response': [
                iteration1_response,
                iteration2_response,
                iteration3_response,
                iteration4_response
            ],
            'processing_time_seconds': [
                round(iteration1_time, 2),
                round(iteration2_time, 2), 
                round(iteration3_time, 2),
                round(iteration4_time, 2)
            ],
            'document_name': [pdf_name] * 4,
            'analysis_timestamp': [timestamp] * 4
        }
        
        results_df = pd.DataFrame(results_data)
        
        print(f"Complete 4-iteration analysis completed!")
        print(f"- Total processing time: {total_time:.2f} seconds")
        print(f"- Initial red flags found: {len(red_flags)}")
        print(f"- Final categorized summary available in iteration 4")
        
        return results_df
    
    def _parse_response(self, response: str) -> list:
        """Parse LLM response into structured red flags"""
        red_flags = []
        
        # Split response into sections
        sections = response.split('\n\n')
        current_flag = {}
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Check if this is a new red flag (starts with number)
            if section[0].isdigit() and '. The potential red flag you observed' in section:
                # Save previous flag if exists
                if current_flag:
                    red_flags.append(current_flag)
                
                # Start new flag
                current_flag = {
                    'red_flag_number': len(red_flags) + 1,
                    'red_flag_keyword': section.split(' - ')[-1] if ' - ' in section else 'Not specified',
                    'original_quote': '',
                    'page_number': 'Not specified'
                }
            
            elif 'Original Quote:' in section:
                # Extract quote and page number
                quote_text = section.replace('Original Quote:', '').strip()
                
                # Try to extract page number
                if 'Page' in quote_text:
                    import re
                    page_match = re.search(r'\(Page (\d+)\)', quote_text)
                    if page_match:
                        current_flag['page_number'] = page_match.group(1)
                        quote_text = quote_text.replace(page_match.group(0), '').strip()
                
                current_flag['original_quote'] = quote_text
        
        # Add the last flag
        if current_flag:
            red_flags.append(current_flag)
        
        return red_flags
    
    def save_to_excel(self, results_df: pd.DataFrame, output_path: str = None) -> str:
        """Save complete 4-iteration results to Excel file with multiple sheets"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"financial_red_flags_complete_analysis_{timestamp}.xlsx"
        
        # Create Excel writer with formatting
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Write main results - all 4 iterations
            results_df.to_excel(writer, sheet_name='Complete Analysis', index=False)
            
            # Create individual sheets for each iteration
            for idx, row in results_df.iterrows():
                sheet_name = f"Iteration {row['iteration']} - {row['stage'][:15]}"
                
                # Create a detailed sheet for this iteration
                iteration_data = {
                    'Stage': [row['stage']],
                    'Processing Time (seconds)': [row['processing_time_seconds']],
                    'Document Name': [row['document_name']],
                    'Analysis Timestamp': [row['analysis_timestamp']],
                    'Full Response': [row['response']]
                }
                iteration_df = pd.DataFrame(iteration_data)
                iteration_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create a summary sheet for iteration 1 (parsed red flags)
            if len(results_df) > 0:
                iteration1_response = results_df.iloc[0]['response']
                red_flags = self._parse_response(iteration1_response)
                
                if red_flags:
                    red_flags_df = pd.DataFrame(red_flags)
                    red_flags_df.to_excel(writer, sheet_name='Parsed Red Flags (Iter 1)', index=False)
            
            # Get workbook for formatting
            workbook = writer.book
            
            # Format all worksheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # Add formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                cell_format = workbook.add_format({
                    'text_wrap': True,
                    'valign': 'top',
                    'border': 1
                })
                
                # Auto-adjust column widths based on sheet
                if sheet_name == 'Complete Analysis':
                    worksheet.set_column('A:A', 12)  # iteration
                    worksheet.set_column('B:B', 25)  # stage
                    worksheet.set_column('C:C', 100) # response
                    worksheet.set_column('D:D', 20)  # processing_time
                    worksheet.set_column('E:E', 25)  # document_name
                    worksheet.set_column('F:F', 20)  # timestamp
                elif sheet_name == 'Parsed Red Flags (Iter 1)':
                    worksheet.set_column('A:A', 15)  # red_flag_number
                    worksheet.set_column('B:B', 30)  # red_flag_keyword  
                    worksheet.set_column('C:C', 80)  # original_quote
                    worksheet.set_column('D:D', 15)  # page_number
                else:
                    # Individual iteration sheets
                    worksheet.set_column('A:A', 25)  # Stage
                    worksheet.set_column('B:B', 20)  # Processing Time
                    worksheet.set_column('C:C', 25)  # Document Name
                    worksheet.set_column('D:D', 20)  # Timestamp
                    worksheet.set_column('E:E', 120) # Full Response
        
        print(f"Complete analysis results saved to: {output_path}")
        print(f"Excel file contains:")
        print(f"- Complete Analysis: All 4 iterations overview")
        print(f"- Iteration 1-4: Individual detailed sheets")
        print(f"- Parsed Red Flags: Structured red flags from iteration 1")
        
        return output_path

def main():
    """Main function to run the analysis"""
    # Initialize analyzer
    analyzer = FinancialRedFlagAnalyzer()
    
    # Specify your PDF path here
    pdf_path = "your_earnings_call_transcript.pdf"  # Update this path
    
    try:
        # Run analysis
        results_df = analyzer.analyze_pdf(pdf_path)
        
        # Save to Excel
        output_file = analyzer.save_to_excel(results_df)
        
        print(f"\nComplete 4-Iteration Analysis Summary:")
        print(f"- Document analyzed: {pdf_path}")
        print(f"- Total processing time: {results_df['processing_time_seconds'].sum():.2f} seconds")
        print(f"- Output file: {output_file}")
        
        # Display iteration summary
        print(f"\nIteration Summary:")
        for idx, row in results_df.iterrows():
            print(f"- Iteration {row['iteration']}: {row['stage']} ({row['processing_time_seconds']}s)")
        
        # Show snippet of final summary (iteration 4)
        if len(results_df) >= 4:
            final_summary = results_df.iloc[3]['response']
            print(f"\nFinal Summary Preview (first 300 chars):")
            print(f"{final_summary[:300]}...")
        
        return results_df
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    main()
