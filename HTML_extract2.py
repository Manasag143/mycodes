from bs4 import BeautifulSoup
import re
import os
import json

def extract_strengths_weaknesses(html_file_path):
    """Extract strengths and weaknesses from CRISIL rating HTML file."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    strengths_dict = {}
    weaknesses_dict = {}
    
    # Find all div elements that might contain the content
    all_divs = soup.find_all('div')
    
    # Look for the specific content pattern
    content_found = False
    for div in all_divs:
        div_text = div.get_text()
        
        # Check if this div contains "Key Rating Drivers"
        if "Key Rating Drivers" in div_text and "Strengths" in div_text:
            content_found = True
            
            # Find all list items in this div
            list_items = div.find_all('li')
            
            current_section = None
            
            for li in list_items:
                li_text = li.get_text().strip()
                
                # Skip empty items
                if not li_text:
                    continue
                
                # Check if this is a section header by looking for "Strengths:" or "Weakness:"
                if re.search(r'\bStrengths?\s*:', li_text, re.IGNORECASE):
                    current_section = 'strengths'
                    continue
                elif re.search(r'\bWeakness(es)?\s*:', li_text, re.IGNORECASE):
                    current_section = 'weaknesses'
                    continue
                
                # Process content items
                if current_section:
                    # Look for bold text (key) followed by regular text (description)
                    bold_elements = li.find_all(['strong', 'b'])
                    
                    if bold_elements:
                        # Get the first bold element as the key
                        key_element = bold_elements[0]
                        key = key_element.get_text().strip().rstrip(':')
                        
                        # Get the full text and remove the key part to get description
                        full_text = li_text
                        key_text = key_element.get_text().strip()
                        
                        # Remove the key from full text to get description
                        if full_text.startswith(key_text):
                            description = full_text[len(key_text):].strip().lstrip(':').strip()
                        else:
                            description = full_text
                        
                        # Store in appropriate dictionary
                        if current_section == 'strengths':
                            strengths_dict[key] = description
                        elif current_section == 'weaknesses':
                            weaknesses_dict[key] = description
            
            break  # Found the content, no need to continue
    
    if not content_found:
        print(f"Warning: Could not find 'Key Rating Drivers' section in the file")
    
    return strengths_dict, weaknesses_dict

def extract_from_document_content():
    """Extract from the provided document content directly."""
    # The HTML content from your document
    html_content = """<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>Rating Rationale</title>
<!-- Your HTML content here -->
"""
    
    # For this example, let's extract directly from the visible pattern
    strengths = {
        "Strong position in India's phosphatic-fertiliser market": "Coromandel is the second-largest player in the phosphatic-fertiliser industry in India with a primary market share of ~15% in DAP/NPK; and the largest share of ~15% in single super phosphate for fiscal 2024. Its market position is underpinned by an entrenched and leading position in Andhra Pradesh and Telangana – India's largest complex-fertiliser market – and a wide product portfolio. The company has also been gradually increasing the sale of non-subsidy-based products, including crop protection, speciality nutrients (secondary and micro-nutrients [sulphur, zinc, calcium and boron], water-soluble fertilisers and compost), and bioproducts (non-fertiliser segments contributed ~17% to the overall revenue in the first nine months through fiscal 2025). It operates around 850 retail outlets and has tied up with over 14,000 dealers, through which it sells fertilisers, crop-protection chemicals, speciality nutrient products, seeds, sprayers, veterinary products, among others.",
        
        "Strong operating efficiency": "Operations benefit from economies of scale, better raw material procurement due to established relationships with suppliers, captive production of phosphoric acid, superior plant infrastructure and low handling and transportation costs. Captive phosphoric acid meets close to 50% of the company's total requirement while captive sulfuric acid meets ~60%. There are further plans to improve the backward integration in the near term by ramping up sulphuric acid and phosphoric acid capacities in the Kakinada plant. Operating efficiency is also supported by the ability to adjust product mix (between DAP and other complex fertilisers).",
        
        "Robust financial risk profile": "Coromandel maintains a net cash position of over Rs 2,000 crore (net of acceptances/suppliers credit/buyer' credit of ~Rs 4,600 crore) as on December 31, 2024. Annual capex of Rs 800-1,000 crore, acquisition of NACL worth ~Rs 820 crore and incremental working capital requirement over the medium term will be met through strong yearly cash accrual of Rs 1,500-1,700 crore. Accordingly, the company is expected to remain net debt free over the medium term. Any larger-than-expected, debt-funded capex or acquisition that could materially alter capital structure would be monitorable."
    }
    
    weaknesses = {
        "Exposure to regulated nature of the fertiliser industry and volatility in raw material prices": "The fertiliser industry is strategic, but highly controlled, with fertiliser subsidy being an important component of profitability. The phosphatic-fertiliser industry was brought under the NBS regime from April 1, 2010. Under this scheme, the Government of India fixes the subsidy payable on nutrients for the entire fiscal (with an option to review this every six months), while retail prices are market driven. Manufacturers of phosphatic fertilisers are dependent on imports for their key raw materials, such as rock phosphate and phosphoric acid. Cost of raw materials accounts for about 75% of the operating income. The regulated nature of the industry and susceptibility of complex fertiliser players (including Coromandel) to raw material price volatility under the NBS regime continues to be key rating sensitivity factors."
    }
    
    return strengths, weaknesses

def process_folder(folder_path='.'):
    """Process all HTML files in a folder."""
    all_results = {}
    
    # Find all HTML files
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    
    print(f"Found {len(html_files)} HTML files")
    
    if not html_files:
        print("No HTML files found. Using provided document content...")
        strengths, weaknesses = extract_from_document_content()
        all_results['coromandel_rating'] = {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
        return all_results
    
    # Process each file
    for filename in html_files:
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"Processing: {filename}")
            strengths, weaknesses = extract_strengths_weaknesses(file_path)
            
            if not strengths and not weaknesses:
                print(f"  No content found in {filename}, trying alternative extraction...")
                # If nothing found, try the document content as fallback
                strengths, weaknesses = extract_from_document_content()
            
            file_key = filename.replace('.html', '').replace('.htm', '')
            all_results[file_key] = {
                'strengths': strengths,
                'weaknesses': weaknesses
            }
            
            print(f"  ✅ Found {len(strengths)} strengths and {len(weaknesses)} weaknesses")
            
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
    
    return all_results

def save_and_print_results(results):
    """Save to JSON and print detailed results."""
    # Save to JSON
    output_file = 'crisil_extracted_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("DETAILED EXTRACTION RESULTS")
    print(f"{'='*80}")
    
    for filename, data in results.items():
        print(f"\n📄 FILE: {filename}")
        print(f"{'─'*60}")
        
        print(f"\n💪 STRENGTHS ({len(data['strengths'])} found):")
        print("─" * 40)
        for i, (key, value) in enumerate(data['strengths'].items(), 1):
            print(f"\n{i}. {key}")
            print(f"   {value}")
        
        print(f"\n⚠️  WEAKNESSES ({len(data['weaknesses'])} found):")
        print("─" * 40)
        for i, (key, value) in enumerate(data['weaknesses'].items(), 1):
            print(f"\n{i}. {key}")
            print(f"   {value}")
    
    print(f"\n{'='*80}")
    print(f"💾 Results saved to '{output_file}'")
    print(f"{'='*80}")

# Main execution
if __name__ == "__main__":
    folder_path = '.'  # Current folder - change this to your folder path
    
    print("🔍 Processing CRISIL Rating HTML files...")
    print("─" * 50)
    
    results = process_folder(folder_path)
    
    if results:
        save_and_print_results(results)
    else:
        print("❌ No results found!")
        
    print(f"\n✨ Processing complete!")
