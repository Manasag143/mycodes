from bs4 import BeautifulSoup
import re
import json
import os

def extract_rating_drivers(html_file_path):
    """
    Simple pipeline to extract strengths and weaknesses from Rating document.
    Returns dictionary with sub-topics as keys and content as values.
    """
    
    print(f"📄 Processing: {os.path.basename(html_file_path)}")
    
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    # Step 1: Find "Key Rating Drivers & Detailed Description" section
    key_section_found = False
    rating_drivers_section = None
    
    for element in soup.find_all(['p', 'span']):
        text = element.get_text().strip()
        if 'Key Rating Drivers' in text and 'Detailed Description' in text:
            rating_drivers_section = element.find_parent(['div', 'td', 'table'])
            key_section_found = True
            print("✅ Found 'Key Rating Drivers & Detailed Description' section")
            break
    
    if not rating_drivers_section:
        print("❌ Could not find 'Key Rating Drivers & Detailed Description' section")
        return {}
    
    # Step 2: Extract all content from this section
    result_dict = {}
    current_section = None  # 'strengths' or 'weaknesses'
    
    # Get all elements in the rating drivers section
    all_elements = rating_drivers_section.find_all(['p', 'ul', 'li'])
    
    for element in all_elements:
        element_text = element.get_text().strip()
        
        # Step 3: Identify Strengths and Weaknesses headers
        if element.name == 'p':
            # Look for "Strengths:" header
            if re.search(r'\bStrengths?\s*:?\s*$', element_text, re.IGNORECASE):
                current_section = 'strengths'
                print("💪 Found Strengths section")
                continue
            
            # Look for "Weakness:" header  
            elif re.search(r'\bWeakness(es)?\s*:?\s*$', element_text, re.IGNORECASE):
                current_section = 'weaknesses'
                print("⚠️ Found Weaknesses section")
                continue
        
        # Step 4: Extract key-value pairs from list items
        if element.name == 'li' and current_section:
            
            # Find all spans in this list item
            spans = element.find_all('span')
            
            key = None
            value_parts = []
            
            for span in spans:
                span_style = span.get('style', '')
                span_text = span.get_text().strip()
                
                # If it's a bold span and not just a colon, it's our key
                if 'font-weight:bold' in span_style and span_text and span_text != ':':
                    if not key:  # Take the first bold text as key
                        key = span_text.rstrip(':').strip()
                
                # If it's not bold (or just a colon), it's part of the value
                elif span_text and ('font-weight:bold' not in span_style or span_text == ':'):
                    if span_text != ':':  # Skip standalone colons
                        value_parts.append(span_text)
            
            # Clean and combine the value
            if key and value_parts:
                value = ' '.join(value_parts).strip()
                # Remove any leading colons or extra spaces
                value = re.sub(r'^[:\s]+', '', value)
                value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                
                if value:  # Only add if we have actual content
                    result_dict[key] = value
                    print(f"   ✅ {key}: {value[:60]}...")
        
        # Step 5: Stop when we reach next major section
        if element.name == 'p' and current_section:
            lower_text = element_text.lower()
            stop_keywords = ['liquidity', 'outlook', 'analytical approach', 'rating sensitivity', 'about the company']
            if any(keyword in lower_text for keyword in stop_keywords):
                print(f"🛑 Reached next section, stopping extraction")
                break
    
    print(f"📊 Extracted {len(result_dict)} items total")
    return result_dict

def process_files(folder_path='.', specific_file=None):
    """
    Process either a specific file or all HTML files in folder
    """
    
    if specific_file and os.path.exists(specific_file):
        # Process single file
        print(f"🎯 Processing specific file: {specific_file}")
        result = extract_rating_drivers(specific_file)
        
        if result:
            return {specific_file.replace('.html', '').replace('.htm', ''): result}
        else:
            return {}
    
    else:
        # Process all HTML files in folder
        html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
        print(f"🔍 Found {len(html_files)} HTML files in folder")
        
        all_results = {}
        
        for filename in html_files:
            file_path = os.path.join(folder_path, filename)
            print(f"\n{'-'*50}")
            
            result = extract_rating_drivers(file_path)
            
            if result:
                file_key = filename.replace('.html', '').replace('.htm', '')
                all_results[file_key] = result
                print(f"✅ Successfully processed {filename}")
            else:
                print(f"❌ No data extracted from {filename}")
        
        return all_results

def save_and_display_results(results):
    """
    Save results to JSON and display in a clean format
    """
    
    if not results:
        print("❌ No results to save!")
        return
    
    # Save to JSON file
    output_file = 'rating_drivers_extracted.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Display results
    print(f"\n{'='*80}")
    print("📋 EXTRACTED RATING DRIVERS")
    print(f"{'='*80}")
    
    for filename, data in results.items():
        print(f"\n📄 FILE: {filename}")
        print(f"{'='*60}")
        
        # Separate strengths and weaknesses for better display
        strengths = {}
        weaknesses = {}
        
        for key, value in data.items():
            # Simple heuristic to categorize
            key_lower = key.lower()
            if any(word in key_lower for word in ['strong', 'robust', 'good', 'efficient', 'leading', 'position']):
                strengths[key] = value
            elif any(word in key_lower for word in ['exposure', 'risk', 'volatility', 'weakness', 'dependent']):
                weaknesses[key] = value
            else:
                # If unclear, put in strengths by default (most items are strengths)
                strengths[key] = value
        
        # Display strengths
        if strengths:
            print(f"\n💪 STRENGTHS ({len(strengths)}):")
            print("-" * 40)
            for i, (key, value) in enumerate(strengths.items(), 1):
                print(f"{i}. KEY: {key}")
                print(f"   VALUE: {value}")
                print()
        
        # Display weaknesses  
        if weaknesses:
            print(f"⚠️  WEAKNESSES ({len(weaknesses)}):")
            print("-" * 40)
            for i, (key, value) in enumerate(weaknesses.items(), 1):
                print(f"{i}. KEY: {key}")
                print(f"   VALUE: {value}")
                print()
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"📊 Total items extracted: {sum(len(data) for data in results.values())}")

def main():
    """
    Main execution function
    """
    
    print("🚀 RATING DRIVERS EXTRACTION PIPELINE")
    print("="*50)
    
    # Option 1: Process specific file (recommended)
    specific_file = "Rating Rationale.html"  # Change this to your file name
    
    if os.path.exists(specific_file):
        print(f"✅ Found specific file: {specific_file}")
        results = process_files(specific_file=specific_file)
    else:
        print(f"❌ Specific file '{specific_file}' not found")
        print("🔍 Processing all HTML files in current directory...")
        results = process_files('.')
    
    # Save and display results
    if results:
        save_and_display_results(results)
        print("\n🎉 Extraction completed successfully!")
    else:
        print("\n❌ No data extracted. Please check your HTML file structure.")

# Test function to verify our expected results
def test_expected_results():
    """
    Test function to show what we expect to extract
    """
    
    expected_results = {
        "Rating Rationale": {
            "Strong position in India's phosphatic-fertiliser market": "Coromandel is the second-largest player in the phosphatic-fertiliser industry in India with a primary market share of ~15% in DAP/NPK...",
            "Strong operating efficiency": "Operations benefit from economies of scale, better raw material procurement due to established relationships with suppliers...",
            "Robust financial risk profile": "Coromandel maintains a net cash position of over Rs 2,000 crore...",
            "Exposure to regulated nature of the fertiliser industry and volatility in raw material prices": "The fertiliser industry is strategic, but highly controlled, with fertiliser subsidy being an important component of profitability..."
        }
    }
    
    print("🎯 EXPECTED RESULTS:")
    print("="*50)
    for key in expected_results["Rating Rationale"]:
        print(f"• {key}")

if __name__ == "__main__":
    # Uncomment the line below to see expected results
    # test_expected_results()
    
    # Run the main extraction
    main()
