from bs4 import BeautifulSoup
import re
import os
import json

def extract_strengths_weaknesses(html_file_path):
    """Extract strengths and weaknesses from a single HTML file."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    print(f"Parsing file: {html_file_path}")
    
    # Find the "Key Rating Drivers" section
    key_drivers_element = None
    
    # Look for the exact text "Key Rating Drivers & Detailed Description"
    for element in soup.find_all(['p', 'span']):
        text = element.get_text().strip()
        if 'Key Rating Drivers' in text and 'Detailed Description' in text:
            key_drivers_element = element
            print(f"Found Key Rating Drivers section in: {element.name}")
            break
    
    if not key_drivers_element:
        print("Could not find Key Rating Drivers section")
        return {}, {}
    
    # Find the parent container that contains all the rating drivers content
    container = key_drivers_element.find_parent(['div', 'td', 'table'])
    if not container:
        container = key_drivers_element.find_parent()
    
    print(f"Using container: {container.name if container else 'None'}")
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    # Find all elements after the Key Rating Drivers header
    all_elements = container.find_all(['p', 'ul', 'li']) if container else []
    
    print(f"Found {len(all_elements)} elements to process")
    
    # Track if we've found the Key Rating Drivers section
    found_key_section = False
    
    for element in all_elements:
        text = element.get_text().strip()
        
        # Skip until we find the Key Rating Drivers section
        if not found_key_section:
            if 'Key Rating Drivers' in text:
                found_key_section = True
                print("Found Key Rating Drivers section, starting parsing...")
            continue
        
        # Check for section headers
        if element.name == 'p':
            if re.search(r'\bStrengths?\s*:?\s*$', text, re.IGNORECASE):
                current_section = 'strengths'
                print("Found Strengths section")
                continue
            elif re.search(r'\bWeakness(es)?\s*:?\s*$', text, re.IGNORECASE):
                current_section = 'weaknesses'
                print("Found Weaknesses section")
                continue
        
        # Process list items
        if element.name == 'li' and current_section:
            print(f"Processing {current_section} item...")
            
            # Find all spans within the li
            spans = element.find_all('span')
            
            # Look for bold spans (these contain the keys)
            bold_spans = []
            regular_spans = []
            
            for span in spans:
                # Check if span has bold styling
                style = span.get('style', '')
                if 'font-weight:bold' in style or span.find(['strong', 'b']):
                    bold_spans.append(span)
                else:
                    regular_spans.append(span)
            
            # Extract key and value
            if bold_spans:
                # Combine all bold text as key
                key_parts = []
                for bold_span in bold_spans:
                    key_text = bold_span.get_text().strip()
                    if key_text and key_text != ':':
                        key_parts.append(key_text)
                
                key = ' '.join(key_parts).rstrip(':').strip()
                
                # Combine all regular text as value
                value_parts = []
                for regular_span in regular_spans:
                    value_text = regular_span.get_text().strip()
                    if value_text:
                        value_parts.append(value_text)
                
                value = ' '.join(value_parts).strip()
                
                # Clean up value - remove any leading colons or spaces
                value = re.sub(r'^[:\s]+', '', value)
                
                if key and value:
                    print(f"  Key: {key[:50]}...")
                    print(f"  Value: {value[:50]}...")
                    
                    if current_section == 'strengths':
                        strengths_dict[key] = value
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = value
                else:
                    print(f"  Skipping - Key: '{key}', Value: '{value}'")
        
        # Stop processing if we hit another major section
        if element.name == 'p' and any(keyword in text.lower() for keyword in ['liquidity', 'outlook', 'analytical approach']):
            print(f"Reached next section: {text[:30]}... Stopping parsing.")
            break
    
    print(f"Extraction complete - Strengths: {len(strengths_dict)}, Weaknesses: {len(weaknesses_dict)}")
    return strengths_dict, weaknesses_dict

def process_single_file(file_path):
    """Process a single HTML file for testing."""
    try:
        print(f"Processing: {file_path}")
        strengths, weaknesses = extract_strengths_weaknesses(file_path)
        
        result = {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
        
        return result
        
    except Exception as e:
        print(f"Error with {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_folder(folder_path='.'):
    """Process all HTML files in a folder."""
    all_results = {}
    
    # Find all HTML files
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    
    print(f"Found {len(html_files)} HTML files")
    
    # Process each file
    for filename in html_files:
        file_path = os.path.join(folder_path, filename)
        result = process_single_file(file_path)
        
        if result:
            file_key = filename.replace('.html', '').replace('.htm', '')
            all_results[file_key] = result
            print(f"  - Found {len(result['strengths'])} strengths and {len(result['weaknesses'])} weaknesses")
        else:
            print(f"  - Failed to process {filename}")
    
    return all_results

def save_and_print_results(results):
    """Save to JSON and print detailed results."""
    # Save to JSON
    output_file = 'extracted_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("DETAILED EXTRACTION RESULTS")
    print(f"{'='*80}")
    
    for filename, data in results.items():
        print(f"\n📄 FILE: {filename}")
        print(f"{'='*50}")
        
        print(f"\n💪 STRENGTHS ({len(data['strengths'])}):")
        print("-" * 40)
        for i, (key, value) in enumerate(data['strengths'].items(), 1):
            print(f"{i}. KEY: {key}")
            print(f"   VALUE: {value}")
            print()
        
        print(f"\n⚠️  WEAKNESSES ({len(data['weaknesses'])}):")
        print("-" * 40)
        for i, (key, value) in enumerate(data['weaknesses'].items(), 1):
            print(f"{i}. KEY: {key}")
            print(f"   VALUE: {value}")
            print()
    
    print(f"\n💾 Results saved to '{output_file}'")
    return output_file

# Main execution
if __name__ == "__main__":
    # You can specify a specific file for testing
    specific_file = "Rating Rationale.html"  # Change this to your file name
    
    if os.path.exists(specific_file):
        print(f"🔍 Processing specific file: {specific_file}")
        result = process_single_file(specific_file)
        if result:
            results = {specific_file.replace('.html', ''): result}
            save_and_print_results(results)
        else:
            print("❌ Failed to process the file!")
    else:
        print(f"🔍 Processing all HTML files in current folder...")
        results = process_folder('.')
        
        if results:
            save_and_print_results(results)
        else:
            print("❌ No HTML files found or processed successfully!")
