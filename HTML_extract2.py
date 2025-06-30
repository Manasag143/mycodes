from bs4 import BeautifulSoup
import re
import os
import json

def extract_strengths_weaknesses(html_file_path):
    """Extract strengths and weaknesses from a single HTML file."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
    
    print(f"📄 Processing file: {os.path.basename(html_file_path)}")
    
    strengths_dict = {}
    weaknesses_dict = {}
    
    # Find the Key Rating Drivers section first
    key_drivers_found = False
    current_section = None
    
    # Get all elements in the document
    all_elements = soup.find_all(['p', 'ul', 'li'])
    
    for element in all_elements:
        element_text = element.get_text().strip()
        
        # Look for the Key Rating Drivers section
        if not key_drivers_found:
            if 'Key Rating Drivers' in element_text and 'Detailed Description' in element_text:
                key_drivers_found = True
                print("✅ Found Key Rating Drivers section")
                continue
        
        if not key_drivers_found:
            continue
        
        # Look for section headers
        if element.name == 'p':
            # Check for Strengths header
            if element.find('span', string=re.compile(r'^\s*Strengths?\s*$', re.IGNORECASE)):
                current_section = 'strengths'
                print("✅ Found Strengths section")
                continue
            # Check for Weakness header  
            elif element.find('span', string=re.compile(r'^\s*Weakness(es)?\s*:?\s*$', re.IGNORECASE)):
                current_section = 'weaknesses'
                print("✅ Found Weaknesses section")
                continue
        
        # Process list items
        if element.name == 'li' and current_section:
            print(f"🔍 Processing {current_section} item...")
            
            # Get all spans in this li
            spans = element.find_all('span')
            
            # Find the key (first bold span that's not just a colon)
            key = None
            key_span = None
            
            for span in spans:
                style = span.get('style', '')
                text = span.get_text().strip()
                
                if 'font-weight:bold' in style and text and text != ':':
                    key = text.rstrip(':').strip()
                    key_span = span
                    break
            
            if key and key_span:
                print(f"   🔑 Key found: {key}")
                
                # Get all text content after the key span
                # Find all non-bold spans that come after the key
                value_parts = []
                found_key_span = False
                
                for span in spans:
                    if span == key_span:
                        found_key_span = True
                        continue
                    
                    if found_key_span:
                        style = span.get('style', '')
                        text = span.get_text().strip()
                        
                        # Skip bold spans (except if they're just colons) and empty spans
                        if text and not ('font-weight:bold' in style and text != ':'):
                            value_parts.append(text)
                
                # Join all value parts
                value = ' '.join(value_parts).strip()
                
                # Clean up the value
                value = re.sub(r'^[:\s]+', '', value)  # Remove leading colons and spaces
                value = re.sub(r'\s+', ' ', value)     # Normalize whitespace
                
                if value:
                    print(f"   💎 Value found: {value[:80]}...")
                    
                    if current_section == 'strengths':
                        strengths_dict[key] = value
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = value
                else:
                    print(f"   ⚠️ No value found for key: {key}")
            else:
                print(f"   ❌ No key found in this list item")
        
        # Stop if we reach another major section
        if element.name == 'p' and current_section:
            lower_text = element_text.lower()
            if any(keyword in lower_text for keyword in ['liquidity', 'outlook', 'analytical approach', 'rating sensitivity']):
                print(f"🛑 Reached next section: {element_text[:50]}... Stopping")
                break
    
    print(f"📊 Extraction complete:")
    print(f"   💪 Strengths: {len(strengths_dict)}")
    print(f"   ⚠️ Weaknesses: {len(weaknesses_dict)}")
    
    return strengths_dict, weaknesses_dict

def process_single_file(file_path):
    """Process a single HTML file for testing."""
    try:
        strengths, weaknesses = extract_strengths_weaknesses(file_path)
        
        result = {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_folder(folder_path='.'):
    """Process all HTML files in a folder."""
    all_results = {}
    
    # Find all HTML files
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    
    print(f"🔍 Found {len(html_files)} HTML files")
    
    # Process each file
    for filename in html_files:
        file_path = os.path.join(folder_path, filename)
        print(f"\n{'='*60}")
        result = process_single_file(file_path)
        
        if result:
            file_key = filename.replace('.html', '').replace('.htm', '')
            all_results[file_key] = result
            print(f"✅ Successfully processed {filename}")
        else:
            print(f"❌ Failed to process {filename}")
    
    return all_results

def save_and_print_results(results):
    """Save to JSON and print detailed results."""
    if not results:
        print("❌ No results to save!")
        return None
        
    # Save to JSON
    output_file = 'extracted_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("📋 DETAILED EXTRACTION RESULTS")
    print(f"{'='*80}")
    
    total_strengths = 0
    total_weaknesses = 0
    
    for filename, data in results.items():
        print(f"\n📄 FILE: {filename}")
        print(f"{'='*50}")
        
        file_strengths = len(data['strengths'])
        file_weaknesses = len(data['weaknesses'])
        
        total_strengths += file_strengths
        total_weaknesses += file_weaknesses
        
        print(f"\n💪 STRENGTHS ({file_strengths}):")
        print("-" * 40)
        for i, (key, value) in enumerate(data['strengths'].items(), 1):
            print(f"{i}. KEY: {key}")
            print(f"   VALUE: {value}")
            print()
        
        print(f"\n⚠️  WEAKNESSES ({file_weaknesses}):")
        print("-" * 40)
        for i, (key, value) in enumerate(data['weaknesses'].items(), 1):
            print(f"{i}. KEY: {key}")
            print(f"   VALUE: {value}")
            print()
    
    print(f"\n📊 SUMMARY:")
    print(f"   📁 Files processed: {len(results)}")
    print(f"   💪 Total strengths: {total_strengths}")
    print(f"   ⚠️  Total weaknesses: {total_weaknesses}")
    print(f"   💾 Results saved to: {output_file}")
    
    return output_file

# Test with a specific function
def test_with_sample():
    """Test the parsing logic with a sample HTML snippet."""
    sample_html = '''
    <div>
    <p><span style="font-weight:bold; text-decoration:underline">Key Rating Drivers & Detailed Description</span></p>
    <p><span style="font-weight:bold">Strengths</span><span style="font-weight:bold">:</span></p>
    <ul>
        <li><span style="font-weight:bold">Strong position in market</span><span style="font-weight:bold">: </span><span>Company is the leader with 25% market share.</span></li>
    </ul>
    <p><span style="font-weight:bold">Weakness:</span></p>
    <ul>
        <li><span style="font-weight:bold">High dependency on imports</span><span style="font-weight:bold">: </span><span>Company imports 80% of raw materials.</span></li>
    </ul>
    </div>
    '''
    
    # Save sample to temp file
    with open('test_sample.html', 'w', encoding='utf-8') as f:
        f.write(sample_html)
    
    print("🧪 Testing with sample HTML...")
    result = process_single_file('test_sample.html')
    
    if result:
        print("✅ Sample test successful!")
        for key, value in result['strengths'].items():
            print(f"   💪 {key}: {value}")
        for key, value in result['weaknesses'].items():
            print(f"   ⚠️ {key}: {value}")
    
    # Clean up
    os.remove('test_sample.html')

# Main execution
if __name__ == "__main__":
    # First test with sample
    print("🧪 Running sample test first...")
    test_with_sample()
    
    print(f"\n{'='*60}")
    
    # Then process actual files
    specific_file = "Rating Rationale.html"  # Change this to your file name
    
    if os.path.exists(specific_file):
        print(f"🎯 Processing specific file: {specific_file}")
        result = process_single_file(specific_file)
        if result:
            results = {specific_file.replace('.html', ''): result}
            save_and_print_results(results)
        else:
            print("❌ Failed to process the specific file!")
    else:
        print(f"🔍 Processing all HTML files in current folder...")
        results = process_folder('.')
        
        if results:
            save_and_print_results(results)
        else:
            print("❌ No HTML files found or processed successfully!")
