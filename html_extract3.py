from bs4 import BeautifulSoup
import re
import os
import json

def extract_strengths_weaknesses(html_file_path):
    """Extract strengths and weaknesses from a single HTML file."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    print(f"\n🔍 DEBUGGING FILE: {html_file_path}")
    print("="*60)
    
    # Find the "Key Rating Drivers" section
    target_section = None
    for element in soup.find_all(['p', 'span', 'div', 'h1', 'h2', 'h3']):
        if 'Key Rating Drivers' in element.get_text():
            target_section = element.find_parent()
            break
    
    if not target_section:
        print("❌ Could not find 'Key Rating Drivers' section")
        print("🔍 Let's see what sections exist:")
        all_text = soup.get_text()
        print(f"File contains: {all_text[:200]}...")
        return {}, {}
    
    print("✅ Found 'Key Rating Drivers' section")
    
    # Get ALL paragraph elements in the target section
    all_p_elements = target_section.find_all('p')
    print(f"📄 Found {len(all_p_elements)} <p> elements in target section")
    
    # DEBUG: Print ALL paragraphs with their styles
    print(f"\n🔍 ALL PARAGRAPHS IN TARGET SECTION:")
    print("-" * 60)
    for i, p in enumerate(all_p_elements):
        text = p.get_text().strip()
        style = p.get('style', '')
        print(f"P{i+1}: '{text}'")
        print(f"     Style: '{style}'")
        print(f"     Has font-weight: {'font-weight' in style.lower()}")
        print(f"     Has bold: {'bold' in style.lower()}")
        print()
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    print(f"\n🔍 PROCESSING LOGIC:")
    print("-" * 60)
    
    i = 0
    while i < len(all_p_elements):
        p = all_p_elements[i]
        text = p.get_text().strip()
        
        if not text:  # Skip empty paragraphs
            i += 1
            continue
        
        print(f"\nProcessing P{i+1}: '{text[:50]}...'")
        
        # Check if this is a section header
        if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
            current_section = 'strengths'
            print(f"🎯 FOUND STRENGTHS SECTION!")
            i += 1
            continue
        elif re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE):
            current_section = 'weaknesses' 
            print(f"🎯 FOUND WEAKNESSES SECTION!")
            i += 1
            continue
        
        # If we're in a section, check if this is a bold element (key)
        if current_section:
            print(f"   Current section: {current_section}")
            style = p.get('style', '')
            print(f"   Style: '{style}'")
            
            # Check for bold in multiple ways
            is_bold_fontweight = 'font-weight' in style.lower() and 'bold' in style.lower()
            is_bold_700 = 'font-weight:700' in style.replace(' ', '') or 'font-weight: 700' in style
            is_bold_b_tag = p.find('b') is not None or p.find('strong') is not None
            
            print(f"   Bold checks:")
            print(f"     - font-weight:bold: {is_bold_fontweight}")
            print(f"     - font-weight:700: {is_bold_700}")
            print(f"     - <b>/<strong> tags: {is_bold_b_tag}")
            
            is_bold = is_bold_fontweight or is_bold_700 or is_bold_b_tag
            
            if is_bold:  # This is a key
                key = text.rstrip(':').strip()
                print(f"🔑 FOUND KEY: '{key}'")
                
                # Collect all following non-bold paragraphs as value
                value_parts = []
                j = i + 1
                
                print(f"   Looking for values starting from P{j+1}...")
                
                while j < len(all_p_elements):
                    next_p = all_p_elements[j]
                    next_text = next_p.get_text().strip()
                    
                    if not next_text:  # Skip empty
                        j += 1
                        continue
                    
                    print(f"   Checking P{j+1}: '{next_text[:30]}...'")
                    
                    # Check if next element is also bold (another key)
                    next_style = next_p.get('style', '')
                    next_is_bold_fontweight = 'font-weight' in next_style.lower() and 'bold' in next_style.lower()
                    next_is_bold_700 = 'font-weight:700' in next_style.replace(' ', '') or 'font-weight: 700' in next_style
                    next_is_bold_b_tag = next_p.find('b') is not None or next_p.find('strong') is not None
                    next_is_bold = next_is_bold_fontweight or next_is_bold_700 or next_is_bold_b_tag
                    
                    if next_is_bold:  # Hit another key, stop
                        print(f"   Found another bold element, stopping value collection")
                        break
                    
                    # Check if we hit a new section
                    if re.search(r'\b(Strengths?|Weakness(es)?)\s*:?', next_text, re.IGNORECASE):
                        print(f"   Found new section, stopping value collection")
                        break
                    
                    # Add this text to value
                    value_parts.append(next_text)
                    print(f"   Added to value: '{next_text[:30]}...'")
                    j += 1
                
                # Save the key-value pair
                if value_parts:
                    full_value = ' '.join(value_parts).strip()
                    
                    if current_section == 'strengths':
                        strengths_dict[key] = full_value
                        print(f"✅ ADDED TO STRENGTHS: {key} -> {full_value[:50]}...")
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = full_value
                        print(f"⚠️ ADDED TO WEAKNESSES: {key} -> {full_value[:50]}...")
                    
                    i = j  # Move to next unprocessed element
                    continue
                else:
                    print(f"❌ No value found for key: {key}")
            else:
                print(f"   Not bold, skipping...")
        else:
            print(f"   Not in any section, skipping...")
        
        i += 1
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"📈 Strengths found: {len(strengths_dict)}")
    for key, value in strengths_dict.items():
        print(f"   💪 {key}: {value[:50]}...")
    
    print(f"📉 Weaknesses found: {len(weaknesses_dict)}")
    for key, value in weaknesses_dict.items():
        print(f"   ⚡ {key}: {value[:50]}...")
    
    return strengths_dict, weaknesses_dict

def process_folder(folder_path='.'):
    """Process all HTML files in a folder."""
    all_results = {}
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist!")
        return {}
    
    # Find all HTML files
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    
    print(f"📁 Found {len(html_files)} HTML files in '{folder_path}'")
    
    if not html_files:
        print("❌ No HTML files found!")
        return {}
    
    for filename in html_files:
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"\n{'='*60}")
            print(f"📄 Processing: {filename}")
            print(f"{'='*60}")
            
            strengths, weaknesses = extract_strengths_weaknesses(file_path)
            
            file_key = filename.replace('.html', '').replace('.htm', '')
            all_results[file_key] = {
                'strengths': strengths,
                'weaknesses': weaknesses
            }
            
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results

def save_and_print_results(results):
    """Save to JSON and print summary."""
    # Save to JSON
    with open('extracted_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    total_strengths = 0
    total_weaknesses = 0
    
    for filename, data in results.items():
        print(f"\n📄 {filename}:")
        print(f"   ✅ Strengths: {len(data['strengths'])}")
        print(f"   ⚠️ Weaknesses: {len(data['weaknesses'])}")
        
        total_strengths += len(data['strengths'])
        total_weaknesses += len(data['weaknesses'])
        
        for key, value in data['strengths'].items():
            print(f"   💪 {key}: {value[:100]}...")
        
        for key, value in data['weaknesses'].items():
            print(f"   ⚡ {key}: {value[:100]}...")
    
    print(f"\n🎯 TOTAL EXTRACTED:")
    print(f"   📈 Strengths: {total_strengths}")
    print(f"   📉 Weaknesses: {total_weaknesses}")
    print(f"   💾 Results saved to 'extracted_results.json'")

# Main execution
if __name__ == "__main__":
    folder_path = 'html_files'  # Change this to your folder path
    
    print("🚀 Starting HTML processing with detailed debugging...")
    print(f"📂 Target folder: {folder_path}")
    
    results = process_folder(folder_path)
    
    if results:
        save_and_print_results(results)
    else:
        print("❌ No results found!")
        print("\n🔍 TROUBLESHOOTING TIPS:")
        print("1. Make sure the folder 'html_files' exists")
        print("2. Make sure it contains .html files")
        print("3. Make sure the HTML files contain 'Key Rating Drivers' text")
        print("4. Check the debug output above to see what's being found")
