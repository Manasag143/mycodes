from bs4 import BeautifulSoup
import re
import os
import json

def extract_strengths_weaknesses(html_file_path):
    """Extract strengths and weaknesses from a single HTML file."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    # Find the "Key Rating Drivers" section
    target_section = None
    for element in soup.find_all(['p', 'span', 'div', 'h1', 'h2', 'h3']):
        if 'Key Rating Drivers' in element.get_text():
            target_section = element.find_parent()
            break
    
    if not target_section:
        print("❌ Could not find 'Key Rating Drivers' section")
        return {}, {}
    
    print("✅ Found target section")
    
    # Get all paragraph elements in the target section
    all_p_elements = target_section.find_all('p')
    print(f"📄 Found {len(all_p_elements)} <p> elements")
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    i = 0
    while i < len(all_p_elements):
        p = all_p_elements[i]
        text = p.get_text().strip()
        
        if not text:  # Skip empty paragraphs
            i += 1
            continue
        
        # Check if this is a section header
        if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
            current_section = 'strengths'
            print(f"🎯 Found STRENGTHS section")
            i += 1
            continue
        elif re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE):
            current_section = 'weaknesses' 
            print(f"🎯 Found WEAKNESSES section")
            i += 1
            continue
        
        # If we're in a section, check if this is a bold element (key)
        if current_section:
            style = p.get('style', '')
            is_bold = 'font-weight' in style.lower() and 'bold' in style.lower()
            
            if is_bold:  # This is a key
                key = text.rstrip(':').strip()
                print(f"🔑 Found KEY: '{key}'")
                
                # Collect all following non-bold paragraphs as value
                value_parts = []
                j = i + 1
                
                while j < len(all_p_elements):
                    next_p = all_p_elements[j]
                    next_text = next_p.get_text().strip()
                    
                    if not next_text:  # Skip empty
                        j += 1
                        continue
                    
                    # Check if next element is also bold (another key)
                    next_style = next_p.get('style', '')
                    next_is_bold = 'font-weight' in next_style.lower() and 'bold' in next_style.lower()
                    
                    if next_is_bold:  # Hit another key, stop
                        break
                    
                    # Check if we hit a new section
                    if re.search(r'\b(Strengths?|Weakness(es)?)\s*:?', next_text, re.IGNORECASE):
                        break
                    
                    # Add this text to value
                    value_parts.append(next_text)
                    j += 1
                
                # Save the key-value pair
                if value_parts:
                    full_value = ' '.join(value_parts).strip()
                    
                    if current_section == 'strengths':
                        strengths_dict[key] = full_value
                        print(f"✅ Added to STRENGTHS: {key}")
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = full_value
                        print(f"⚠️ Added to WEAKNESSES: {key}")
                    
                    i = j  # Move to next unprocessed element
                    continue
        
        i += 1
    
    print(f"\n📈 Strengths found: {len(strengths_dict)}")
    print(f"📉 Weaknesses found: {len(weaknesses_dict)}")
    
    return strengths_dict, weaknesses_dict

def process_folder(folder_path='.'):
    """Process all HTML files in a folder."""
    all_results = {}
    
    # Find all HTML files
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    
    print(f"Found {len(html_files)} HTML files")
    
    for filename in html_files:
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"\nProcessing: {filename}")
            strengths, weaknesses = extract_strengths_weaknesses(file_path)
            
            file_key = filename.replace('.html', '').replace('.htm', '')
            all_results[file_key] = {
                'strengths': strengths,
                'weaknesses': weaknesses
            }
            
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")
    
    return all_results

def save_and_print_results(results):
    """Save to JSON and print summary."""
    # Save to JSON
    with open('extracted_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    
    for filename, data in results.items():
        print(f"\n📄 {filename}:")
        print(f"   ✅ Strengths: {len(data['strengths'])}")
        print(f"   ⚠️ Weaknesses: {len(data['weaknesses'])}")
        
        for key, value in data['strengths'].items():
            print(f"   💪 {key}: {value[:100]}...")
        
        for key, value in data['weaknesses'].items():
            print(f"   ⚡ {key}: {value[:100]}...")
    
    print(f"\n💾 Results saved to 'extracted_results.json'")

# Main execution
if __name__ == "__main__":
    folder_path = 'html_files'  # Change this to your folder path
    
    print("🚀 Processing HTML files...")
    results = process_folder(folder_path)
    
    if results:
        save_and_print_results(results)
    else:
        print("❌ No HTML files found or processed!")
