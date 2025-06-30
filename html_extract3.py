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
    print(f"📄 Found {len(all_p_elements)} <p> elements in target section")
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    print(f"\n🔍 PROCESSING ELEMENTS:")
    print("=" * 60)
    
    i = 0
    while i < len(all_p_elements):
        p = all_p_elements[i]
        text = p.get_text().strip()
        
        if not text:  # Skip empty paragraphs
            i += 1
            continue
        
        print(f"\nP{i+1}: '{text[:50]}...' ")
        
        # Check if this is a section header (Strengths or Weaknesses)
        if re.search(r'\bStrengths?\s*:?\s*$', text, re.IGNORECASE):
            current_section = 'strengths'
            print(f"🎯 Found STRENGTHS section header")
            i += 1
            continue
        elif re.search(r'\bWeakness(es)?\s*:?\s*$', text, re.IGNORECASE):
            current_section = 'weaknesses'
            print(f"🎯 Found WEAKNESSES section header")
            i += 1
            continue
        
        # If we're in a section, process key-value pairs
        if current_section:
            # Check if this element is bold (potential key)
            is_bold = is_element_bold(p)
            
            if is_bold and text:  # This is a key
                key = text.rstrip(':').strip()
                print(f"🔑 Found KEY: '{key}' (bold)")
                
                # Collect all following non-bold paragraphs as the value
                value_parts = []
                j = i + 1
                
                while j < len(all_p_elements):
                    next_p = all_p_elements[j]
                    next_text = next_p.get_text().strip()
                    
                    if not next_text:  # Skip empty paragraphs
                        j += 1
                        continue
                    
                    next_is_bold = is_element_bold(next_p)
                    
                    # If we hit another bold element, it's the next key
                    if next_is_bold:
                        print(f"🔑 Next bold element found, stopping value collection")
                        break
                    
                    # Check if we hit a new section
                    if re.search(r'\b(Strengths?|Weakness(es)?)\s*:?\s*$', next_text, re.IGNORECASE):
                        print(f"🎯 New section found, stopping value collection")
                        break
                    
                    # Add this paragraph to the value
                    value_parts.append(next_text)
                    print(f"💭 Added to value: '{next_text[:30]}...'")
                    j += 1
                
                # Combine all value parts
                if value_parts:
                    full_value = ' '.join(value_parts).strip()
                    print(f"📝 Complete VALUE: '{full_value[:50]}...'")
                    
                    if current_section == 'strengths':
                        strengths_dict[key] = full_value
                        print(f"✅ Added to STRENGTHS: {key}")
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = full_value
                        print(f"⚠️  Added to WEAKNESSES: {key}")
                    
                    i = j  # Move to the next unprocessed element
                    continue
                else:
                    print(f"❌ No value found for key: {key}")
        
        i += 1
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"📈 Strengths found: {len(strengths_dict)}")
    for key, value in strengths_dict.items():
        print(f"   💪 {key}: {value[:50]}...")
    
    print(f"📉 Weaknesses found: {len(weaknesses_dict)}")
    for key, value in weaknesses_dict.items():
        print(f"   ⚡ {key}: {value[:50]}...")
    
    return strengths_dict, weaknesses_dict

def is_element_bold(element):
    """Check if an element is bold based on various methods."""
    
    # Method 1: Check inline style for font-weight: bold
    style = element.get('style', '')
    if style:
        # Look for font-weight: bold in various formats
        if re.search(r'font-weight\s*:\s*bold', style, re.IGNORECASE):
            return True
        # Also check for font-weight: 700 or higher (bold)
        weight_match = re.search(r'font-weight\s*:\s*(\d+)', style, re.IGNORECASE)
        if weight_match and int(weight_match.group(1)) >= 700:
            return True
    
    # Method 2: Check if element is wrapped in <b> or <strong> tags
    if element.find('b') or element.find('strong'):
        return True
    
    # Method 3: Check if the element itself is <b> or <strong>
    if element.name in ['b', 'strong']:
        return True
    
    # Method 4: Check parent elements for bold styling
    parent = element.parent
    while parent and parent.name != 'body':
        parent_style = parent.get('style', '')
        if re.search(r'font-weight\s*:\s*bold', parent_style, re.IGNORECASE):
            return True
        if parent.name in ['b', 'strong']:
            return True
        parent = parent.parent
    
    # Method 5: Check for class names that might indicate bold text
    class_names = element.get('class', [])
    bold_classes = ['bold', 'font-bold', 'fw-bold', 'text-bold']
    if any(cls in ' '.join(class_names).lower() for cls in bold_classes):
        return True
    
    return False

def process_folder(folder_path='.'):
    """Process all HTML files in a folder."""
    all_results = {}
    
    # Find all HTML files
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    
    print(f"Found {len(html_files)} HTML files")
    
    # Process each file
    for filename in html_files:
        file_path = os.path.join(folder_path, filename)
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {filename}")
            print(f"{'='*50}")
            
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
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*50}")
    
    total_strengths = 0
    total_weaknesses = 0
    
    for filename, data in results.items():
        print(f"\n📄 {filename}:")
        print(f"   ✅ Strengths: {len(data['strengths'])}")
        print(f"   ⚠️  Weaknesses: {len(data['weaknesses'])}")
        
        total_strengths += len(data['strengths'])
        total_weaknesses += len(data['weaknesses'])
        
        # Print actual content with better formatting
        if data['strengths']:
            print(f"   📈 STRENGTHS:")
            for key, value in data['strengths'].items():
                print(f"      💪 {key}: {value[:80]}...")
        
        if data['weaknesses']:
            print(f"   📉 WEAKNESSES:")
            for key, value in data['weaknesses'].items():
                print(f"      ⚡ {key}: {value[:80]}...")
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   📁 Files processed: {len(results)}")
    print(f"   💪 Total strengths: {total_strengths}")
    print(f"   ⚡ Total weaknesses: {total_weaknesses}")
    print(f"   💾 Results saved to 'extracted_results.json'")

# Main execution
if __name__ == "__main__":
    folder_path = 'html_files'  # Change this to your folder path
    
    print("🚀 Starting HTML processing...")
    print(f"📂 Target folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist!")
        print("Please create the folder and add your HTML files, or change the folder_path variable.")
    else:
        results = process_folder(folder_path)
        
        if results:
            save_and_print_results(results)
        else:
            print("❌ No HTML files found or processed successfully!")
            print("Make sure your HTML files are in the correct folder and contain 'Key Rating Drivers' sections.")
