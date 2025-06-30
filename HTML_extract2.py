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
    for element in soup.find_all(['p', 'span']):
        if 'Key Rating Drivers' in element.get_text():
            target_section = element.find_parent()
            break
    
    if not target_section:
        print("❌ No 'Key Rating Drivers' section found!")
        return {}, {}
    
    print(f"✅ Found target section: {target_section.name}")
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    # Find all li elements and print count
    all_li_elements = target_section.find_all('li')
    print(f"📋 Found {len(all_li_elements)} <li> elements in target section")
    
    # Process all list items
    for i, li in enumerate(all_li_elements):
        text = li.get_text().strip()
        
        print(f"\n--- Processing <li> #{i+1} ---")
        print(f"Raw HTML: {li}")
        print(f"Text content: '{text}'")
        
        # Skip empty items
        if not text:
            print("⏭️  Skipping empty item")
            continue
        
        # Check for section headers (these items identify the section)
        if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
            current_section = 'strengths'
            print(f"🏷️  Found STRENGTHS section header")
            continue
        elif re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE):
            current_section = 'weaknesses'
            print(f"🏷️  Found WEAKNESSES section header")
            continue
        
        print(f"📍 Current section: {current_section}")
        
        # Extract bold text as key and remaining content as value
        if current_section:
            bold_element = li.find(['strong', 'b'])
            if bold_element:
                print(f"🔍 Found bold element: {bold_element}")
                
                # Get the bold text as key (remove colons if present)
                key = bold_element.get_text().strip().rstrip(':')
                print(f"🔑 Extracted key: '{key}'")
                
                # Get the remaining text as value
                # Remove the bold text from the full text
                value = text.replace(bold_element.get_text().strip(), '', 1).strip()
                # Clean up any leading colons or whitespace
                value = value.lstrip(':').strip()
                print(f"💭 Extracted value: '{value}'")
                
                # Add to appropriate dictionary based on current section
                if current_section == 'strengths':
                    strengths_dict[key] = value
                    print(f"✅ Added to STRENGTHS: {key} -> {value}")
                elif current_section == 'weaknesses':
                    weaknesses_dict[key] = value
                    print(f"⚠️  Added to WEAKNESSES: {key} -> {value}")
            else:
                print(f"❌ No bold element found in this <li>")
        else:
            print(f"❓ No current section set - skipping this item")
    
    print(f"\n🎯 Final Results:")
    print(f"📈 Strengths found: {len(strengths_dict)}")
    print(f"📉 Weaknesses found: {len(weaknesses_dict)}")
    
    return strengths_dict, weaknesses_dict

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
            print(f"Processing: {filename}")
            strengths, weaknesses = extract_strengths_weaknesses(file_path)
            
            file_key = filename.replace('.html', '').replace('.htm', '')
            all_results[file_key] = {
                'strengths': strengths,
                'weaknesses': weaknesses
            }
            
        except Exception as e:
            print(f"Error with {filename}: {e}")
    
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
        print(f"   ⚠️  Weaknesses: {len(data['weaknesses'])}")
        
        # Print actual content
        for key, value in data['strengths'].items():
            print(f"   💪 {key}: {value[:100]}...")
        
        for key, value in data['weaknesses'].items():
            print(f"   ⚡ {key}: {value[:100]}...")
    
    print(f"\n💾 Full results saved to 'extracted_results.json'")

# Main execution
if __name__ == "__main__":
    folder_path = '.'  # Current folder - change this to your folder path
    
    print("🔍 Processing all HTML files in folder...")
    results = process_folder(folder_path)
    
    if results:
        save_and_print_results(results)
    else:
        print("❌ No HTML files found or processed successfully!")
