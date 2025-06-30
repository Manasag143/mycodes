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
    
    print(f"✅ Found target section")
    
    # Get all text from the target section
    full_text = target_section.get_text()
    print(f"\n📄 FULL TEXT FROM TARGET SECTION:")
    print("=" * 60)
    print(full_text)
    print("=" * 60)
    
    # Find all bold elements in the entire target section
    all_bold_elements = target_section.find_all(['strong', 'b'])
    print(f"\n💪 ALL BOLD ELEMENTS FOUND: {len(all_bold_elements)}")
    print("=" * 60)
    
    for i, bold in enumerate(all_bold_elements):
        bold_text = bold.get_text().strip()
        print(f"{i+1}. Bold text: '{bold_text}'")
    
    # Now let's look for Strengths and Weaknesses sections in the text
    print(f"\n🔍 LOOKING FOR STRENGTHS AND WEAKNESSES SECTIONS:")
    print("=" * 60)
    
    # Split text by lines to analyze
    lines = full_text.split('\n')
    current_section = None
    
    print("All lines from target section:")
    for i, line in enumerate(lines):
        line = line.strip()
        if line:  # Only print non-empty lines
            print(f"Line {i+1}: '{line}'")
            
            # Check if this line contains Strengths or Weaknesses
            if re.search(r'\bStrengths?\s*:?', line, re.IGNORECASE):
                current_section = 'strengths'
                print(f"    🎯 FOUND STRENGTHS SECTION!")
            elif re.search(r'\bWeakness(es)?\s*:?', line, re.IGNORECASE):
                current_section = 'weaknesses'
                print(f"    🎯 FOUND WEAKNESSES SECTION!")
            elif current_section:
                print(f"    📍 In {current_section} section")
    
    # Alternative approach: Find all elements that contain bold text
    print(f"\n🔍 ELEMENTS CONTAINING BOLD TEXT:")
    print("=" * 60)
    
    # Find all elements that have bold children
    elements_with_bold = []
    for element in target_section.find_all():
        if element.find(['strong', 'b']):
            elements_with_bold.append(element)
    
    print(f"Found {len(elements_with_bold)} elements containing bold text:")
    for i, elem in enumerate(elements_with_bold):
        print(f"\nElement {i+1}:")
        print(f"  Tag: <{elem.name}>")
        print(f"  Full text: '{elem.get_text().strip()}'")
        
        # Find bold elements within this element
        bold_in_elem = elem.find_all(['strong', 'b'])
        for j, bold in enumerate(bold_in_elem):
            print(f"    Bold {j+1}: '{bold.get_text().strip()}'")
    
    return {}, {}

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
            print(f"\n{'='*80}")
            print(f"🔍 Processing: {filename}")
            print(f"{'='*80}")
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
