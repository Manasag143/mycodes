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
        return {}, {}
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    # Process all list items
    for li in target_section.find_all('li'):
        text = li.get_text().strip()
        
        # Check for section headers
        if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
            current_section = 'strengths'
            continue
        elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
            current_section = 'weaknesses'
            continue
        
        # Extract bold text as key and remaining text as value
        if current_section:
            bold_element = li.find(['strong', 'b'])
            if bold_element:
                key = bold_element.get_text().strip().rstrip(':')
                value = text.replace(bold_element.get_text().strip(), '', 1).strip().lstrip(':').strip()
                
                if current_section == 'strengths':
                    strengths_dict[key] = value
                elif current_section == 'weaknesses':
                    weaknesses_dict[key] = value
    
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
