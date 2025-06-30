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
    
    # Look for the specific text pattern in the HTML
    for element in soup.find_all(text=re.compile(r'Key Rating Drivers.*Detailed Description', re.IGNORECASE | re.DOTALL)):
        target_section = element.find_parent()
        break
    
    # Alternative search method if first doesn't work
    if not target_section:
        for element in soup.find_all(['p', 'span']):
            if element.get_text() and 'Key Rating Drivers' in element.get_text():
                # Find the containing div or table cell
                target_section = element.find_parent(['div', 'td', 'table'])
                break
    
    if not target_section:
        print("Could not find Key Rating Drivers section")
        return {}, {}
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    # Get all text content and process it
    full_text = target_section.get_text()
    
    # Find strengths and weaknesses sections
    strengths_match = re.search(r'Strengths?\s*:', full_text, re.IGNORECASE)
    weakness_match = re.search(r'Weakness(es)?\s*:', full_text, re.IGNORECASE)
    
    if not strengths_match:
        print("Could not find Strengths section")
        return {}, {}
    
    # Process list items within the target section
    for li in target_section.find_all('li'):
        text = li.get_text().strip()
        
        # Skip empty items
        if not text:
            continue
            
        # Check for section headers
        if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
            current_section = 'strengths'
            continue
        elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
            current_section = 'weaknesses'
            continue
        
        # Extract content for current section
        if current_section:
            # Look for bold text as key
            bold_elements = li.find_all(['strong', 'b'])
            
            if bold_elements:
                # Get the first bold element as the key
                key_element = bold_elements[0]
                key = key_element.get_text().strip().rstrip(':')
                
                # Get the full text and remove the key part to get the value
                full_li_text = li.get_text().strip()
                
                # Find where the key ends and value begins
                key_text = key_element.get_text().strip()
                
                # Split the text and get everything after the key
                if ':' in full_li_text:
                    # Find the first colon after the key
                    colon_index = full_li_text.find(':', full_li_text.find(key_text))
                    if colon_index != -1:
                        value = full_li_text[colon_index + 1:].strip()
                    else:
                        value = full_li_text.replace(key_text, '', 1).strip()
                else:
                    value = full_li_text.replace(key_text, '', 1).strip()
                
                # Clean up the value
                value = re.sub(r'^[:\s]+', '', value)  # Remove leading colons and spaces
                
                if key and value:  # Only add if both key and value exist
                    if current_section == 'strengths':
                        strengths_dict[key] = value
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = value
    
    # If no list items found, try parsing paragraphs
    if not strengths_dict and not weaknesses_dict:
        print("No list items found, trying paragraph parsing...")
        strengths_dict, weaknesses_dict = parse_paragraphs(target_section)
    
    return strengths_dict, weaknesses_dict

def parse_paragraphs(target_section):
    """Alternative parsing method using paragraphs instead of list items."""
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    # Get all paragraphs and spans
    elements = target_section.find_all(['p', 'span', 'div'])
    
    for element in elements:
        text = element.get_text().strip()
        
        if not text:
            continue
            
        # Check for section headers
        if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
            current_section = 'strengths'
            continue
        elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
            current_section = 'weaknesses'
            continue
        
        # Look for bold text within the element
        if current_section:
            bold_elements = element.find_all(['strong', 'b'])
            
            for bold_element in bold_elements:
                key = bold_element.get_text().strip().rstrip(':')
                
                # Get the parent text and extract value
                parent_text = element.get_text().strip()
                key_text = bold_element.get_text().strip()
                
                # Find the value after the key
                key_index = parent_text.find(key_text)
                if key_index != -1:
                    value_start = key_index + len(key_text)
                    value = parent_text[value_start:].strip()
                    value = re.sub(r'^[:\s]+', '', value)  # Remove leading colons and spaces
                    
                    if key and value:
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
            
            print(f"  - Found {len(strengths)} strengths and {len(weaknesses)} weaknesses")
            
        except Exception as e:
            print(f"Error with {filename}: {e}")
    
    return all_results

def save_and_print_results(results):
    """Save to JSON and print summary."""
    # Save to JSON
    with open('extracted_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    
    for filename, data in results.items():
        print(f"\n📄 FILE: {filename}")
        print(f"{'='*50}")
        
        print(f"\n💪 STRENGTHS ({len(data['strengths'])}):")
        print("-" * 40)
        for i, (key, value) in enumerate(data['strengths'].items(), 1):
            print(f"{i}. Key: {key}")
            print(f"   Value: {value}")
            print()
        
        print(f"\n⚠️  WEAKNESSES ({len(data['weaknesses'])}):")
        print("-" * 40)
        for i, (key, value) in enumerate(data['weaknesses'].items(), 1):
            print(f"{i}. Key: {key}")
            print(f"   Value: {value}")
            print()
    
    print(f"\n💾 Full results saved to 'extracted_results.json'")

def test_with_provided_html():
    """Test function specifically for the provided HTML content."""
    html_content = """
    <!-- Your HTML content would go here -->
    """
    
    soup = BeautifulSoup(html_content, 'html.parser')
    print("Testing with provided HTML content...")
    
    # Debug: Find all elements that might contain our target text
    for element in soup.find_all(text=re.compile(r'Key Rating Drivers', re.IGNORECASE)):
        print(f"Found 'Key Rating Drivers' in: {element.parent.name}")
        print(f"Parent text: {element.parent.get_text()[:100]}...")

# Main execution
if __name__ == "__main__":
    folder_path = '.'  # Current folder - change this to your folder path
    
    print("🔍 Processing all HTML files in folder...")
    results = process_folder(folder_path)
    
    if results:
        save_and_print_results(results)
    else:
        print("❌ No HTML files found or processed successfully!")
        
    # Uncomment the line below to test with specific HTML content
    # test_with_provided_html()
