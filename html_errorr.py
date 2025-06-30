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
    
    # Get all text elements after finding the target section
    all_elements = target_section.find_all(['p', 'ul', 'li'])
    
    for element in all_elements:
        text = element.get_text().strip()
        
        # Check for section headers in paragraphs
        if element.name == 'p':
            if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
                current_section = 'strengths'
                continue
            elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
                current_section = 'weaknesses'
                continue
        
        # Process list items
        if element.name == 'li' and current_section:
            # Find bold elements (strong or b tags)
            bold_elements = element.find_all(['strong', 'b'])
            
            if bold_elements:
                # Get the first bold element as the key
                key_element = bold_elements[0]
                key = key_element.get_text().strip().rstrip(':')
                
                # Get the full text and remove the key part to get the value
                full_text = element.get_text().strip()
                
                # Find the position where the key ends and extract the value
                key_text = key_element.get_text().strip()
                if key_text in full_text:
                    # Split at the key and take everything after
                    parts = full_text.split(key_text, 1)
                    if len(parts) > 1:
                        value = parts[1].strip().lstrip(':').strip()
                    else:
                        value = ""
                else:
                    value = full_text.replace(key_text, '', 1).strip().lstrip(':').strip()
                
                # Store in appropriate dictionary
                if current_section == 'strengths':
                    strengths_dict[key] = value
                elif current_section == 'weaknesses':
                    weaknesses_dict[key] = value
    
    return strengths_dict, weaknesses_dict

def extract_from_html_content(html_content):
    """Extract strengths and weaknesses from HTML content string."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
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
    
    # Get all text elements after finding the target section
    all_elements = target_section.find_all(['p', 'ul', 'li'])
    
    for element in all_elements:
        text = element.get_text().strip()
        
        # Check for section headers in paragraphs
        if element.name == 'p':
            if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
                current_section = 'strengths'
                continue
            elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
                current_section = 'weaknesses'
                continue
        
        # Process list items
        if element.name == 'li' and current_section:
            # Find bold elements (strong or b tags)
            bold_elements = element.find_all(['strong', 'b'])
            
            if bold_elements:
                # Get the first bold element as the key
                key_element = bold_elements[0]
                key = key_element.get_text().strip().rstrip(':')
                
                # Get the full text and remove the key part to get the value
                full_text = element.get_text().strip()
                
                # Find the position where the key ends and extract the value
                key_text = key_element.get_text().strip()
                if key_text in full_text:
                    # Split at the key and take everything after
                    parts = full_text.split(key_text, 1)
                    if len(parts) > 1:
                        value = parts[1].strip().lstrip(':').strip()
                    else:
                        value = ""
                else:
                    value = full_text.replace(key_text, '', 1).strip().lstrip(':').strip()
                
                # Store in appropriate dictionary
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
        print("\n   STRENGTHS:")
        for key, value in data['strengths'].items():
            print(f"   💪 {key}: {value[:100]}...")
        
        print("\n   WEAKNESSES:")
        for key, value in data['weaknesses'].items():
            print(f"   ⚡ {key}: {value[:100]}...")
    
    print(f"\n💾 Full results saved to 'extracted_results.json'")

# Example usage with the provided HTML content
def demo_with_provided_html():
    """Demo function to show how it works with your HTML content."""
    # Your HTML content would go here
    html_content = """
    <!-- Your HTML content here -->
    """
    
    strengths, weaknesses = extract_from_html_content(html_content)
    
    print("EXTRACTED STRENGTHS:")
    for key, value in strengths.items():
        print(f"Key: {key}")
        print(f"Value: {value}")
        print("-" * 50)
    
    print("\nEXTRACTED WEAKNESSES:")
    for key, value in weaknesses.items():
        print(f"Key: {key}")
        print(f"Value: {value}")
        print("-" * 50)
    
    return {'strengths': strengths, 'weaknesses': weaknesses}

# Main execution
if __name__ == "__main__":
    folder_path = '.'  # Current folder - change this to your folder path
    
    print("🔍 Processing all HTML files in folder...")
    results = process_folder(folder_path)
    
    if results:
        save_and_print_results(results)
    else:
        print("❌ No HTML files found or processed successfully!")
        
    # Uncomment to test with specific HTML content
    # demo_with_provided_html()
