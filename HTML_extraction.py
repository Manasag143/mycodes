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
    
    # Get all elements (paragraphs and list items) in order
    all_elements = target_section.find_all(['p', 'li'])
    
    for element in all_elements:
        text = element.get_text().strip()
        
        # Skip empty elements
        if not text:
            continue
        
        # Check for section headers in paragraphs
        if element.name == 'p':
            if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
                current_section = 'strengths'
                continue
            elif re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE):
                current_section = 'weaknesses'
                continue
        
        # Process list items within a section
        if element.name == 'li' and current_section:
            # Look for spans with bold styling
            spans = element.find_all('span')
            bold_spans = []
            regular_spans = []
            
            for span in spans:
                style = span.get('style', '')
                span_text = span.get_text().strip()
                
                # Check if span has bold styling
                if 'font-weight' in style and ('bold' in style or '700' in style):
                    bold_spans.append(span_text)
                else:
                    regular_spans.append(span_text)
            
            # Also check for strong/b tags as fallback
            strong_tags = element.find_all(['strong', 'b'])
            if strong_tags and not bold_spans:
                bold_spans = [tag.get_text().strip() for tag in strong_tags]
            
            if bold_spans:
                # Build the key from bold spans
                key = ""
                for bold_text in bold_spans:
                    if bold_text == ':':
                        break
                    elif bold_text.endswith(':'):
                        key += bold_text.rstrip(':')
                        break
                    else:
                        key += bold_text
                        # Add space if this isn't the last bold span and next isn't a colon
                        if bold_spans.index(bold_text) < len(bold_spans) - 1:
                            next_span = bold_spans[bold_spans.index(bold_text) + 1]
                            if next_span != ':':
                                key += " "
                
                # Get value from regular spans
                value = ' '.join(regular_spans).strip()
                
                # Store in appropriate dictionary
                if key and current_section:
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
