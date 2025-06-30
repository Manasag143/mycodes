from bs4 import BeautifulSoup
import re
import os
import json

def extract_strengths_weaknesses(html_file_path):
    """Extract strengths and weaknesses from a single HTML file."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    return extract_from_soup(soup)

def extract_from_html_content(html_content):
    """Extract strengths and weaknesses from HTML content string."""
    soup = BeautifulSoup(html_content, 'html.parser')
    return extract_from_soup(soup)

def is_bold_element(element):
    """Check if element has bold styling using various methods."""
    # Method 1: Check style attribute for font-weight
    style = element.get('style', '')
    if 'font-weight' in style and ('bold' in style or '700' in style or 'bolder' in style):
        return True
    
    # Method 2: Check for bold tags (strong, b)
    if element.name in ['strong', 'b']:
        return True
    
    # Method 3: Check class attribute
    classes = element.get('class', [])
    for cls in classes:
        if 'bold' in cls.lower():
            return True
    
    # Method 4: Check parent elements for bold styling
    parent = element.parent
    if parent:
        parent_style = parent.get('style', '')
        if 'font-weight' in parent_style and ('bold' in parent_style or '700' in parent_style):
            return True
    
    return False

def extract_from_soup(soup):
    """Extract strengths and weaknesses from BeautifulSoup object."""
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
    
    # Get all elements in the target section (paragraphs, lists, list items)
    all_elements = target_section.find_all(['p', 'ul', 'li'])
    
    for element in all_elements:
        text = element.get_text().strip()
        
        # Skip empty elements
        if not text:
            continue
        
        # Check if this is a section header (Strengths/Weaknesses)
        if element.name == 'p':
            if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
                current_section = 'strengths'
                continue
            elif re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE):
                current_section = 'weaknesses'
                continue
        
        # Process list items within a section
        if element.name == 'li' and current_section:
            # Find all spans in this list item
            spans = element.find_all('span')
            bold_spans = []
            regular_spans = []
            
            for span in spans:
                if is_bold_element(span):
                    bold_spans.append(span)
                else:
                    regular_spans.append(span)
            
            if bold_spans:
                # The first bold span should be the key
                # Check if it ends with ':' or if the next bold span is ':'
                key_text = ""
                value_spans = []
                
                # Build the key from bold spans
                for i, bold_span in enumerate(bold_spans):
                    span_text = bold_span.get_text().strip()
                    if span_text == ':' and i > 0:
                        # This is just a colon separator, don't add to key
                        break
                    elif span_text.endswith(':'):
                        # Key ends here
                        key_text += span_text.rstrip(':')
                        break
                    else:
                        # Continue building key
                        key_text += span_text
                        if i < len(bold_spans) - 1 and bold_spans[i+1].get_text().strip() != ':':
                            key_text += " "
                
                # Get all non-bold spans as value
                value_parts = []
                for span in regular_spans:
                    span_text = span.get_text().strip()
                    if span_text:
                        value_parts.append(span_text)
                
                # Also check if there are any remaining bold spans after the colon
                colon_found = False
                for bold_span in bold_spans:
                    if ':' in bold_span.get_text():
                        colon_found = True
                        # Check if this span has content after the colon
                        span_text = bold_span.get_text()
                        if ':' in span_text:
                            after_colon = span_text.split(':', 1)[1].strip()
                            if after_colon:
                                value_parts.insert(0, after_colon)
                        break
                
                key = key_text.strip()
                value = ' '.join(value_parts).strip()
                
                # Store in appropriate dictionary
                if key and current_section:
                    if current_section == 'strengths':
                        strengths_dict[key] = value
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = value
    
    return strengths_dict, weaknesses_dict

def debug_html_structure(html_file_path):
    """Debug function to understand HTML structure."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    # Find the target section
    target_section = None
    for element in soup.find_all(['p', 'span']):
        if 'Key Rating Drivers' in element.get_text():
            target_section = element.find_parent()
            break
    
    if target_section:
        print("Found target section!")
        
        # Look for Strengths section
        strengths_found = False
        for element in target_section.find_all(['p']):
            text = element.get_text().strip()
            if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
                print(f"\nFound Strengths section: {text}")
                strengths_found = True
                break
        
        if strengths_found:
            # Find all list items after strengths
            all_elements = target_section.find_all(['p', 'ul', 'li'])
            in_strengths = False
            
            for element in all_elements:
                text = element.get_text().strip()
                
                if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
                    in_strengths = True
                    print(f"\n--- ENTERING STRENGTHS SECTION ---")
                    continue
                elif re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE):
                    in_strengths = False
                    print(f"\n--- ENTERING WEAKNESSES SECTION ---")
                    continue
                
                if in_strengths and element.name == 'li':
                    print(f"\n📋 List item found:")
                    print(f"  Text: {text[:150]}...")
                    
                    # Check for spans
                    spans = element.find_all('span')
                    print(f"  Total spans: {len(spans)}")
                    
                    bold_spans = []
                    regular_spans = []
                    
                    for i, span in enumerate(spans):
                        style = span.get('style', '')
                        span_text = span.get_text().strip()
                        is_bold = 'font-weight' in style and 'bold' in style
                        
                        if is_bold:
                            bold_spans.append(span_text)
                        else:
                            regular_spans.append(span_text)
                        
                        print(f"    Span {i+1}: '{span_text}' (Bold: {is_bold})")
                    
                    print(f"  📝 Bold spans: {bold_spans}")
                    print(f"  📄 Regular spans: {regular_spans}")
                    
                    # Show potential key extraction
                    if bold_spans:
                        key_candidate = ""
                        for bold_text in bold_spans:
                            if bold_text == ':':
                                break
                            elif bold_text.endswith(':'):
                                key_candidate += bold_text.rstrip(':')
                                break
                            else:
                                key_candidate += bold_text + " "
                        
                        value_candidate = ' '.join(regular_spans)
                        print(f"  🔑 Extracted Key: '{key_candidate.strip()}'")
                        print(f"  💡 Extracted Value: '{value_candidate[:100]}...'")
    else:
        print("Target section not found!")

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
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for filename, data in results.items():
        print(f"\n📄 {filename}:")
        print(f"   ✅ Strengths: {len(data['strengths'])}")
        print(f"   ⚠️  Weaknesses: {len(data['weaknesses'])}")
        
        # Print actual content
        if data['strengths']:
            print("\n   STRENGTHS:")
            for i, (key, value) in enumerate(data['strengths'].items(), 1):
                print(f"   {i}. Key: {key}")
                print(f"      Value: {value[:200]}...")
                print()
        
        if data['weaknesses']:
            print("\n   WEAKNESSES:")
            for i, (key, value) in enumerate(data['weaknesses'].items(), 1):
                print(f"   {i}. Key: {key}")
                print(f"      Value: {value[:200]}...")
                print()
    
    print(f"\n💾 Full results saved to 'extracted_results.json'")

# Main execution
if __name__ == "__main__":
    folder_path = '.'  # Current folder - change this to your folder path
    
    # Debug the actual HTML structure
    html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
    if html_files:
        print("🔍 Debugging actual HTML structure...")
        debug_html_structure(os.path.join(folder_path, html_files[0]))
        print("\n" + "="*80 + "\n")
    
    print("🔍 Processing all HTML files in folder...")
    results = process_folder(folder_path)
    
    if results:
        save_and_print_results(results)
    else:
        print("❌ No HTML files found or processed successfully!")
