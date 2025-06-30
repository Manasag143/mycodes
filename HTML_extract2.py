from bs4 import BeautifulSoup
import re
import json
import os

def extract_rating_drivers(html_file_path):
    """
    Simple pipeline to extract strengths and weaknesses from Rating document.
    Returns dictionary with sub-topics as keys and content as values.
    """
    
    print(f"📄 Processing: {os.path.basename(html_file_path)}")
    
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    # Step 1: Find "Key Rating Drivers & Detailed Description" section
    key_section_found = False
    rating_drivers_section = None
    
    for element in soup.find_all(['p', 'span']):
        text = element.get_text().strip()
        if 'Key Rating Drivers' in text and 'Detailed Description' in text:
            rating_drivers_section = element.find_parent(['div', 'td', 'table'])
            key_section_found = True
            print("✅ Found 'Key Rating Drivers & Detailed Description' section")
            break
    
    if not rating_drivers_section:
        print("❌ Could not find 'Key Rating Drivers & Detailed Description' section")
        return {}
    
    # Step 2: Extract all content from this section
    result_dict = {}
    current_section = None  # 'strengths' or 'weaknesses'
    processing_started = False
    
    # Get all elements in the rating drivers section
    all_elements = rating_drivers_section.find_all(['p', 'ul', 'li'])
    
    print(f"🔍 Found {len(all_elements)} elements to analyze")
    
    for i, element in enumerate(all_elements):
        element_text = element.get_text().strip()
        
        # Skip empty elements
        if not element_text or element_text.isspace():
            continue
            
        print(f"   Analyzing element {i}: {element.name} - '{element_text[:50]}...'")
        
        # Step 3: Start processing after finding Key Rating Drivers
        if not processing_started:
            if 'Key Rating Drivers' in element_text:
                processing_started = True
                print("🚀 Starting extraction after Key Rating Drivers header")
            continue
        
        # Step 4: Identify section headers
        if element.name == 'p':
            # Look for "Strengths:" header (more specific pattern)
            strengths_spans = element.find_all('span')
            for span in strengths_spans:
                span_text = span.get_text().strip()
                if re.match(r'^Strengths?:?$', span_text, re.IGNORECASE):
                    current_section = 'strengths'
                    print("💪 Found Strengths section")
                    break
            
            # Look for "Weakness:" header (more specific pattern)
            for span in strengths_spans:
                span_text = span.get_text().strip()
                if re.match(r'^Weakness(es)?:?$', span_text, re.IGNORECASE):
                    current_section = 'weaknesses'
                    print("⚠️ Found Weaknesses section")
                    break
        
        # Step 5: Extract key-value pairs from list items
        elif element.name == 'li' and current_section:
            print(f"      🔎 Processing {current_section} list item...")
            
            # Find all spans in this list item
            spans = element.find_all('span')
            
            if not spans:
                print("      ❌ No spans found in this list item")
                continue
            
            key = None
            value_parts = []
            found_key = False
            
            for span in spans:
                span_style = span.get('style', '')
                span_text = span.get_text().strip()
                
                print(f"         Span: '{span_text}' | Bold: {'font-weight:bold' in span_style}")
                
                # If it's a bold span and not just a colon, it's our key
                if 'font-weight:bold' in span_style and span_text and span_text != ':':
                    if not found_key:  # Take the first bold text as key
                        key = span_text.rstrip(':').strip()
                        found_key = True
                        print(f"         🔑 Found key: '{key}'")
                
                # If it's not bold and not empty, it's part of the value
                elif span_text and 'font-weight:bold' not in span_style:
                    value_parts.append(span_text)
                    print(f"         📝 Added to value: '{span_text[:30]}...'")
            
            # Clean and combine the value
            if key and value_parts:
                value = ' '.join(value_parts).strip()
                # Remove any leading colons or extra spaces
                value = re.sub(r'^[:\s]+', '', value)
                value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                
                if value:  # Only add if we have actual content
                    result_dict[key] = value
                    print(f"      ✅ Added: {key} = {value[:60]}...")
                else:
                    print(f"      ❌ Empty value for key: {key}")
            else:
                print(f"      ❌ Missing key or value. Key: '{key}', Value parts: {len(value_parts)}")
        
        # Step 6: More specific stop conditions - only stop at major section headers
        elif element.name == 'p' and processing_started:
            # Only stop at very specific major section headers
            major_sections = [
                'Liquidity:', 'Outlook:', 'Rating sensitivity factors', 'Analytical Approach',
                'About the Company', 'Key Financial Indicators'
            ]
            
            if any(section in element_text for section in major_sections):
                print(f"🛑 Reached major section: '{element_text[:50]}...' - Stopping extraction")
                break
    
    print(f"📊 Extraction complete! Found {len(result_dict)} items")
    return result_dict

def debug_html_structure(html_file_path):
    """
    Debug function to show the HTML structure around the Key Rating Drivers section
    """
    print("🔧 DEBUG MODE - Analyzing HTML Structure")
    print("="*60)
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    # Find the Key Rating Drivers section
    for element in soup.find_all(['p', 'span']):
        text = element.get_text().strip()
        if 'Key Rating Drivers' in text and 'Detailed Description' in text:
            container = element.find_parent(['div', 'td', 'table'])
            print(f"✅ Found container: {container.name}")
            
            # Show the next 20 elements after this
            all_elements = container.find_all(['p', 'ul', 'li'])
            start_index = 0
            
            # Find where we are in the list
            for i, elem in enumerate(all_elements):
                if 'Key Rating Drivers' in elem.get_text():
                    start_index = i
                    break
            
            print(f"\n📋 Next 20 elements after Key Rating Drivers:")
            for i in range(start_index, min(start_index + 20, len(all_elements))):
                elem = all_elements[i]
                text = elem.get_text().strip()
                print(f"   {i}: {elem.name} - '{text[:60]}...'")
                
                # Show spans for list items
                if elem.name == 'li':
                    spans = elem.find_all('span')
                    for j, span in enumerate(spans):
                        style = span.get('style', '')
                        span_text = span.get_text().strip()
                        is_bold = 'font-weight:bold' in style
                        print(f"      Span {j}: {'[BOLD]' if is_bold else '[REGULAR]'} '{span_text}'")
            break

def process_files(folder_path='.', specific_file=None, debug_mode=False):
    """
    Process either a specific file or all HTML files in folder
    """
    
    if specific_file and os.path.exists(specific_file):
        # Debug mode
        if debug_mode:
            debug_html_structure(specific_file)
            return {}
        
        # Process single file
        print(f"🎯 Processing specific file: {specific_file}")
        result = extract_rating_drivers(specific_file)
        
        if result:
            return {specific_file.replace('.html', '').replace('.htm', ''): result}
        else:
            return {}
    
    else:
        # Process all HTML files in folder
        html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
        print(f"🔍 Found {len(html_files)} HTML files in folder")
        
        all_results = {}
        
        for filename in html_files:
            file_path = os.path.join(folder_path, filename)
            print(f"\n{'-'*50}")
            
            result = extract_rating_drivers(file_path)
            
            if result:
                file_key = filename.replace('.html', '').replace('.htm', '')
                all_results[file_key] = result
                print(f"✅ Successfully processed {filename}")
            else:
                print(f"❌ No data extracted from {filename}")
        
        return all_results

def save_and_display_results(results):
    """
    Save results to JSON and display in a clean format
    """
    
    if not results:
        print("❌ No results to save!")
        return
    
    # Save to JSON file
    output_file = 'rating_drivers_extracted.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Display results
    print(f"\n{'='*80}")
    print("📋 EXTRACTED RATING DRIVERS")
    print(f"{'='*80}")
    
    for filename, data in results.items():
        print(f"\n📄 FILE: {filename}")
        print(f"{'='*60}")
        
        for i, (key, value) in enumerate(data.items(), 1):
            # Determine if it's likely a strength or weakness
            key_lower = key.lower()
            if any(word in key_lower for word in ['exposure', 'risk', 'volatility', 'weakness', 'dependent']):
                icon = "⚠️"
                category = "WEAKNESS"
            else:
                icon = "💪"
                category = "STRENGTH"
            
            print(f"{i}. {icon} {category}")
            print(f"   KEY: {key}")
            print(f"   VALUE: {value}")
            print()
    
    print(f"💾 Results saved to: {output_file}")
    print(f"📊 Total items extracted: {sum(len(data) for data in results.values())}")

def main():
    """
    Main execution function
    """
    
    print("🚀 RATING DRIVERS EXTRACTION PIPELINE")
    print("="*50)
    
    # Option 1: Process specific file (recommended)
    specific_file = "Rating Rationale.html"  # Change this to your file name
    
    # Debug mode - set to True to see HTML structure
    debug_mode = False  # Change to True for debugging
    
    if os.path.exists(specific_file):
        print(f"✅ Found specific file: {specific_file}")
        
        if debug_mode:
            print("🔧 Running in DEBUG mode...")
            process_files(specific_file=specific_file, debug_mode=True)
        else:
            results = process_files(specific_file=specific_file)
            
            # Save and display results
            if results:
                save_and_display_results(results)
                print("\n🎉 Extraction completed successfully!")
            else:
                print("\n❌ No data extracted. Try running in debug mode to see the HTML structure.")
                print("💡 Set debug_mode = True in the main() function")
    else:
        print(f"❌ Specific file '{specific_file}' not found")
        print("🔍 Processing all HTML files in current directory...")
        results = process_files('.')
        
        if results:
            save_and_display_results(results)
            print("\n🎉 Extraction completed successfully!")
        else:
            print("\n❌ No data extracted. Please check your HTML file structure.")

if __name__ == "__main__":
    main()
