from bs4 import BeautifulSoup
import re
import os
import json

def extract_strengths_weaknesses(html_file_path):
    """Extract strengths and weaknesses from a single HTML file."""
    print("🚀 Starting extraction function...")
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    print("✅ File read and parsed with BeautifulSoup")
    
    # Find the "Key Rating Drivers" section
    print("🔍 Looking for 'Key Rating Drivers' section...")
    target_section = None
    
    elements_found = soup.find_all(['p', 'span'])
    print(f"📋 Found {len(elements_found)} p and span elements to check")
    
    for i, element in enumerate(elements_found):
        element_text = element.get_text()
        print(f"  Checking element {i+1}: '{element_text[:50]}...'")
        
        if 'Key Rating Drivers' in element_text:
            print(f"  ✅ FOUND 'Key Rating Drivers' in element {i+1}!")
            target_section = element.find_parent()
            print(f"  📦 Got parent element: <{target_section.name}>")
            break
        else:
            print(f"  ❌ No 'Key Rating Drivers' in this element")
    
    if not target_section:
        print("❌ FINAL RESULT: No 'Key Rating Drivers' section found!")
        return {}, {}
    
    print(f"✅ FINAL RESULT: Found target section of type <{target_section.name}>")
    
    # Get all <p> elements in order
    print("\n🔍 Looking for <p> elements in target section...")
    all_p_elements = target_section.find_all('p')
    print(f"📄 Found {len(all_p_elements)} <p> elements")
    
    if len(all_p_elements) == 0:
        print("❌ PROBLEM: NO <p> ELEMENTS FOUND!")
        print("🔍 Let's see what elements ARE in the target section:")
        all_children = target_section.find_all()
        print(f"📋 Total child elements: {len(all_children)}")
        for i, child in enumerate(all_children[:10]):
            print(f"  Child {i+1}: <{child.name}> = '{child.get_text().strip()[:30]}...'")
        return {}, {}
    
    print(f"✅ SUCCESS: We have {len(all_p_elements)} <p> elements to process")
    
    # Initialize variables
    print("\n🔧 Initializing variables...")
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    i = 0
    print(f"✅ Variables initialized: strengths={len(strengths_dict)}, weaknesses={len(weaknesses_dict)}, section={current_section}, i={i}")
    
    print(f"\n🔄 Starting main processing loop...")
    
    while i < len(all_p_elements):
        print(f"\n--- LOOP ITERATION {i+1} ---")
        p = all_p_elements[i]
        print(f"📍 Processing element {i+1}/{len(all_p_elements)}")
        
        text = p.get_text().strip()
        print(f"📝 Text extracted: '{text}'")
        
        style = p.get('style', '')
        print(f"🎨 Style attribute: '{style}'")
        
        is_bold = 'font-weight' in style.lower() and 'bold' in style.lower()
        print(f"💪 Is bold check: {is_bold}")
        
        # Check if this is Strengths section
        print("🔍 Checking if this is Strengths section...")
        strengths_match = re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE)
        print(f"🎯 Strengths regex match: {strengths_match is not None}")
        
        if strengths_match:
            current_section = 'strengths'
            print(f"✅ STRENGTHS section started! current_section = '{current_section}'")
            i += 1
            print(f"⏭️  Moving to next element, i = {i}")
            continue
            
        # Check if this is Weaknesses section
        print("🔍 Checking if this is Weaknesses section...")
        weakness_match = re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE)
        print(f"🎯 Weaknesses regex match: {weakness_match is not None}")
        
        if weakness_match:
            current_section = 'weaknesses'
            print(f"✅ WEAKNESSES section started! current_section = '{current_section}'")
            i += 1
            print(f"⏭️  Moving to next element, i = {i}")
            continue
        
        print(f"📍 Current section status: '{current_section}'")
        print(f"💪 Is bold status: {is_bold}")
        print(f"📝 Has text: {bool(text)}")
        
        # If we're in a section and this paragraph is BOLD = it's a KEY
        if current_section and is_bold and text:
            print(f"🔑 KEY DETECTED! We're in section '{current_section}' with bold text")
            
            key = text.rstrip(':').strip()
            print(f"🔑 KEY processed: '{key}'")
            
            # Collect all NON-BOLD paragraphs after this until next BOLD or section change
            print(f"📝 Starting to collect value parts...")
            value_parts = []
            j = i + 1
            print(f"🔢 Starting inner loop from position {j}")
            
            while j < len(all_p_elements):
                print(f"\n  --- VALUE COLLECTION LOOP {j+1} ---")
                next_p = all_p_elements[j]
                print(f"  📍 Checking element {j+1}/{len(all_p_elements)} for value")
                
                next_text = next_p.get_text().strip()
                print(f"  📝 Next text: '{next_text}'")
                
                next_style = next_p.get('style', '')
                print(f"  🎨 Next style: '{next_style}'")
                
                next_is_bold = 'font-weight' in next_style.lower() and 'bold' in next_style.lower()
                print(f"  💪 Next is bold: {next_is_bold}")
                
                # Stop if we hit BOLD (next key) or Strengths/Weaknesses header
                if next_is_bold:
                    print(f"  🛑 STOPPING: Found next bold element")
                    break
                    
                section_header_match = re.search(r'\b(Strengths?|Weakness(es)?)\s*:?', next_text, re.IGNORECASE)
                print(f"  🔍 Section header check: {section_header_match is not None}")
                
                if section_header_match:
                    print(f"  🛑 STOPPING: Found section header")
                    break
                
                # Add this paragraph to value
                if next_text:
                    value_parts.append(next_text)
                    print(f"  ➕ ADDED to value_parts: '{next_text[:30]}...'")
                    print(f"  📊 value_parts now has {len(value_parts)} items")
                else:
                    print(f"  ⏭️  SKIPPED: Empty text")
                
                j += 1
                print(f"  🔢 Inner loop counter j = {j}")
            
            print(f"📝 Value collection finished. Total parts: {len(value_parts)}")
            
            # Save the key-value pair
            if value_parts:
                combined_value = ' '.join(value_parts)
                print(f"🔗 Combined value: '{combined_value[:50]}...'")
                
                if current_section == 'strengths':
                    strengths_dict[key] = combined_value
                    print(f"✅ SAVED TO STRENGTHS: '{key}' = '{combined_value[:30]}...'")
                    print(f"📊 Strengths dict now has {len(strengths_dict)} items")
                elif current_section == 'weaknesses':
                    weaknesses_dict[key] = combined_value
                    print(f"⚠️  SAVED TO WEAKNESSES: '{key}' = '{combined_value[:30]}...'")
                    print(f"📊 Weaknesses dict now has {len(weaknesses_dict)} items")
            else:
                print(f"❌ NO VALUE PARTS: No value found for key '{key}'")
            
            # Jump to where we stopped
            i = j
            print(f"⏭️  Jumping to position i = {i}")
            continue
        else:
            print(f"⏭️  SKIPPING: Not a key (section={current_section}, bold={is_bold}, text={bool(text)})")
        
        i += 1
        print(f"🔢 Main loop counter i = {i}")
    
    print(f"\n🏁 PROCESSING COMPLETE!")
    print(f"📈 Final strengths count: {len(strengths_dict)}")
    print(f"📉 Final weaknesses count: {len(weaknesses_dict)}")
    
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
