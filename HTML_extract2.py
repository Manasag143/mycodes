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
    
    # Get all <p> elements in order
    all_p_elements = target_section.find_all('p')
    print(f"\n📄 Found {len(all_p_elements)} <p> elements")
    
    # Print all p elements for debugging
    print("\n🔍 ALL <p> ELEMENTS:")
    print("=" * 60)
    for i, p in enumerate(all_p_elements):
        text = p.get_text().strip()
        style = p.get('style', '')
        is_bold = 'font-weight' in style.lower() and 'bold' in style.lower()
        print(f"P{i+1}: '{text}' | Style: '{style}' | Bold: {is_bold}")
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    i = 0
    
    print(f"\n🔍 PROCESSING ELEMENTS (SIMPLIFIED):")
    print("=" * 60)
    
    while i < len(all_p_elements):
        p = all_p_elements[i]
        text = p.get_text().strip()
        style = p.get('style', '')
        is_bold = 'font-weight' in style.lower() and 'bold' in style.lower()
        
        print(f"\nP{i+1}: '{text}' | Bold: {is_bold}")
        
        # Check if this is Strengths section
        if re.search(r'\bStrengths?\s*:?', text, re.IGNORECASE):
            current_section = 'strengths'
            print(f"🎯 STRENGTHS section started")
            i += 1
            continue
            
        # Check if this is Weaknesses section
        elif re.search(r'\bWeakness(es)?\s*:?', text, re.IGNORECASE):
            current_section = 'weaknesses'
            print(f"🎯 WEAKNESSES section started")
            i += 1
            continue
        
        # If we're in a section and this paragraph is BOLD = it's a KEY
        if current_section and is_bold and text:
            key = text.rstrip(':').strip()
            print(f"🔑 KEY found: '{key}'")
            
            # Collect all NON-BOLD paragraphs after this until next BOLD or section change
            value_parts = []
            j = i + 1
            
            while j < len(all_p_elements):
                next_p = all_p_elements[j]
                next_text = next_p.get_text().strip()
                next_style = next_p.get('style', '')
                next_is_bold = 'font-weight' in next_style.lower() and 'bold' in next_style.lower()
                
                # Stop if we hit BOLD (next key) or Strengths/Weaknesses header
                if next_is_bold:
                    print(f"   🛑 Stopped at BOLD paragraph")
                    break
                if re.search(r'\b(Strengths?|Weakness(es)?)\s*:?', next_text, re.IGNORECASE):
                    print(f"   🛑 Stopped at section header")
                    break
                
                # Add this paragraph to value
                if next_text:
                    value_parts.append(next_text)
                    print(f"   ➕ Value part: '{next_text[:30]}...'")
                
                j += 1
            
            # Save the key-value pair
            if value_parts:
                combined_value = ' '.join(value_parts)
                
                if current_section == 'strengths':
                    strengths_dict[key] = combined_value
                    print(f"✅ STRENGTH: {key} = {combined_value[:50]}...")
                elif current_section == 'weaknesses':
                    weaknesses_dict[key] = combined_value
                    print(f"⚠️  WEAKNESS: {key} = {combined_value[:50]}...")
            
            # Jump to where we stopped
            i = j
            continue
        
        i += 1
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"📈 Strengths found: {len(strengths_dict)}")
    for key, value in strengths_dict.items():
        print(f"   💪 {key}: {value[:50]}...")
    
    print(f"📉 Weaknesses found: {len(weaknesses_dict)}")
    for key, value in weaknesses_dict.items():
        print(f"   ⚡ {key}: {value[:50]}...")
    
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
