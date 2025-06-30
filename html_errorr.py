# BLOCK 1: Import libraries and basic setup
from bs4 import BeautifulSoup
import re
import os

print("✅ Libraries imported successfully")

# BLOCK 2: Check if HTML file exists and list all HTML files
folder_path = '.'  # Current folder
html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]

print(f"📁 Folder path: {folder_path}")
print(f"📄 HTML files found: {html_files}")
print(f"📊 Total HTML files: {len(html_files)}")

# BLOCK 3: Test reading one HTML file
if html_files:
    test_file = html_files[0]  # Take first HTML file
    file_path = os.path.join(folder_path, test_file)
    
    print(f"\n🧪 Testing file: {test_file}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        print(f"✅ File read successfully")
        print(f"📏 File size: {len(html_content)} characters")
        print(f"🔍 First 200 characters:")
        print(html_content[:200])
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
else:
    print("❌ No HTML files found!")

# BLOCK 4: Test HTML parsing
if html_files:
    soup = BeautifulSoup(html_content, 'html.parser')
    print(f"\n✅ HTML parsed successfully")
    print(f"📊 Total elements found: {len(soup.find_all())}")
    
    # Check for specific elements
    p_tags = soup.find_all('p')
    span_tags = soup.find_all('span')
    li_tags = soup.find_all('li')
    
    print(f"📄 <p> tags: {len(p_tags)}")
    print(f"📄 <span> tags: {len(span_tags)}")
    print(f"📄 <li> tags: {len(li_tags)}")

# BLOCK 5: Search for "Key Rating Drivers" text
if html_files:
    print(f"\n🔍 Searching for 'Key Rating Drivers' text...")
    
    found_elements = []
    for element in soup.find_all(['p', 'span', 'div']):
        text = element.get_text()
        if 'Key Rating Drivers' in text:
            found_elements.append(element)
            print(f"✅ Found in <{element.name}>: {text[:100]}...")
    
    if not found_elements:
        print("❌ 'Key Rating Drivers' not found!")
        
        # Let's search for similar terms
        print("\n🔍 Searching for similar terms...")
        search_terms = ['rating', 'drivers', 'strengths', 'weakness', 'detailed description']
        
        for term in search_terms:
            found = False
            for element in soup.find_all(['p', 'span', 'div']):
                if term.lower() in element.get_text().lower():
                    print(f"✅ Found '{term}' in: {element.get_text()[:100]}...")
                    found = True
                    break
            if not found:
                print(f"❌ '{term}' not found")

# BLOCK 6: Test target section extraction
if html_files and found_elements:
    print(f"\n🎯 Testing target section extraction...")
    
    target_section = found_elements[0].find_parent()
    print(f"✅ Target section found: <{target_section.name}>")
    
    # Check what's in the target section
    li_in_section = target_section.find_all('li')
    print(f"📄 <li> tags in target section: {len(li_in_section)}")
    
    if li_in_section:
        print(f"\n📝 First few <li> elements:")
        for i, li in enumerate(li_in_section[:5]):  # Show first 5
            print(f"{i+1}. {li.get_text()[:100]}...")
    else:
        print("❌ No <li> tags found in target section!")

# BLOCK 7: Test section detection (Strengths/Weaknesses)
if html_files and found_elements:
    print(f"\n🔍 Testing section detection...")
    
    target_section = found_elements[0].find_parent()
    
    strengths_found = False
    weaknesses_found = False
    
    for li in target_section.find_all('li'):
        text = li.get_text().strip()
        
        if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
            print(f"✅ STRENGTHS section found: {text}")
            strengths_found = True
        elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
            print(f"✅ WEAKNESSES section found: {text}")
            weaknesses_found = True
    
    if not strengths_found:
        print("❌ Strengths section not found!")
    if not weaknesses_found:
        print("❌ Weaknesses section not found!")

# BLOCK 8: Test bold text extraction
if html_files and found_elements:
    print(f"\n🔍 Testing bold text extraction...")
    
    target_section = found_elements[0].find_parent()
    bold_elements_found = 0
    
    for li in target_section.find_all('li'):
        bold_element = li.find(['strong', 'b'])
        if bold_element:
            bold_elements_found += 1
            key = bold_element.get_text().strip().rstrip(':')
            full_text = li.get_text().strip()
            value = full_text.replace(bold_element.get_text().strip(), '', 1).strip().lstrip(':').strip()
            
            print(f"🔸 Bold text found:")
            print(f"   Key: {key}")
            print(f"   Value: {value[:100]}...")
            print()
    
    print(f"📊 Total bold elements found: {bold_elements_found}")

# BLOCK 9: Test the complete extraction function
def debug_extract_strengths_weaknesses(html_file_path):
    """Debug version with detailed logging."""
    print(f"\n🧪 DEBUG: Starting extraction for {html_file_path}")
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    # Find the "Key Rating Drivers" section
    target_section = None
    for element in soup.find_all(['p', 'span']):
        if 'Key Rating Drivers' in element.get_text():
            target_section = element.find_parent()
            print(f"✅ Target section found")
            break
    
    if not target_section:
        print(f"❌ Target section not found")
        return {}, {}
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    li_elements = target_section.find_all('li')
    print(f"📄 Processing {len(li_elements)} list items")
    
    # Process all list items
    for i, li in enumerate(li_elements):
        text = li.get_text().strip()
        print(f"\n📝 Item {i+1}: {text[:50]}...")
        
        # Check for section headers
        if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
            current_section = 'strengths'
            print(f"   🎯 Entered STRENGTHS section")
            continue
        elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
            current_section = 'weaknesses'
            print(f"   🎯 Entered WEAKNESSES section")
            continue
        
        # Extract bold text as key and remaining text as value
        if current_section:
            bold_element = li.find(['strong', 'b'])
            if bold_element:
                key = bold_element.get_text().strip().rstrip(':')
                value = text.replace(bold_element.get_text().strip(), '', 1).strip().lstrip(':').strip()
                
                print(f"   ✅ Extracted - Key: {key[:30]}...")
                print(f"   ✅ Extracted - Value: {value[:50]}...")
                
                if current_section == 'strengths':
                    strengths_dict[key] = value
                elif current_section == 'weaknesses':
                    weaknesses_dict[key] = value
            else:
                print(f"   ❌ No bold text found in this item")
        else:
            print(f"   ⏸️  Not in any section yet")
    
    print(f"\n📊 Final Results:")
    print(f"   Strengths: {len(strengths_dict)}")
    print(f"   Weaknesses: {len(weaknesses_dict)}")
    
    return strengths_dict, weaknesses_dict

# BLOCK 10: Run the debug extraction
if html_files:
    test_file = html_files[0]
    file_path = os.path.join(folder_path, test_file)
    
    print(f"\n{'='*60}")
    print(f"RUNNING DEBUG EXTRACTION")
    print(f"{'='*60}")
    
    strengths, weaknesses = debug_extract_strengths_weaknesses(file_path)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    print(f"📊 Strengths found: {len(strengths)}")
    for key, value in strengths.items():
        print(f"   💪 {key}: {value[:100]}...")
    
    print(f"\n📊 Weaknesses found: {len(weaknesses)}")
    for key, value in weaknesses.items():
        print(f"   ⚡ {key}: {value[:100]}...")









# Try different file extensions or check folder path
print(os.listdir('.'))  # See all files in folder


# Search for any text containing "rating" or "drivers"
for element in soup.find_all():
    text = element.get_text().lower()
    if 'rating' in text or 'drivers' in text:
        print(f"Found: {text[:100]}")


# Search for "strength" or "weakness" anywhere
for li in soup.find_all('li'):
    text = li.get_text().lower()
    if 'strength' in text or 'weakness' in text:
        print(f"Found section: {text}")
