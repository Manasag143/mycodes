# FIX BLOCK 1: Investigate the target section structure
if html_files and found_elements:
    print(f"🔍 INVESTIGATING TARGET SECTION STRUCTURE")
    print("="*50)
    
    target_section = found_elements[0].find_parent()
    print(f"Target section tag: <{target_section.name}>")
    
    # Check all child elements
    all_children = target_section.find_all()
    print(f"Total child elements: {len(all_children)}")
    
    # Count different types of elements
    element_types = {}
    for child in all_children:
        tag = child.name
        element_types[tag] = element_types.get(tag, 0) + 1
    
    print(f"\nElement types found:")
    for tag, count in element_types.items():
        print(f"  <{tag}>: {count}")

# FIX BLOCK 2: Look for any lists (ul/ol) in a wider search
if html_files:
    print(f"\n🔍 SEARCHING FOR LISTS IN ENTIRE DOCUMENT")
    print("="*50)
    
    # Find all ul and ol elements
    all_ul = soup.find_all('ul')
    all_ol = soup.find_all('ol')
    
    print(f"Found {len(all_ul)} <ul> elements")
    print(f"Found {len(all_ol)} <ol> elements")
    
    # Check content around these lists
    for i, ul in enumerate(all_ul):
        print(f"\n📋 UL #{i+1}:")
        
        # Check previous elements for context
        prev_elements = []
        current = ul
        for _ in range(3):  # Check 3 previous elements
            current = current.find_previous(['p', 'span', 'div', 'td'])
            if current:
                text = current.get_text().strip()
                if text and len(text) < 200:  # Only show short texts
                    prev_elements.append(text)
        
        print(f"  Context: {' | '.join(prev_elements)}")
        
        # Show first few list items
        li_items = ul.find_all('li', recursive=False)
        print(f"  List items: {len(li_items)}")
        for j, li in enumerate(li_items[:2]):  # Show first 2
            text = li.get_text().strip()
            print(f"    {j+1}. {text[:100]}...")

# FIX BLOCK 3: Search for Strengths/Weaknesses text in any element
if html_files:
    print(f"\n🔍 SEARCHING FOR STRENGTHS/WEAKNESSES TEXT")
    print("="*50)
    
    strengths_elements = []
    weaknesses_elements = []
    
    # Search in all elements
    for element in soup.find_all():
        text = element.get_text()
        
        if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
            strengths_elements.append(element)
            print(f"✅ STRENGTHS found in <{element.name}>: {text[:100]}...")
        
        if re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
            weaknesses_elements.append(element)
            print(f"✅ WEAKNESSES found in <{element.name}>: {text[:100]}...")
    
    print(f"\nTotal Strengths elements: {len(strengths_elements)}")
    print(f"Total Weaknesses elements: {len(weaknesses_elements)}")

# FIX BLOCK 4: Check if content is in paragraphs instead of lists
if html_files and strengths_elements:
    print(f"\n🔍 CHECKING PARAGRAPH STRUCTURE")
    print("="*50)
    
    # Take first strengths element and check its siblings
    first_strength = strengths_elements[0]
    parent = first_strength.find_parent()
    
    print(f"Strengths parent: <{parent.name}>")
    
    # Get all siblings after the strengths element
    all_siblings = []
    current = first_strength
    for _ in range(10):  # Check next 10 elements
        current = current.find_next_sibling()
        if current:
            all_siblings.append(current)
        else:
            break
    
    print(f"Found {len(all_siblings)} sibling elements")
    
    for i, sibling in enumerate(all_siblings):
        text = sibling.get_text().strip()
        if text:
            print(f"{i+1}. <{sibling.name}>: {text[:100]}...")
            
            # Check for bold text
            bold_elements = sibling.find_all(['strong', 'b'])
            if bold_elements:
                print(f"   💪 Bold text: {bold_elements[0].get_text()}")

# FIX BLOCK 5: Alternative extraction method based on what we find
def extract_from_paragraphs(soup):
    """Try extracting from paragraph structure instead of lists."""
    print(f"\n🔧 TRYING PARAGRAPH-BASED EXTRACTION")
    print("="*50)
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    # Find all paragraphs
    all_paragraphs = soup.find_all('p')
    
    for p in all_paragraphs:
        text = p.get_text().strip()
        
        # Check for section headers
        if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
            current_section = 'strengths'
            print(f"✅ Found Strengths section")
            continue
        elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
            current_section = 'weaknesses'
            print(f"✅ Found Weaknesses section")
            continue
        
        # Extract content if we're in a section
        if current_section and text:
            bold_element = p.find(['strong', 'b'])
            if bold_element:
                key = bold_element.get_text().strip().rstrip(':')
                value = text.replace(bold_element.get_text().strip(), '', 1).strip().lstrip(':').strip()
                
                print(f"📝 Found: {key}")
                
                if current_section == 'strengths':
                    strengths_dict[key] = value
                elif current_section == 'weaknesses':
                    weaknesses_dict[key] = value
    
    return strengths_dict, weaknesses_dict

# FIX BLOCK 6: Test the paragraph-based extraction
if html_files:
    strengths, weaknesses = extract_from_paragraphs(soup)
    
    print(f"\n📊 PARAGRAPH EXTRACTION RESULTS:")
    print(f"Strengths: {len(strengths)}")
    print(f"Weaknesses: {len(weaknesses)}")
    
    if strengths:
        print(f"\n💪 STRENGTHS:")
        for key, value in strengths.items():
            print(f"  • {key}: {value[:100]}...")
    
    if weaknesses:
        print(f"\n⚡ WEAKNESSES:")
        for key, value in weaknesses.items():
            print(f"  • {key}: {value[:100]}...")

# FIX BLOCK 7: Create corrected extraction function
def corrected_extract_strengths_weaknesses(html_file_path):
    """Corrected extraction function based on actual HTML structure."""
    with open(html_file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    
    strengths_dict = {}
    weaknesses_dict = {}
    current_section = None
    
    # Method 1: Try list-based extraction
    all_lists = soup.find_all('ul')
    for ul in all_lists:
        # Check if this list is in strengths/weaknesses context
        context = ""
        prev = ul.find_previous(['p', 'span', 'div'])
        if prev:
            context = prev.get_text().lower()
        
        if 'strength' in context or 'weakness' in context:
            for li in ul.find_all('li'):
                text = li.get_text().strip()
                
                if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
                    current_section = 'strengths'
                    continue
                elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
                    current_section = 'weaknesses'
                    continue
                
                if current_section:
                    bold_element = li.find(['strong', 'b'])
                    if bold_element:
                        key = bold_element.get_text().strip().rstrip(':')
                        value = text.replace(bold_element.get_text().strip(), '', 1).strip().lstrip(':').strip()
                        
                        if current_section == 'strengths':
                            strengths_dict[key] = value
                        elif current_section == 'weaknesses':
                            weaknesses_dict[key] = value
    
    # Method 2: If lists didn't work, try paragraphs
    if not strengths_dict and not weaknesses_dict:
        current_section = None
        for element in soup.find_all(['p', 'div']):
            text = element.get_text().strip()
            
            if re.search(r'\bStrengths?\s*:', text, re.IGNORECASE):
                current_section = 'strengths'
                continue
            elif re.search(r'\bWeakness(es)?\s*:', text, re.IGNORECASE):
                current_section = 'weaknesses'
                continue
            
            if current_section and text:
                bold_element = element.find(['strong', 'b'])
                if bold_element:
                    key = bold_element.get_text().strip().rstrip(':')
                    value = text.replace(bold_element.get_text().strip(), '', 1).strip().lstrip(':').strip()
                    
                    if current_section == 'strengths':
                        strengths_dict[key] = value
                    elif current_section == 'weaknesses':
                        weaknesses_dict[key] = value
    
    return strengths_dict, weaknesses_dict

# FIX BLOCK 8: Test the corrected function
if html_files:
    print(f"\n{'='*60}")
    print(f"TESTING CORRECTED EXTRACTION FUNCTION")
    print(f"{'='*60}")
    
    test_file = html_files[0]
    file_path = os.path.join(folder_path, test_file)
    
    final_strengths, final_weaknesses = corrected_extract_strengths_weaknesses(file_path)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"✅ Strengths: {len(final_strengths)}")
    print(f"✅ Weaknesses: {len(final_weaknesses)}")
    
    if final_strengths:
        print(f"\n💪 STRENGTHS:")
        for key, value in final_strengths.items():
            print(f"  • {key}")
            print(f"    {value[:150]}...")
            print()
    
    if final_weaknesses:
        print(f"\n⚡ WEAKNESSES:")
        for key, value in final_weaknesses.items():
            print(f"  • {key}")
            print(f"    {value[:150]}...")
            print()
