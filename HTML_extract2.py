from bs4 import BeautifulSoup
import re

def extract_strengths_weaknesses_from_html(html_file_path):
    """
    Extract strengths and weaknesses from HTML file under 'Key Rating Drivers & Detailed Description' section
    
    Args:
        html_file_path (str): Path to the HTML file
    
    Returns:
        dict: Dictionary containing strengths and weaknesses
    """
    
    # Read the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Initialize result dictionary
    result = {
        "Strengths": {},
        "Weakness": {}
    }
    
    # Find the "Key Rating Drivers & Detailed Description" section
    # Look for text containing this heading
    key_rating_section = None
    
    # Method 1: Search for the exact text in spans or other elements
    for element in soup.find_all(['span', 'p', 'div', 'td']):
        if element.get_text(strip=True) == "Key Rating Drivers & Detailed Description":
            key_rating_section = element
            break
    
    # Method 2: Search using regex if exact match not found
    if not key_rating_section:
        for element in soup.find_all(['span', 'p', 'div', 'td']):
            text = element.get_text(strip=True)
            if re.search(r'Key\s+Rating\s+Drivers.*Detailed\s+Description', text, re.IGNORECASE):
                key_rating_section = element
                break
    
    if not key_rating_section:
        print("Key Rating Drivers section not found")
        return result
    
    # Find the parent container that contains the content
    content_container = key_rating_section.find_parent(['table', 'div', 'td'])
    
    # If we found the container, get the next sibling or parent that contains the actual content
    if content_container:
        # Look for the div that contains the strengths and weaknesses content
        content_div = content_container.find('div')
        if content_div:
            content_text = content_div.get_text()
        else:
            content_text = content_container.get_text()
    else:
        # Fallback: get text from the entire document and find the relevant section
        full_text = soup.get_text()
        
        # Find the section between "Key Rating Drivers" and next major section
        pattern = r'Key\s+Rating\s+Drivers.*?Detailed\s+Description(.*?)(?:Liquidity|Outlook|Analytical\s+Approach|$)'
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content_text = match.group(1)
        else:
            print("Could not extract content from Key Rating Drivers section")
            return result
    
    return parse_strengths_weaknesses(content_text)

def parse_strengths_weaknesses(content_text):
    """
    Parse the content text to extract strengths and weaknesses
    
    Args:
        content_text (str): Text content containing strengths and weaknesses
    
    Returns:
        dict: Dictionary with extracted strengths and weaknesses
    """
    
    result = {
        "Strengths": {},
        "Weakness": {}
    }
    
    # Clean up the text
    content_text = re.sub(r'\s+', ' ', content_text).strip()
    
    # Split into sections based on "Strengths:" and "Weakness:"
    sections = re.split(r'\b(Strengths?:|Weakness(?:es)?:)', content_text, flags=re.IGNORECASE)
    
    current_section = None
    
    for i, section in enumerate(sections):
        section = section.strip()
        
        # Identify section headers
        if re.match(r'Strengths?:', section, re.IGNORECASE):
            current_section = "Strengths"
            continue
        elif re.match(r'Weakness(?:es)?:', section, re.IGNORECASE):
            current_section = "Weakness"
            continue
        
        # Process content for current section
        if current_section and section:
            if current_section == "Strengths":
                result["Strengths"].update(extract_bullet_points(section))
            elif current_section == "Weakness":
                result["Weakness"].update(extract_bullet_points(section))
    
    return result

def extract_bullet_points(text):
    """
    Extract individual bullet points (strengths/weaknesses) from text
    
    Args:
        text (str): Text containing bullet points
    
    Returns:
        dict: Dictionary with titles as keys and descriptions as values
    """
    
    points = {}
    
    # Split by bullet point indicators or new lines with bold text
    # Look for patterns like "Strong position in...: description"
    
    # Method 1: Split by common bullet patterns
    bullet_patterns = [
        r'(?:^|\n)\s*[•·▪▫-]\s*',  # Bullet symbols
        r'(?:^|\n)\s*\d+\.\s*',    # Numbered lists
        r'(?:^|\n)(?=[A-Z][^:]*:)'  # Capital letter starting lines ending with colon
    ]
    
    # Try to split text into individual points
    segments = re.split('|'.join(bullet_patterns), text, flags=re.MULTILINE)
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Look for title: description pattern
        colon_match = re.match(r'^([^:]+):\s*(.*)', segment, re.DOTALL)
        
        if colon_match:
            title = colon_match.group(1).strip()
            description = colon_match.group(2).strip()
            
            # Clean up title and description
            title = re.sub(r'\s+', ' ', title)
            description = re.sub(r'\s+', ' ', description)
            
            # Filter out very short titles or descriptions
            if len(title) > 5 and len(description) > 20:
                points[title] = description
    
    # Method 2: If no clear bullet points found, try to extract from paragraph structure
    if not points:
        # Look for bold text patterns followed by descriptions
        paragraphs = text.split('\n')
        current_title = None
        current_desc = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this looks like a title (short, ends with colon, or is in bold context)
            if (len(para.split()) <= 10 and 
                (para.endswith(':') or 
                 re.search(r'^[A-Z][^:]*(?:efficiency|position|profile|industry)', para, re.IGNORECASE))):
                
                # Save previous point
                if current_title and current_desc:
                    points[current_title] = ' '.join(current_desc).strip()
                
                # Start new point
                current_title = para.replace(':', '').strip()
                current_desc = []
            
            elif current_title:
                current_desc.append(para)
        
        # Save last point
        if current_title and current_desc:
            points[current_title] = ' '.join(current_desc).strip()
    
    return points

def print_results(strengths_weaknesses):
    """
    Print the extracted strengths and weaknesses in a formatted way
    """
    
    print("=== EXTRACTED STRENGTHS AND WEAKNESSES ===\n")
    
    print("STRENGTHS:")
    print("-" * 60)
    if strengths_weaknesses["Strengths"]:
        for i, (title, content) in enumerate(strengths_weaknesses["Strengths"].items(), 1):
            print(f"\n{i}. {title}:")
            print(f"   {content}")
    else:
        print("   No strengths found")
    
    print("\n\nWEAKNESSES:")
    print("-" * 60)
    if strengths_weaknesses["Weakness"]:
        for i, (title, content) in enumerate(strengths_weaknesses["Weakness"].items(), 1):
            print(f"\n{i}. {title}:")
            print(f"   {content}")
    else:
        print("   No weaknesses found")
    
    print("\n\n=== DICTIONARY FORMAT ===")
    import json
    print(json.dumps(strengths_weaknesses, indent=2, ensure_ascii=False))

# Alternative method for direct HTML content
def extract_from_html_content(html_content):
    """
    Extract strengths and weaknesses directly from HTML content string
    
    Args:
        html_content (str): HTML content as string
    
    Returns:
        dict: Dictionary containing strengths and weaknesses
    """
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all text content
    full_text = soup.get_text()
    
    # Find the Key Rating Drivers section
    pattern = r'Key\s+Rating\s+Drivers.*?Detailed\s+Description(.*?)(?:Liquidity|Outlook|Analytical\s+Approach|About\s+the\s+Company)'
    match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        content_text = match.group(1)
        return parse_strengths_weaknesses(content_text)
    else:
        print("Key Rating Drivers section not found in HTML content")
        return {"Strengths": {}, "Weakness": {}}

# Example usage
if __name__ == "__main__":
    # Method 1: From HTML file
    try:
        html_file_path = "Rating_Rationale.html"  # Replace with your HTML file path
        result = extract_strengths_weaknesses_from_html(html_file_path)
        print_results(result)
    except FileNotFoundError:
        print(f"HTML file not found. Please check the file path.")
    except Exception as e:
        print(f"Error processing HTML file: {e}")
    
    # Method 2: From HTML content string (if you have the HTML as a string)
    # html_content = "<html>...</html>"  # Your HTML content
    # result = extract_from_html_content(html_content)
    # print_results(result)
