# Block 1: Import necessary libraries
import re
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Tuple

print("Libraries imported successfully!")

# Block 2: Load and parse the HTML document
def load_html_content(file_path: str = None, html_content: str = None) -> BeautifulSoup:
    """
    Load HTML content either from file or direct string
    """
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
    elif file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
    else:
        raise ValueError("Either file_path or html_content must be provided")
    
    return soup

# Test with your document content (replace with actual HTML content)
# For now, let's assume the HTML content is in a variable called 'html_content'
html_content = """
<!-- Paste your actual HTML content here or load from file -->
"""

print("HTML parsing function defined!")

# Block 3: Extract text content and clean it
def extract_clean_text(soup: BeautifulSoup) -> str:
    """
    Extract and clean text from HTML, removing extra whitespace and formatting
    """
    # Get all text content
    text = soup.get_text()
    
    # Clean up text - remove extra whitespace, newlines, etc.
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Test the text extraction
# soup = load_html_content(html_content=html_content)
# clean_text = extract_clean_text(soup)
# print("Text extraction function defined!")

# Block 4: Find the Key Rating Drivers section
def find_rating_drivers_section(text: str) -> str:
    """
    Extract the Key Rating Drivers section from the document
    """
    # Pattern to find the Key Rating Drivers section
    pattern = r'Key Rating Drivers.*?(?=Rating sensitivity factors|Liquidity:|Outlook:|$)'
    
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(0)
    else:
        return ""

print("Rating drivers section extraction function defined!")

# Block 5: Extract strengths with sub-topics
def extract_strengths(rating_drivers_text: str) -> Dict[str, str]:
    """
    Extract strengths and their content from the rating drivers section
    """
    strengths_dict = {}
    
    # Find the strengths section
    strengths_pattern = r'Strengths?:(.*?)(?=Weakness|$)'
    strengths_match = re.search(strengths_pattern, rating_drivers_text, re.DOTALL | re.IGNORECASE)
    
    if strengths_match:
        strengths_content = strengths_match.group(1)
        
        # Split by bullet points or numbered items
        # Look for bold text followed by colon as sub-topics
        subtopic_pattern = r'([A-Z][^:]*?):\s*(.*?)(?=(?:[A-Z][^:]*?:|$))'
        
        matches = re.findall(subtopic_pattern, strengths_content, re.DOTALL)
        
        for i, (topic, content) in enumerate(matches):
            # Clean up the topic and content
            topic = re.sub(r'\s+', ' ', topic.strip())
            content = re.sub(r'\s+', ' ', content.strip())
            
            # Remove any remaining bullet points or formatting
            topic = re.sub(r'^[•\-\*\d\.]+\s*', '', topic)
            content = re.sub(r'^[•\-\*\d\.]+\s*', '', content)
            
            if topic and content:
                strengths_dict[topic] = content
    
    return strengths_dict

print("Strengths extraction function defined!")

# Block 6: Extract weaknesses with sub-topics
def extract_weaknesses(rating_drivers_text: str) -> Dict[str, str]:
    """
    Extract weaknesses and their content from the rating drivers section
    """
    weaknesses_dict = {}
    
    # Find the weakness section
    weakness_pattern = r'Weakness(?:es)?:(.*?)(?=$)'
    weakness_match = re.search(weakness_pattern, rating_drivers_text, re.DOTALL | re.IGNORECASE)
    
    if weakness_match:
        weakness_content = weakness_match.group(1)
        
        # Split by bullet points or numbered items
        # Look for bold text followed by colon as sub-topics
        subtopic_pattern = r'([A-Z][^:]*?):\s*(.*?)(?=(?:[A-Z][^:]*?:|$))'
        
        matches = re.findall(subtopic_pattern, weakness_content, re.DOTALL)
        
        for i, (topic, content) in enumerate(matches):
            # Clean up the topic and content
            topic = re.sub(r'\s+', ' ', topic.strip())
            content = re.sub(r'\s+', ' ', content.strip())
            
            # Remove any remaining bullet points or formatting
            topic = re.sub(r'^[•\-\*\d\.]+\s*', '', topic)
            content = re.sub(r'^[•\-\*\d\.]+\s*', '', content)
            
            if topic and content:
                weaknesses_dict[topic] = content
    
    return weaknesses_dict

print("Weaknesses extraction function defined!")

# Block 7: Alternative extraction method using HTML structure
def extract_strengths_weaknesses_from_html(soup: BeautifulSoup) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Alternative method to extract strengths and weaknesses using HTML structure
    """
    strengths_dict = {}
    weaknesses_dict = {}
    
    # Find all text content and look for patterns
    all_text = soup.get_text()
    
    # Split into sections
    sections = re.split(r'(Strengths?:|Weakness(?:es)?:)', all_text, flags=re.IGNORECASE)
    
    current_section = None
    for i, section in enumerate(sections):
        if re.match(r'Strengths?:', section, re.IGNORECASE):
            current_section = 'strengths'
        elif re.match(r'Weakness(?:es)?:', section, re.IGNORECASE):
            current_section = 'weaknesses'
        elif current_section and section.strip():
            # Process the content
            content = section.strip()
            
            # Look for bullet points or sub-topics
            lines = content.split('\n')
            current_topic = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line looks like a topic (often in bold or starts with bullet)
                if ':' in line and len(line.split(':')[0]) < 100:  # Likely a topic
                    # Save previous topic if exists
                    if current_topic and current_content:
                        content_text = ' '.join(current_content).strip()
                        if current_section == 'strengths':
                            strengths_dict[current_topic] = content_text
                        else:
                            weaknesses_dict[current_topic] = content_text
                    
                    # Start new topic
                    parts = line.split(':', 1)
                    current_topic = parts[0].strip()
                    current_content = [parts[1].strip()] if len(parts) > 1 else []
                else:
                    # Add to current content
                    if current_topic:
                        current_content.append(line)
            
            # Save last topic
            if current_topic and current_content:
                content_text = ' '.join(current_content).strip()
                if current_section == 'strengths':
                    strengths_dict[current_topic] = content_text
                else:
                    weaknesses_dict[current_topic] = content_text
    
    return strengths_dict, weaknesses_dict

print("Alternative HTML extraction function defined!")

# Block 8: Main extraction function
def extract_all_data(file_path: str = None, html_content: str = None) -> Dict:
    """
    Main function to extract all strengths and weaknesses data
    """
    try:
        # Load HTML content
        soup = load_html_content(file_path, html_content)
        
        # Method 1: Text-based extraction
        clean_text = extract_clean_text(soup)
        rating_drivers_section = find_rating_drivers_section(clean_text)
        
        strengths_method1 = extract_strengths(rating_drivers_section)
        weaknesses_method1 = extract_weaknesses(rating_drivers_section)
        
        # Method 2: HTML structure-based extraction
        strengths_method2, weaknesses_method2 = extract_strengths_weaknesses_from_html(soup)
        
        # Combine results (prefer method 1, fallback to method 2)
        final_strengths = strengths_method1 if strengths_method1 else strengths_method2
        final_weaknesses = weaknesses_method1 if weaknesses_method1 else weaknesses_method2
        
        return {
            'strengths': final_strengths,
            'weaknesses': final_weaknesses,
            'debug_info': {
                'rating_drivers_section_found': bool(rating_drivers_section),
                'method1_strengths_count': len(strengths_method1),
                'method1_weaknesses_count': len(weaknesses_method1),
                'method2_strengths_count': len(strengths_method2),
                'method2_weaknesses_count': len(weaknesses_method2)
            }
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'strengths': {},
            'weaknesses': {}
        }

print("Main extraction function defined!")

# Block 9: Display functions
def display_results(results: Dict):
    """
    Display the extracted results in a formatted way
    """
    print("="*80)
    print("EXTRACTION RESULTS")
    print("="*80)
    
    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"✅ Debug Info: {results.get('debug_info', {})}")
    print()
    
    print("📈 STRENGTHS:")
    print("-" * 40)
    for i, (topic, content) in enumerate(results['strengths'].items(), 1):
        print(f"{i}. {topic}")
        print(f"   {content}")
        print()
    
    print("📉 WEAKNESSES:")
    print("-" * 40)
    for i, (topic, content) in enumerate(results['weaknesses'].items(), 1):
        print(f"{i}. {topic}")
        print(f"   {content}")
        print()

def create_dataframe(results: Dict) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the results
    """
    data = []
    
    # Add strengths
    for topic, content in results['strengths'].items():
        data.append({
            'Category': 'Strength',
            'Topic': topic,
            'Content': content
        })
    
    # Add weaknesses
    for topic, content in results['weaknesses'].items():
        data.append({
            'Category': 'Weakness',
            'Topic': topic,
            'Content': content
        })
    
    return pd.DataFrame(data)

print("Display functions defined!")

# Block 10: CRISIL-specific extraction function
def extract_crisil_strengths_weaknesses(html_content: str) -> Dict:
    """
    Extract strengths and weaknesses specifically from CRISIL rating document
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    
    strengths = {}
    weaknesses = {}
    
    # Find strengths section
    strengths_pattern = r'Strengths:(.*?)(?=Weakness:|$)'
    strengths_match = re.search(strengths_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if strengths_match:
        strengths_text = strengths_match.group(1)
        
        # Extract individual strength points (they start with bullet points and have bold titles)
        # Pattern for: bullet + bold text + colon + content
        strength_items = re.findall(r'•\s*([^:]+):\s*([^•]+?)(?=•|$)', strengths_text, re.DOTALL)
        
        for title, content in strength_items:
            title = re.sub(r'\s+', ' ', title.strip())
            content = re.sub(r'\s+', ' ', content.strip())
            if title and content:
                strengths[title] = content
    
    # Find weakness section  
    weakness_pattern = r'Weakness:(.*?)(?=Liquidity:|Outlook:|$)'
    weakness_match = re.search(weakness_pattern, text, re.DOTALL | re.IGNORECASE)
    
    if weakness_match:
        weakness_text = weakness_match.group(1)
        
        # Extract individual weakness points
        weakness_items = re.findall(r'•\s*([^:]+):\s*([^•]+?)(?=•|$)', weakness_text, re.DOTALL)
        
        for title, content in weakness_items:
            title = re.sub(r'\s+', ' ', title.strip())
            content = re.sub(r'\s+', ' ', content.strip())
            if title and content:
                weaknesses[title] = content
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'debug_info': {
            'strengths_found': len(strengths),
            'weaknesses_found': len(weaknesses),
            'strengths_section_exists': bool(strengths_match),
            'weakness_section_exists': bool(weakness_match)
        }
    }

# Block 11: Run with your actual CRISIL document
def test_with_crisil_document():
    """
    Test with the actual CRISIL document content
    """
    # Your CRISIL HTML content (from the uploaded document)
    crisil_html = """<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>Rating Rationale</title>
<!-- ... rest of your HTML content ... -->
</head>
<body>
<!-- Paste the full content from your document here -->
""" + """
Key Rating Drivers & Detailed Description

Strengths:

• Strong position in India's phosphatic-fertiliser market: Coromandel is the second-largest player in the phosphatic-fertiliser industry in India with a primary market share of ~15% in DAP/NPK; and the largest share of ~15% in single super phosphate for fiscal 2024. Its market position is underpinned by an entrenched and leading position in Andhra Pradesh and Telangana – India's largest complex-fertiliser market – and a wide product portfolio. The company has also been gradually increasing the sale of non-subsidy-based products, including crop protection, speciality nutrients (secondary and micro-nutrients [sulphur, zinc, calcium and boron], water-soluble fertilisers and compost), and bioproducts (non-fertiliser segments contributed ~17% to the overall revenue in the first nine months through fiscal 2025). It operates around 850 retail outlets and has tied up with over 14,000 dealers, through which it sells fertilisers, crop-protection chemicals, speciality nutrient products, seeds, sprayers, veterinary products, among others.

• Strong operating efficiency: Operations benefit from economies of scale, better raw material procurement due to established relationships with suppliers, captive production of phosphoric acid, superior plant infrastructure and low handling and transportation costs. Captive phosphoric acid meets close to 50% of the company's total requirement while captive sulfuric acid meets ~60%. There are further plans to improve the backward integration in the near term by ramping up sulphuric acid and phosphoric acid capacities in the Kakinada plant. Operating efficiency is also supported by the ability to adjust product mix (between DAP and other complex fertilisers).

• Robust financial risk profile: Coromandel maintains a net cash position of over Rs 2,000 crore (net of acceptances/suppliers credit/buyer' credit of ~Rs 4,600 crore) as on December 31, 2024. Annual capex of Rs 800-1,000 crore, acquisition of NACL worth ~Rs 820 crore and incremental working capital requirement over the medium term will be met through strong yearly cash accrual of Rs 1,500-1,700 crore. Accordingly, the company is expected to remain net debt free over the medium term. Any larger-than-expected, debt-funded capex or acquisition that could materially alter capital structure would be monitorable.

Weakness:

• Exposure to regulated nature of the fertiliser industry and volatility in raw material prices: The fertiliser industry is strategic, but highly controlled, with fertiliser subsidy being an important component of profitability. The phosphatic-fertiliser industry was brought under the NBS regime from April 1, 2010. Under this scheme, the Government of India fixes the subsidy payable on nutrients for the entire fiscal (with an option to review this every six months), while retail prices are market driven. Manufacturers of phosphatic fertilisers are dependent on imports for their key raw materials, such as rock phosphate and phosphoric acid. Cost of raw materials accounts for about 75% of the operating income. The regulated nature of the industry and susceptibility of complex fertiliser players (including Coromandel) to raw material price volatility under the NBS regime continues to be key rating sensitivity factors.
""" + """
</body>
</html>
"""
    
    print("🔍 Extracting from CRISIL document...")
    results = extract_crisil_strengths_weaknesses(crisil_html)
    
    print("📋 EXTRACTION RESULTS:")
    print("=" * 60)
    
    print(f"Debug Info: {results['debug_info']}")
    print()
    
    print("💪 STRENGTHS:")
    print("-" * 40)
    for i, (topic, content) in enumerate(results['strengths'].items(), 1):
        print(f"{i}. Topic: {topic}")
        print(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
        print()
    
    print("⚠️ WEAKNESSES:")
    print("-" * 40)
    for i, (topic, content) in enumerate(results['weaknesses'].items(), 1):
        print(f"{i}. Topic: {topic}")
        print(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
        print()
    
    # Create DataFrame
    df = create_dataframe(results)
    print("📊 DATAFRAME:")
    print(df.to_string(max_colwidth=50))
    
    return results

print("CRISIL-specific extraction functions defined!")
print("\n" + "="*60)
print("🎯 READY TO EXTRACT FROM YOUR CRISIL DOCUMENT!")
print("="*60)
print("Run: test_with_crisil_document()")
print("="*60)
