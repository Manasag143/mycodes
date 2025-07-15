import re
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging

class RatingDriversExtractor:
    """
    Simplified pipeline to extract strengths and weaknesses from rating rationale HTML documents.
    Strategy: Extract target area first, then parse with simple HTML methods.
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_html_file(self, file_path: str) -> str:
        """Load HTML file and return content as string"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.logger.info(f"Successfully loaded HTML file: {file_path}")
            return content
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            self.logger.info(f"Loaded HTML file with latin-1 encoding: {file_path}")
            return content
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def extract_target_area(self, html_content: str) -> Optional[str]:
        """
        Step 1: Extract the target area containing Key Rating Drivers
        
        Args:
            html_content (str): Full HTML content
            
        Returns:
            str: HTML content of the target area
        """
        try:
            # Parse the full HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Strategy 1: Find by text content "Key Rating Drivers"
            target_text = soup.find(string=re.compile(r"Key Rating Drivers.*Detailed Description", re.IGNORECASE))
            
            if target_text:
                # Get the parent container that has the full content
                parent = target_text.find_parent()
                
                # Navigate up to find the table cell or div containing all the content
                while parent and parent.name not in ['td', 'div']:
                    parent = parent.find_parent()
                
                if parent:
                    # Get the next sibling or parent that contains the actual content
                    content_container = parent.find_next('td') or parent
                    
                    if content_container:
                        target_html = str(content_container)
                        self.logger.info(f"Found target area: {len(target_html)} characters")
                        return target_html
            
            # Strategy 2: Search by specific HTML patterns
            # Look for table cells containing the key terms
            all_tds = soup.find_all('td')
            for td in all_tds:
                td_text = td.get_text()
                if (len(td_text) > 1000 and  # Must be substantial content
                    'Key Rating Drivers' in td_text and 
                    'Strengths' in td_text and
                    'Weakness' in td_text):
                    
                    target_html = str(td)
                    self.logger.info(f"Found target area via TD search: {len(target_html)} characters")
                    return target_html
            
            # Strategy 3: Look for div elements
            all_divs = soup.find_all('div')
            for div in all_divs:
                div_text = div.get_text()
                if (len(div_text) > 1000 and
                    'Key Rating Drivers' in div_text and 
                    'Strengths' in div_text):
                    
                    target_html = str(div)
                    self.logger.info(f"Found target area via DIV search: {len(target_html)} characters")
                    return target_html
            
            self.logger.warning("Could not find target area")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting target area: {str(e)}")
            return None
    
    def find_keywords(self, target_html: str) -> Dict[str, List[int]]:
        """
        Step 2: Find keyword positions in the target area
        
        Args:
            target_html (str): HTML content of target area
            
        Returns:
            Dict: Positions of key sections
        """
        try:
            # Convert to text for position finding
            soup = BeautifulSoup(target_html, 'html.parser')
            text = soup.get_text()
            
            # Find keyword positions
            positions = {}
            
            # Look for section headers
            keywords = [
                'Key Rating Drivers',
                'Strengths:',
                'Strengths',
                'Weakness:',
                'Weaknesses:',
                'Weakness',
                'Liquidity',
                'Outlook',
                'Analytical Approach'
            ]
            
            for keyword in keywords:
                # Find all occurrences (case insensitive)
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                matches = []
                for match in pattern.finditer(text):
                    matches.append(match.start())
                
                if matches:
                    positions[keyword.lower()] = matches
            
            self.logger.info(f"Found keywords: {list(positions.keys())}")
            return positions
            
        except Exception as e:
            self.logger.error(f"Error finding keywords: {str(e)}")
            return {}
    
    def extract_sections(self, target_html: str, positions: Dict[str, List[int]]) -> Dict[str, str]:
        """
        Step 3: Extract text sections based on keyword positions
        
        Args:
            target_html (str): HTML content of target area
            positions (Dict): Keyword positions
            
        Returns:
            Dict: Extracted sections
        """
        try:
            soup = BeautifulSoup(target_html, 'html.parser')
            text = soup.get_text()
            
            sections = {}
            
            # Find strengths section
            strengths_start = None
            for key in ['strengths:', 'strengths']:
                if key in positions:
                    strengths_start = positions[key][0]
                    break
            
            # Find weakness section  
            weakness_start = None
            for key in ['weakness:', 'weaknesses:', 'weakness']:
                if key in positions:
                    weakness_start = positions[key][0]
                    break
            
            # Find end boundaries
            end_boundaries = []
            for key in ['liquidity', 'outlook', 'analytical approach']:
                if key in positions:
                    end_boundaries.extend(positions[key])
            
            if strengths_start is not None:
                # Find end of strengths section
                strengths_end = weakness_start if weakness_start else (min(end_boundaries) if end_boundaries else len(text))
                strengths_text = text[strengths_start:strengths_end]
                sections['strengths'] = self.clean_text(strengths_text)
                self.logger.info(f"Extracted strengths section: {len(strengths_text)} characters")
            
            if weakness_start is not None:
                # Find end of weakness section
                weakness_end = min(end_boundaries) if end_boundaries else len(text)
                weakness_text = text[weakness_start:weakness_end]
                sections['weaknesses'] = self.clean_text(weakness_text)
                self.logger.info(f"Extracted weakness section: {len(weakness_text)} characters")
            
            return sections
            
        except Exception as e:
            self.logger.error(f"Error extracting sections: {str(e)}")
            return {}
    
    def parse_bullet_points(self, section_text: str) -> Dict[str, str]:
        """
        Step 4: Parse bullet points from section text using simple patterns
        
        Args:
            section_text (str): Text content of a section
            
        Returns:
            Dict: Title -> Description mapping
        """
        try:
            bullet_points = {}
            
            # Remove the section header
            text = re.sub(r'^(Strengths?:|Weaknesses?:|Weakness:)', '', section_text, flags=re.IGNORECASE).strip()
            
            # Method 1: Look for bullet points marked by HTML list items or bullet symbols
            # Split by potential bullet indicators
            bullet_patterns = [
                r'(?:^|\n)\s*[•·▪▫◦‣⁃]\s*',  # Unicode bullet points
                r'(?:^|\n)\s*\*\s*',          # Asterisk bullets
                r'(?:^|\n)\s*-\s*',           # Dash bullets
                r'(?:^|\n)\s*\d+\.\s*',       # Numbered lists
                r'(?:^|\n)\s*[a-zA-Z]\.\s*',  # Lettered lists
            ]
            
            # Try each pattern
            for pattern in bullet_patterns:
                parts = re.split(pattern, text)
                if len(parts) > 1:  # Found splits
                    for part in parts[1:]:  # Skip first empty part
                        part = part.strip()
                        if len(part) > 20:  # Minimum meaningful length
                            title, desc = self.extract_title_description(part)
                            if title and desc:
                                bullet_points[title] = desc
                    
                    if bullet_points:  # If we found something, stop trying other patterns
                        break
            
            # Method 2: If no bullets found, look for bold patterns in HTML
            if not bullet_points:
                # Look for colon-separated items
                colon_items = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
                
                for item in colon_items:
                    item = item.strip()
                    if ':' in item and len(item) > 30:
                        title, desc = self.extract_title_description(item)
                        if title and desc:
                            bullet_points[title] = desc
            
            # Method 3: Simple sentence-based extraction
            if not bullet_points:
                sentences = re.split(r'(?<=\.)\s+', text)
                current_title = None
                current_desc = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if ':' in sentence and len(sentence) < 200:  # Likely a title
                        if current_title and current_desc:
                            bullet_points[current_title] = current_desc.strip()
                        
                        parts = sentence.split(':', 1)
                        current_title = self.clean_text(parts[0])
                        current_desc = self.clean_text(parts[1]) if len(parts) > 1 else ""
                    else:
                        if current_title:
                            current_desc += " " + sentence
                
                # Add the last item
                if current_title and current_desc:
                    bullet_points[current_title] = current_desc.strip()
            
            self.logger.info(f"Parsed {len(bullet_points)} bullet points")
            return bullet_points
            
        except Exception as e:
            self.logger.error(f"Error parsing bullet points: {str(e)}")
            return {}
    
    def extract_title_description(self, text: str) -> tuple:
        """
        Extract title and description from a text item
        
        Args:
            text (str): Text item to parse
            
        Returns:
            tuple: (title, description)
        """
        try:
            if ':' in text:
                parts = text.split(':', 1)
                title = self.clean_text(parts[0])
                description = self.clean_text(parts[1])
                
                # Validate title and description
                if (len(title) > 5 and len(title) < 200 and 
                    len(description) > 20 and len(description) < 2000):
                    return title, description
            
            return None, None
            
        except Exception as e:
            self.logger.error(f"Error extracting title/description: {str(e)}")
            return None, None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s.,;:()\-&%/~]', '', text)
        
        return text
    
    def extract_rating_drivers(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """
        Main simplified pipeline
        
        Args:
            file_path (str): Path to HTML file
            
        Returns:
            Dict: Extracted strengths and weaknesses
        """
        try:
            # Step 1: Load HTML
            html_content = self.load_html_file(file_path)
            
            # Step 2: Extract target area
            target_html = self.extract_target_area(html_content)
            if not target_html:
                return {"strengths": {}, "weaknesses": {}}
            
            # Step 3: Find keywords
            positions = self.find_keywords(target_html)
            if not positions:
                self.logger.warning("No keywords found")
                return {"strengths": {}, "weaknesses": {}}
            
            # Step 4: Extract sections
            sections = self.extract_sections(target_html, positions)
            
            # Step 5: Parse bullet points
            result = {"strengths": {}, "weaknesses": {}}
            
            if 'strengths' in sections:
                result['strengths'] = self.parse_bullet_points(sections['strengths'])
            
            if 'weaknesses' in sections:
                result['weaknesses'] = self.parse_bullet_points(sections['weaknesses'])
            
            self.logger.info(f"Final result: {len(result['strengths'])} strengths, {len(result['weaknesses'])} weaknesses")
            
            # Debug output if nothing found
            if not result['strengths'] and not result['weaknesses']:
                self.debug_output(target_html, positions, sections)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in main pipeline: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def debug_output(self, target_html: str, positions: Dict, sections: Dict):
        """Debug output to help troubleshoot"""
        self.logger.info("=== DEBUG OUTPUT ===")
        self.logger.info(f"Target HTML length: {len(target_html)}")
        self.logger.info(f"Keyword positions: {positions}")
        self.logger.info(f"Sections found: {list(sections.keys())}")
        
        for section_name, section_text in sections.items():
            self.logger.info(f"\n{section_name.upper()} SECTION ({len(section_text)} chars):")
            self.logger.info(section_text[:500] + "..." if len(section_text) > 500 else section_text)
    
    def save_results(self, results: Dict[str, Dict[str, str]], output_path: str):
        """Save results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

# Example usage
def main():
    """Example usage of the simplified pipeline"""
    extractor = RatingDriversExtractor()
    
    # Replace with your actual file path
    file_path = "Rating Rationale.html"
    
    try:
        # Extract rating drivers
        results = extractor.extract_rating_drivers(file_path)
        
        # Print results
        print("=== SIMPLIFIED EXTRACTION RESULTS ===")
        print(f"\nFound {len(results['strengths'])} strengths:")
        for title, description in results['strengths'].items():
            print(f"\n📈 {title}")
            print(f"   {description[:200]}..." if len(description) > 200 else f"   {description}")
        
        print(f"\nFound {len(results['weaknesses'])} weaknesses:")
        for title, description in results['weaknesses'].items():
            print(f"\n📉 {title}")
            print(f"   {description[:200]}..." if len(description) > 200 else f"   {description}")
        
        # Save results
        extractor.save_results(results, "extracted_rating_drivers.json")
        
        return results
        
    except Exception as e:
        print(f"Error running extraction pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    main()
