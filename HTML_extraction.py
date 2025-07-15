import re
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging

class RatingDriversExtractor:
    """
    A pipeline to extract strengths and weaknesses from rating rationale HTML documents.
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
        """
        Load HTML file and return content as string
        
        Args:
            file_path (str): Path to the HTML file
            
        Returns:
            str: HTML content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            self.logger.info(f"Successfully loaded HTML file: {file_path}")
            return content
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
            self.logger.info(f"Loaded HTML file with latin-1 encoding: {file_path}")
            return content
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def parse_html(self, html_content: str) -> BeautifulSoup:
        """
        Parse HTML content using BeautifulSoup
        
        Args:
            html_content (str): Raw HTML content
            
        Returns:
            BeautifulSoup: Parsed HTML object
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            self.logger.info("Successfully parsed HTML content")
            return soup
        except Exception as e:
            self.logger.error(f"Error parsing HTML: {str(e)}")
            raise
    
    def find_rating_drivers_section(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """
        Find the section containing Key Rating Drivers & Detailed Description
        
        Args:
            soup (BeautifulSoup): Parsed HTML object
            
        Returns:
            BeautifulSoup: Section containing rating drivers
        """
        try:
            # Look for the heading "Key Rating Drivers & Detailed Description"
            rating_drivers_heading = soup.find(string=re.compile(r"Key Rating Drivers.*Detailed Description", re.IGNORECASE))
            
            if rating_drivers_heading:
                # Find the parent container
                parent = rating_drivers_heading.find_parent()
                while parent and parent.name not in ['td', 'div', 'section']:
                    parent = parent.find_parent()
                
                if parent:
                    # Get the content container
                    content_container = parent.find_parent()
                    self.logger.info("Found Key Rating Drivers section")
                    return content_container
            
            # Alternative search method - look for specific text patterns
            all_text = soup.get_text()
            if "Key Rating Drivers" in all_text and "Strengths:" in all_text:
                # Find all table cells that might contain the content
                tds = soup.find_all('td')
                for td in tds:
                    if "Key Rating Drivers" in td.get_text() and "Strengths:" in td.get_text():
                        self.logger.info("Found Key Rating Drivers section via alternative method")
                        return td
            
            self.logger.warning("Could not find Key Rating Drivers section")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding rating drivers section: {str(e)}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:()\-&%/]', '', text)
        
        return text
    
    def extract_bullet_points(self, section: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract bullet points from the rating drivers section
        
        Args:
            section (BeautifulSoup): HTML section containing bullet points
            
        Returns:
            List[Dict]: List of extracted bullet point data
        """
        bullet_points = []
        
        try:
            # Method 1: Look for <li> elements
            li_elements = section.find_all('li')
            for li in li_elements:
                text = li.get_text()
                if text and len(text.strip()) > 10:  # Filter out very short items
                    # Try to split title and description by colon
                    if ':' in text:
                        parts = text.split(':', 1)
                        title = self.clean_text(parts[0])
                        description = self.clean_text(parts[1])
                        bullet_points.append({
                            'title': title,
                            'description': description,
                            'full_text': self.clean_text(text)
                        })
            
            # Method 2: If no <li> elements, look for bold text patterns
            if not bullet_points:
                bold_elements = section.find_all(['b', 'strong', 'span'])
                for bold in bold_elements:
                    if bold.get('style') and 'font-weight:bold' in bold.get('style', ''):
                        text = bold.get_text()
                        if ':' in text:
                            # Get the next sibling text
                            parent = bold.find_parent()
                            full_text = parent.get_text() if parent else text
                            
                            if ':' in full_text:
                                parts = full_text.split(':', 1)
                                title = self.clean_text(parts[0])
                                description = self.clean_text(parts[1])
                                bullet_points.append({
                                    'title': title,
                                    'description': description,
                                    'full_text': self.clean_text(full_text)
                                })
            
            self.logger.info(f"Extracted {len(bullet_points)} bullet points")
            return bullet_points
            
        except Exception as e:
            self.logger.error(f"Error extracting bullet points: {str(e)}")
            return []
    
    def categorize_bullet_points(self, bullet_points: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
        """
        Categorize bullet points into strengths and weaknesses
        
        Args:
            bullet_points (List[Dict]): List of bullet point data
            
        Returns:
            Dict: Categorized strengths and weaknesses
        """
        result = {
            "strengths": {},
            "weaknesses": {}
        }
        
        try:
            current_category = None
            
            for point in bullet_points:
                full_text = point['full_text'].lower()
                title = point['title']
                description = point['description']
                
                # Check for category indicators
                if 'strength' in full_text and len(title) < 20:
                    current_category = 'strengths'
                    continue
                elif 'weakness' in full_text and len(title) < 20:
                    current_category = 'weaknesses'
                    continue
                
                # Skip if no category determined yet
                if current_category is None:
                    continue
                
                # Add to appropriate category if we have meaningful content
                if title and description and len(description) > 20:
                    result[current_category][title] = description
                elif title and len(title) > 10:
                    # If no clear description, use the full text
                    result[current_category][title] = point['full_text']
            
            self.logger.info(f"Categorized into {len(result['strengths'])} strengths and {len(result['weaknesses'])} weaknesses")
            return result
            
        except Exception as e:
            self.logger.error(f"Error categorizing bullet points: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def extract_from_text_patterns(self, text: str) -> Dict[str, Dict[str, str]]:
        """
        Fallback method: Extract using text patterns when HTML parsing fails
        
        Args:
            text (str): Full text content
            
        Returns:
            Dict: Extracted strengths and weaknesses
        """
        result = {
            "strengths": {},
            "weaknesses": {}
        }
        
        try:
            # Find strengths section
            strengths_match = re.search(r'Strengths?:\s*(.*?)(?=Weakness|$)', text, re.DOTALL | re.IGNORECASE)
            if strengths_match:
                strengths_text = strengths_match.group(1)
                # Look for bullet points or numbered items
                strength_items = re.findall(r'(?:•|\*|\d+\.)\s*([^•\*\d]+?)(?=•|\*|\d+\.|$)', strengths_text, re.DOTALL)
                
                for item in strength_items:
                    item = self.clean_text(item)
                    if ':' in item and len(item) > 20:
                        parts = item.split(':', 1)
                        title = self.clean_text(parts[0])
                        description = self.clean_text(parts[1])
                        if title and description:
                            result['strengths'][title] = description
            
            # Find weaknesses section
            weakness_match = re.search(r'Weakness(?:es)?:\s*(.*?)(?=\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
            if weakness_match:
                weakness_text = weakness_match.group(1)
                weakness_items = re.findall(r'(?:•|\*|\d+\.)\s*([^•\*\d]+?)(?=•|\*|\d+\.|$)', weakness_text, re.DOTALL)
                
                for item in weakness_items:
                    item = self.clean_text(item)
                    if ':' in item and len(item) > 20:
                        parts = item.split(':', 1)
                        title = self.clean_text(parts[0])
                        description = self.clean_text(parts[1])
                        if title and description:
                            result['weaknesses'][title] = description
            
            self.logger.info("Applied text pattern extraction as fallback")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in text pattern extraction: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def extract_rating_drivers(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """
        Main pipeline method to extract rating drivers from HTML file
        
        Args:
            file_path (str): Path to the HTML file
            
        Returns:
            Dict: Dictionary containing strengths and weaknesses
        """
        try:
            # Step 1: Load HTML file
            html_content = self.load_html_file(file_path)
            
            # Step 2: Parse HTML
            soup = self.parse_html(html_content)
            
            # Step 3: Find rating drivers section
            section = self.find_rating_drivers_section(soup)
            
            if section:
                # Step 4: Extract bullet points
                bullet_points = self.extract_bullet_points(section)
                
                # Step 5: Categorize into strengths and weaknesses
                result = self.categorize_bullet_points(bullet_points)
                
                # If HTML parsing didn't yield good results, try text pattern matching
                if not result['strengths'] and not result['weaknesses']:
                    self.logger.info("HTML parsing yielded no results, trying text pattern extraction")
                    result = self.extract_from_text_patterns(soup.get_text())
            else:
                # Fallback to text pattern extraction
                self.logger.info("Could not find section via HTML parsing, using text pattern extraction")
                result = self.extract_from_text_patterns(soup.get_text())
            
            # Validate results
            if not result['strengths'] and not result['weaknesses']:
                self.logger.warning("No strengths or weaknesses extracted")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in main extraction pipeline: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def save_results(self, results: Dict[str, Dict[str, str]], output_path: str):
        """
        Save extraction results to JSON file
        
        Args:
            results (Dict): Extracted results
            output_path (str): Path to save JSON file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")

# Example usage and testing
def main():
    """
    Example usage of the RatingDriversExtractor pipeline
    """
    # Initialize the extractor
    extractor = RatingDriversExtractor()
    
    # Example file path (replace with your actual file path)
    file_path = "Rating Rationale.html"
    
    try:
        # Extract rating drivers
        results = extractor.extract_rating_drivers(file_path)
        
        # Print results
        print("=== EXTRACTION RESULTS ===")
        print(f"\nFound {len(results['strengths'])} strengths:")
        for title, description in results['strengths'].items():
            print(f"\n📈 {title}")
            print(f"   {description[:200]}..." if len(description) > 200 else f"   {description}")
        
        print(f"\nFound {len(results['weaknesses'])} weaknesses:")
        for title, description in results['weaknesses'].items():
            print(f"\n📉 {title}")
            print(f"   {description[:200]}..." if len(description) > 200 else f"   {description}")
        
        # Save to JSON file
        extractor.save_results(results, "extracted_rating_drivers.json")
        
        return results
        
    except Exception as e:
        print(f"Error running extraction pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    main()
