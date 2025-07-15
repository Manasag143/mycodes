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
    
    def find_rating_drivers_section(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Find the section containing Key Rating Drivers & Detailed Description
        and return the text content
        
        Args:
            soup (BeautifulSoup): Parsed HTML object
            
        Returns:
            str: Text content of the rating drivers section
        """
        try:
            # Get all text content
            full_text = soup.get_text()
            
            # Look for the Key Rating Drivers section
            pattern = r'Key Rating Drivers.*?Detailed Description(.*?)(?=Analytical Approach|Liquidity|Outlook|$)'
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                section_text = match.group(1)
                self.logger.info("Found Key Rating Drivers section")
                return section_text
            
            # Alternative pattern - look for just the content after "Key Rating Drivers"
            pattern2 = r'Key Rating Drivers[^:]*:(.*?)(?=Analytical Approach|Liquidity|Outlook|About the Company|$)'
            match2 = re.search(pattern2, full_text, re.DOTALL | re.IGNORECASE)
            
            if match2:
                section_text = match2.group(1)
                self.logger.info("Found Key Rating Drivers section (alternative pattern)")
                return section_text
            
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
        
        # Remove unwanted characters but keep essential punctuation
        text = re.sub(r'[^\w\s.,;:()\-&%/~]', '', text)
        
        return text
    
    def extract_strengths_and_weaknesses(self, section_text: str) -> Dict[str, Dict[str, str]]:
        """
        Extract strengths and weaknesses from the section text
        
        Args:
            section_text (str): Text content of the rating drivers section
            
        Returns:
            Dict: Dictionary containing strengths and weaknesses
        """
        result = {
            "strengths": {},
            "weaknesses": {}
        }
        
        try:
            # Clean the text first
            section_text = self.clean_text(section_text)
            
            # Find the strengths section
            strengths_pattern = r'Strengths?:\s*(.*?)(?=Weakness|$)'
            strengths_match = re.search(strengths_pattern, section_text, re.DOTALL | re.IGNORECASE)
            
            if strengths_match:
                strengths_text = strengths_match.group(1)
                self.logger.info(f"Found strengths section: {len(strengths_text)} characters")
                
                # Extract individual strength items
                # Look for bullet points or bold patterns followed by descriptions
                strength_items = self.extract_bullet_items(strengths_text)
                
                for item in strength_items:
                    if item['title'] and item['description']:
                        result['strengths'][item['title']] = item['description']
            
            # Find the weaknesses section  
            weakness_pattern = r'Weakness(?:es)?:\s*(.*?)$'
            weakness_match = re.search(weakness_pattern, section_text, re.DOTALL | re.IGNORECASE)
            
            if weakness_match:
                weakness_text = weakness_match.group(1)
                self.logger.info(f"Found weakness section: {len(weakness_text)} characters")
                
                # Extract individual weakness items
                weakness_items = self.extract_bullet_items(weakness_text)
                
                for item in weakness_items:
                    if item['title'] and item['description']:
                        result['weaknesses'][item['title']] = item['description']
            
            self.logger.info(f"Extracted {len(result['strengths'])} strengths and {len(result['weaknesses'])} weaknesses")
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting strengths and weaknesses: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def extract_bullet_items(self, text: str) -> List[Dict[str, str]]:
        """
        Extract bullet point items from text
        
        Args:
            text (str): Text containing bullet points
            
        Returns:
            List[Dict]: List of extracted items with title and description
        """
        items = []
        
        try:
            # Split by potential bullet point indicators
            # The text uses bullet points that might not be standard characters
            
            # Method 1: Split by sentences and look for title: description pattern
            sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
            
            for sentence in sentences:
                # Look for pattern: "Title: Description"
                if ':' in sentence and len(sentence) > 50:  # Minimum length for meaningful content
                    colon_pos = sentence.find(':')
                    potential_title = sentence[:colon_pos].strip()
                    potential_desc = sentence[colon_pos+1:].strip()
                    
                    # Filter valid titles (should be meaningful phrases)
                    if (len(potential_title) > 10 and len(potential_title) < 150 and 
                        len(potential_desc) > 30):
                        items.append({
                            'title': potential_title,
                            'description': potential_desc
                        })
            
            # Method 2: If method 1 doesn't work, try paragraph-based splitting
            if not items:
                # Split by double spaces or line breaks that might indicate new items
                paragraphs = re.split(r'\s{2,}', text)
                
                for para in paragraphs:
                    para = para.strip()
                    if ':' in para and len(para) > 50:
                        colon_pos = para.find(':')
                        potential_title = para[:colon_pos].strip()
                        potential_desc = para[colon_pos+1:].strip()
                        
                        if (len(potential_title) > 10 and len(potential_title) < 150 and 
                            len(potential_desc) > 30):
                            items.append({
                                'title': potential_title,
                                'description': potential_desc
                            })
            
            # Method 3: Look for patterns like "Strong position in..." or "Exposure to..."
            if not items:
                # Common starting patterns for strengths and weaknesses
                patterns = [
                    r'(Strong [^:]+):\s*([^.]+(?:\.[^.]*)*)',
                    r'(Robust [^:]+):\s*([^.]+(?:\.[^.]*)*)', 
                    r'(Exposure to [^:]+):\s*([^.]+(?:\.[^.]*)*)',
                    r'([A-Z][^:]{20,100}):\s*([^.]+(?:\.[^.]*){2,})'
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.DOTALL)
                    for match in matches:
                        title = self.clean_text(match.group(1))
                        desc = self.clean_text(match.group(2))
                        if len(title) > 10 and len(desc) > 30:
                            items.append({
                                'title': title,
                                'description': desc
                            })
            
            self.logger.info(f"Extracted {len(items)} bullet items")
            return items
            
        except Exception as e:
            self.logger.error(f"Error extracting bullet items: {str(e)}")
            return []
    
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
            
            # Step 3: Find rating drivers section text
            section_text = self.find_rating_drivers_section(soup)
            
            if section_text:
                # Step 4: Extract strengths and weaknesses
                result = self.extract_strengths_and_weaknesses(section_text)
            else:
                self.logger.warning("Could not find rating drivers section")
                result = {"strengths": {}, "weaknesses": {}}
            
            # Validate results
            if not result['strengths'] and not result['weaknesses']:
                self.logger.warning("No strengths or weaknesses extracted - trying debug mode")
                self.debug_extraction(soup)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in main extraction pipeline: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def debug_extraction(self, soup: BeautifulSoup):
        """
        Debug method to help identify why extraction failed
        
        Args:
            soup (BeautifulSoup): Parsed HTML object
        """
        try:
            full_text = soup.get_text()
            
            # Check if key terms exist
            has_key_terms = {
                'Key Rating Drivers': 'Key Rating Drivers' in full_text,
                'Strengths': 'Strengths' in full_text or 'strengths' in full_text.lower(),
                'Weakness': 'Weakness' in full_text or 'weakness' in full_text.lower(),
                'Strong position': 'Strong position' in full_text,
                'Exposure to': 'Exposure to' in full_text
            }
            
            self.logger.info("=== DEBUG INFO ===")
            self.logger.info(f"Document length: {len(full_text)} characters")
            for term, found in has_key_terms.items():
                self.logger.info(f"'{term}' found: {found}")
            
            # Show a sample of text around "Strengths"
            if has_key_terms['Strengths']:
                strengths_pos = full_text.lower().find('strengths')
                if strengths_pos > -1:
                    sample = full_text[max(0, strengths_pos-100):strengths_pos+500]
                    self.logger.info(f"Sample text around 'Strengths':\n{sample}")
            
        except Exception as e:
            self.logger.error(f"Error in debug extraction: {str(e)}")
    
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
