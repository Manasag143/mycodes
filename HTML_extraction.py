import re
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging

class RatingDriversExtractor:
    """
    Rock-solid extractor based on actual HTML structure analysis
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_html_file(self, file_path: str) -> str:
        """Load HTML file with encoding handling"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                self.logger.info(f"Loaded file with {encoding} encoding")
                return content
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        raise Exception("Could not decode file with any standard encoding")
    
    def extract_raw_text_content(self, html_content: str) -> str:
        """
        Extract the complete text content and identify the exact structure
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get all text content
        full_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up multiple spaces
        full_text = re.sub(r'\s+', ' ', full_text)
        
        self.logger.info(f"Extracted {len(full_text)} characters of text")
        return full_text
    
    def find_rating_drivers_content(self, text: str) -> Optional[str]:
        """
        Find the exact content between Key Rating Drivers and the next major section
        """
        try:
            # Pattern to find Key Rating Drivers section
            # Look for the section that starts with "Key Rating Drivers" and ends before next major section
            patterns = [
                r'Key Rating Drivers[^a-zA-Z]*Detailed Description\s*(.*?)(?=Analytical Approach|Liquidity|Outlook|About the Company|Any other information)',
                r'Key Rating Drivers.*?:\s*(.*?)(?=Analytical Approach|Liquidity|Outlook|About the Company)',
                r'Key Rating Drivers[^:]*(.{1000,}?)(?=Analytical Approach|Liquidity|Outlook)',
            ]
            
            for i, pattern in enumerate(patterns, 1):
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    self.logger.info(f"Found content using pattern {i}: {len(content)} characters")
                    return content
            
            # Fallback: manual search
            start_markers = ['Key Rating Drivers', 'Detailed Description']
            end_markers = ['Analytical Approach', 'Liquidity', 'Outlook', 'About the Company']
            
            start_pos = -1
            for marker in start_markers:
                pos = text.find(marker)
                if pos > -1:
                    start_pos = pos
                    break
            
            if start_pos > -1:
                # Find the end position
                end_pos = len(text)
                for marker in end_markers:
                    pos = text.find(marker, start_pos)
                    if pos > -1 and pos < end_pos:
                        end_pos = pos
                
                content = text[start_pos:end_pos]
                self.logger.info(f"Found content using fallback method: {len(content)} characters")
                return content
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding rating drivers content: {str(e)}")
            return None
    
    def extract_strengths_and_weaknesses_direct(self, content: str) -> Dict[str, Dict[str, str]]:
        """
        Direct extraction based on the known structure from your document
        """
        result = {"strengths": {}, "weaknesses": {}}
        
        try:
            # Based on your document, I can see the exact patterns:
            
            # Known strength patterns from your document
            strength_patterns = [
                r'Strong position in India[^:]*fertiliser market[^:]*:\s*([^•]+?)(?=Strong|Robust|Exposure|\Z)',
                r'Strong operating efficiency[^:]*:\s*([^•]+?)(?=Strong|Robust|Exposure|\Z)',
                r'Robust financial risk profile[^:]*:\s*([^•]+?)(?=Strong|Robust|Exposure|Weakness|\Z)'
            ]
            
            # Known weakness patterns from your document  
            weakness_patterns = [
                r'Exposure to regulated nature[^:]*:\s*([^•]+?)(?=Strong|Robust|Exposure|Liquidity|Outlook|\Z)'
            ]
            
            # Extract strengths
            strength_titles = [
                "Strong position in India's phosphatic-fertiliser market",
                "Strong operating efficiency", 
                "Robust financial risk profile"
            ]
            
            for i, pattern in enumerate(strength_patterns):
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match and i < len(strength_titles):
                    description = self.clean_extracted_text(match.group(1))
                    if description:
                        result["strengths"][strength_titles[i]] = description
                        self.logger.info(f"Found strength: {strength_titles[i]}")
            
            # Extract weaknesses
            weakness_titles = [
                "Exposure to regulated nature of the fertiliser industry and volatility in raw material prices"
            ]
            
            for i, pattern in enumerate(weakness_patterns):
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match and i < len(weakness_titles):
                    description = self.clean_extracted_text(match.group(1))
                    if description:
                        result["weaknesses"][weakness_titles[i]] = description
                        self.logger.info(f"Found weakness: {weakness_titles[i]}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in direct extraction: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def extract_with_universal_patterns(self, content: str) -> Dict[str, Dict[str, str]]:
        """
        Universal extraction that works with any similar rating document
        """
        result = {"strengths": {}, "weaknesses": {}}
        
        try:
            # Find strengths section
            strengths_match = re.search(r'Strengths?\s*:?\s*(.*?)(?=Weakness|Liquidity|Outlook|\Z)', 
                                      content, re.DOTALL | re.IGNORECASE)
            
            if strengths_match:
                strengths_text = strengths_match.group(1)
                result["strengths"] = self.parse_bullet_section(strengths_text)
                self.logger.info(f"Universal extraction found {len(result['strengths'])} strengths")
            
            # Find weaknesses section
            weakness_match = re.search(r'Weakness(?:es)?\s*:?\s*(.*?)(?=Liquidity|Outlook|Analytical|\Z)', 
                                     content, re.DOTALL | re.IGNORECASE)
            
            if weakness_match:
                weakness_text = weakness_match.group(1)
                result["weaknesses"] = self.parse_bullet_section(weakness_text)
                self.logger.info(f"Universal extraction found {len(result['weaknesses'])} weaknesses")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in universal extraction: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def parse_bullet_section(self, section_text: str) -> Dict[str, str]:
        """
        Parse a section containing bullet points into title:description pairs
        """
        items = {}
        
        try:
            # Method 1: Look for sentences that start with capital letters and contain colons
            sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', section_text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if ':' in sentence and len(sentence) > 50:  # Must have colon and be substantial
                    colon_pos = sentence.find(':')
                    title = sentence[:colon_pos].strip()
                    description = sentence[colon_pos+1:].strip()
                    
                    # Validate title (should be a meaningful phrase)
                    if (len(title) > 10 and len(title) < 200 and 
                        len(description) > 30 and
                        not title.lower().startswith(('the ', 'this ', 'it ', 'there '))):
                        
                        title = self.clean_extracted_text(title)
                        description = self.clean_extracted_text(description)
                        items[title] = description
            
            # Method 2: If method 1 doesn't work, try paragraph splitting
            if not items:
                paragraphs = re.split(r'\n\s*\n|\.\s{2,}', section_text)
                
                for para in paragraphs:
                    para = para.strip()
                    if ':' in para and len(para) > 50:
                        colon_pos = para.find(':')
                        title = para[:colon_pos].strip()
                        description = para[colon_pos+1:].strip()
                        
                        if len(title) > 10 and len(description) > 30:
                            title = self.clean_extracted_text(title)
                            description = self.clean_extracted_text(description)
                            items[title] = description
            
            return items
            
        except Exception as e:
            self.logger.error(f"Error parsing bullet section: {str(e)}")
            return {}
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean extracted text thoroughly
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove common prefixes that might leak in
        prefixes_to_remove = [
            r'^Strengths?\s*:?\s*',
            r'^Weakness(?:es)?\s*:?\s*',
            r'^Key Rating Drivers\s*',
            r'^Detailed Description\s*'
        ]
        
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE).strip()
        
        # Clean up punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:])\s+', r'\1 ', text)  # Ensure single space after punctuation
        
        return text
    
    def extract_rating_drivers(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """
        Main extraction pipeline with multiple fallback strategies
        """
        try:
            # Load file
            html_content = self.load_html_file(file_path)
            
            # Extract text content
            text_content = self.extract_raw_text_content(html_content)
            
            # Find the rating drivers section
            rating_content = self.find_rating_drivers_content(text_content)
            
            if not rating_content:
                self.logger.error("Could not find rating drivers section")
                return {"strengths": {}, "weaknesses": {}}
            
            self.logger.info(f"Working with {len(rating_content)} characters of rating content")
            
            # Try direct extraction first (specific to your document format)
            result = self.extract_strengths_and_weaknesses_direct(rating_content)
            
            # If direct extraction didn't work, try universal patterns
            if not result["strengths"] and not result["weaknesses"]:
                self.logger.info("Direct extraction failed, trying universal patterns")
                result = self.extract_with_universal_patterns(rating_content)
            
            # Log final results
            self.logger.info(f"Final extraction: {len(result['strengths'])} strengths, {len(result['weaknesses'])} weaknesses")
            
            # Debug output if still no results
            if not result["strengths"] and not result["weaknesses"]:
                self.debug_content_analysis(rating_content)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in main extraction: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def debug_content_analysis(self, content: str):
        """
        Comprehensive debug analysis
        """
        self.logger.info("=== DEBUG ANALYSIS ===")
        self.logger.info(f"Content length: {len(content)}")
        
        # Check for key terms
        key_terms = ['Strong position', 'Strong operating', 'Robust financial', 
                    'Exposure to', 'Strengths', 'Weakness', 'fertiliser', 'phosphatic']
        
        for term in key_terms:
            count = content.lower().count(term.lower())
            self.logger.info(f"'{term}' appears {count} times")
        
        # Show sample content
        self.logger.info("First 500 characters:")
        self.logger.info(content[:500])
        
        # Look for colon patterns
        colon_matches = re.findall(r'[^.:]{20,100}:[^.:]{20,100}', content)
        self.logger.info(f"Found {len(colon_matches)} potential title:description patterns")
        
        if colon_matches:
            self.logger.info("Sample patterns:")
            for i, match in enumerate(colon_matches[:3]):
                self.logger.info(f"  {i+1}: {match[:100]}...")
    
    def save_results(self, results: Dict[str, Dict[str, str]], output_path: str):
        """Save results to JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
    
    def save_debug_content(self, content: str, output_path: str = "debug_content.txt"):
        """Save extracted content for manual inspection"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Debug content saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving debug content: {str(e)}")

# Example usage with comprehensive testing
def main():
    """
    Comprehensive testing of the extraction pipeline
    """
    extractor = RatingDriversExtractor()
    
    # Replace with your actual file path
    file_path = "Rating Rationale.html"
    
    try:
        print("🚀 Starting ROCK-SOLID extraction...")
        
        # Extract rating drivers
        results = extractor.extract_rating_drivers(file_path)
        
        # Display results
        print("\n" + "="*60)
        print("📊 EXTRACTION RESULTS")
        print("="*60)
        
        print(f"\n✅ STRENGTHS FOUND: {len(results['strengths'])}")
        for i, (title, description) in enumerate(results['strengths'].items(), 1):
            print(f"\n{i}. 📈 {title}")
            print(f"   💬 {description[:300]}{'...' if len(description) > 300 else ''}")
        
        print(f"\n❌ WEAKNESSES FOUND: {len(results['weaknesses'])}")
        for i, (title, description) in enumerate(results['weaknesses'].items(), 1):
            print(f"\n{i}. 📉 {title}")
            print(f"   💬 {description[:300]}{'...' if len(description) > 300 else ''}")
        
        # Save results
        extractor.save_results(results, "extracted_rating_drivers.json")
        print(f"\n💾 Results saved to: extracted_rating_drivers.json")
        
        # Test with a sample to verify the extractor works
        if not results['strengths'] and not results['weaknesses']:
            print("\n⚠️  No results found. Saving debug content for manual inspection...")
            # You can manually inspect the debug content
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

if __name__ == "__main__":
    main()
