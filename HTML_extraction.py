import re
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import logging

class GenericRatingDriversExtractor:
    """
    Generic pipeline for extracting strengths and weaknesses from rating HTML documents
    Works in 3 clear steps: Read -> Extract Target Area -> Extract Sections -> Process to Dictionary
    """
    
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def step1_read_html(self, file_path: str) -> str:
        """
        STEP 1: Read HTML file with encoding detection
        
        Args:
            file_path (str): Path to HTML file
            
        Returns:
            str: Raw HTML content
        """
        print("🔄 STEP 1: Reading HTML file...")
        
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    html_content = file.read()
                
                print(f"✅ Successfully read HTML file with {encoding} encoding")
                print(f"📊 File size: {len(html_content):,} characters")
                
                # Show sample of HTML structure
                print("\n📄 HTML Structure Sample (first 500 chars):")
                print("-" * 50)
                print(html_content[:500])
                print("-" * 50)
                
                return html_content
                
            except (UnicodeDecodeError, UnicodeError):
                print(f"❌ Failed with {encoding}, trying next encoding...")
                continue
        
        raise Exception("❌ Could not decode file with any standard encoding")
    
    def step2_extract_target_area(self, html_content: str) -> str:
        """
        STEP 2: Extract the target area containing Key Rating Drivers
        
        Args:
            html_content (str): Raw HTML content
            
        Returns:
            str: Text content of target area
        """
        print("\n🔄 STEP 2: Extracting target area...")
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        full_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up multiple spaces
        full_text = re.sub(r'\s+', ' ', full_text)
        
        print(f"📊 Total document text: {len(full_text):,} characters")
        
        # Define multiple patterns to find the Key Rating Drivers section
        patterns = [
            # Pattern 1: From "Key Rating Drivers" to next major section
            r'Key Rating Drivers[^a-zA-Z]*(?:Detailed Description)?\s*(.*?)(?=Analytical Approach|Liquidity:|Outlook:|About the Company|Rating sensitivity|Any other information)',
            
            # Pattern 2: More flexible version
            r'Key Rating Drivers.*?:\s*(.*?)(?=Analytical Approach|Liquidity|Outlook|About)',
            
            # Pattern 3: Looking for the content between major sections
            r'(?:Key Rating Drivers|Detailed Description)(.*?)(?=Analytical Approach|Liquidity|Outlook)',
            
            # Pattern 4: Broad capture
            r'(Strengths?.*?Weakness.*?)(?=Liquidity|Outlook|Analytical|About|Rating sensitivity)'
        ]
        
        target_content = None
        pattern_used = None
        
        for i, pattern in enumerate(patterns, 1):
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            if match:
                target_content = match.group(1).strip()
                pattern_used = i
                print(f"✅ Found target area using pattern {i}")
                break
        
        if not target_content:
            print("❌ Could not find target area with standard patterns")
            print("🔍 Searching for fallback patterns...")
            
            # Fallback: Look for any section with both "Strengths" and "Weakness"
            if 'strengths' in full_text.lower() and 'weakness' in full_text.lower():
                # Find the boundaries manually
                strengths_pos = full_text.lower().find('strengths')
                
                # Look for end markers after strengths
                end_markers = ['liquidity', 'outlook', 'analytical approach', 'about the company']
                end_pos = len(full_text)
                
                for marker in end_markers:
                    marker_pos = full_text.lower().find(marker, strengths_pos)
                    if marker_pos > -1 and marker_pos < end_pos:
                        end_pos = marker_pos
                
                # Go back to find a reasonable start
                start_pos = max(0, strengths_pos - 500)
                target_content = full_text[start_pos:end_pos].strip()
                pattern_used = "fallback"
                print("✅ Found target area using fallback method")
        
        if target_content:
            print(f"📊 Target area size: {len(target_content):,} characters")
            print(f"🎯 Pattern used: {pattern_used}")
            
            print("\n📄 TARGET AREA CONTENT:")
            print("=" * 80)
            print(target_content)
            print("=" * 80)
            
            return target_content
        else:
            print("❌ Failed to extract target area")
            
            # Debug: Show what sections we can find
            print("\n🔍 DEBUG: Available sections in document:")
            common_sections = ['Key Rating Drivers', 'Strengths', 'Weakness', 'Liquidity', 'Outlook', 'Analytical Approach']
            for section in common_sections:
                if section.lower() in full_text.lower():
                    pos = full_text.lower().find(section.lower())
                    sample = full_text[max(0, pos-50):pos+100]
                    print(f"  ✓ '{section}' found at position {pos}: ...{sample}...")
                else:
                    print(f"  ❌ '{section}' not found")
            
            return ""
    
    def step3_extract_strengths_weaknesses(self, target_content: str) -> Dict[str, str]:
        """
        STEP 3: Extract strengths and weaknesses sections from target area
        
        Args:
            target_content (str): Target area text content
            
        Returns:
            Dict: Raw text of strengths and weaknesses sections
        """
        print("\n🔄 STEP 3: Extracting strengths and weaknesses sections...")
        
        sections = {"strengths_raw": "", "weaknesses_raw": ""}
        
        if not target_content:
            print("❌ No target content provided")
            return sections
        
        # Find strengths section
        strengths_patterns = [
            r'Strengths?\s*:?\s*(.*?)(?=Weakness|Liquidity|Outlook|$)',
            r'(?:^|\n)\s*Strengths?\s*:?\s*(.*?)(?=Weakness|Liquidity|Outlook|$)',
            r'Strengths?\s*(?::|\.)\s*(.*?)(?=Weakness|Liquidity|Outlook|$)'
        ]
        
        for i, pattern in enumerate(strengths_patterns, 1):
            match = re.search(pattern, target_content, re.DOTALL | re.IGNORECASE)
            if match:
                sections["strengths_raw"] = match.group(1).strip()
                print(f"✅ Found strengths section using pattern {i}")
                print(f"📊 Strengths content: {len(sections['strengths_raw'])} characters")
                break
        
        # Find weaknesses section
        weakness_patterns = [
            r'Weakness(?:es)?\s*:?\s*(.*?)(?=Liquidity|Outlook|Analytical|$)',
            r'(?:^|\n)\s*Weakness(?:es)?\s*:?\s*(.*?)(?=Liquidity|Outlook|Analytical|$)',
            r'Weakness(?:es)?\s*(?::|\.)\s*(.*?)(?=Liquidity|Outlook|Analytical|$)'
        ]
        
        for i, pattern in enumerate(weakness_patterns, 1):
            match = re.search(pattern, target_content, re.DOTALL | re.IGNORECASE)
            if match:
                sections["weaknesses_raw"] = match.group(1).strip()
                print(f"✅ Found weaknesses section using pattern {i}")
                print(f"📊 Weaknesses content: {len(sections['weaknesses_raw'])} characters")
                break
        
        # Print the extracted sections
        if sections["strengths_raw"]:
            print("\n📄 STRENGTHS SECTION:")
            print("-" * 60)
            print(sections["strengths_raw"])
            print("-" * 60)
        else:
            print("\n❌ No strengths section found")
        
        if sections["weaknesses_raw"]:
            print("\n📄 WEAKNESSES SECTION:")
            print("-" * 60)
            print(sections["weaknesses_raw"])
            print("-" * 60)
        else:
            print("\n❌ No weaknesses section found")
        
        return sections
    
    def step4_process_to_dictionary(self, sections: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        STEP 4: Process raw sections into structured dictionary
        
        Args:
            sections (Dict): Raw text sections
            
        Returns:
            Dict: Processed dictionary with title -> description mapping
        """
        print("\n🔄 STEP 4: Processing sections into dictionary...")
        
        result = {"strengths": {}, "weaknesses": {}}
        
        # Process strengths
        if sections.get("strengths_raw"):
            result["strengths"] = self.parse_bullet_points(sections["strengths_raw"], "strengths")
        
        # Process weaknesses
        if sections.get("weaknesses_raw"):
            result["weaknesses"] = self.parse_bullet_points(sections["weaknesses_raw"], "weaknesses")
        
        # Print final results
        print(f"\n✅ FINAL RESULTS:")
        print(f"📊 Strengths extracted: {len(result['strengths'])}")
        print(f"📊 Weaknesses extracted: {len(result['weaknesses'])}")
        
        if result["strengths"]:
            print("\n🔹 STRENGTHS:")
            for i, (title, desc) in enumerate(result["strengths"].items(), 1):
                print(f"  {i}. {title}")
                print(f"     → {desc[:100]}{'...' if len(desc) > 100 else ''}")
        
        if result["weaknesses"]:
            print("\n🔸 WEAKNESSES:")
            for i, (title, desc) in enumerate(result["weaknesses"].items(), 1):
                print(f"  {i}. {title}")
                print(f"     → {desc[:100]}{'...' if len(desc) > 100 else ''}")
        
        return result
    
    def parse_bullet_points(self, section_text: str, section_type: str) -> Dict[str, str]:
        """
        Parse bullet points from a section into title:description pairs
        
        Args:
            section_text (str): Raw section text
            section_type (str): Type of section (for logging)
            
        Returns:
            Dict: Title -> Description mapping
        """
        print(f"\n🔍 Parsing {section_type} bullet points...")
        
        bullet_points = {}
        
        # Method 1: Look for colon-separated items
        print("  Trying Method 1: Colon-separated items...")
        
        # Split by sentences that end with period and start with capital letter
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', section_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if ':' in sentence and len(sentence) > 30:  # Must have colon and be substantial
                colon_pos = sentence.find(':')
                title = sentence[:colon_pos].strip()
                description = sentence[colon_pos+1:].strip()
                
                # Validate the title and description
                if (len(title) > 5 and len(title) < 200 and 
                    len(description) > 20 and
                    not title.lower().startswith(('the ', 'this ', 'it ', 'there ', 'these '))):
                    
                    title = self.clean_text(title)
                    description = self.clean_text(description)
                    bullet_points[title] = description
                    print(f"    ✓ Found: {title}")
        
        # Method 2: If Method 1 didn't work, try paragraph-based splitting
        if not bullet_points:
            print("  Trying Method 2: Paragraph-based splitting...")
            
            # Try splitting by double spaces or multiple periods
            paragraphs = re.split(r'\n\s*\n|\.\s{2,}|(?<=\.)\s+(?=[A-Z][^.]{20,}:)', section_text)
            
            for para in paragraphs:
                para = para.strip()
                if ':' in para and len(para) > 30:
                    colon_pos = para.find(':')
                    title = para[:colon_pos].strip()
                    description = para[colon_pos+1:].strip()
                    
                    if len(title) > 5 and len(title) < 200 and len(description) > 20:
                        title = self.clean_text(title)
                        description = self.clean_text(description)
                        bullet_points[title] = description
                        print(f"    ✓ Found: {title}")
        
        # Method 3: Look for specific patterns (Strong/Robust/Exposure etc.)
        if not bullet_points:
            print("  Trying Method 3: Pattern-based extraction...")
            
            # Common patterns for financial rating documents
            patterns = [
                r'(Strong [^:]{10,100}):\s*([^.]+(?:\.[^.]*)*)',
                r'(Robust [^:]{10,100}):\s*([^.]+(?:\.[^.]*)*)',
                r'(Exposure to [^:]{10,100}):\s*([^.]+(?:\.[^.]*)*)',
                r'([A-Z][^:]{15,150}):\s*([^.]+(?:\.[^.]*){2,})'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, section_text, re.DOTALL)
                for match in matches:
                    title = self.clean_text(match.group(1))
                    description = self.clean_text(match.group(2))
                    
                    if len(title) > 5 and len(description) > 20:
                        bullet_points[title] = description
                        print(f"    ✓ Found: {title}")
        
        print(f"  📊 Total {section_type} found: {len(bullet_points)}")
        return bullet_points
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove section headers that might have leaked in
        prefixes_to_remove = [
            r'^Strengths?\s*:?\s*',
            r'^Weakness(?:es)?\s*:?\s*',
            r'^Key Rating Drivers\s*',
            r'^Detailed Description\s*'
        ]
        
        for prefix in prefixes_to_remove:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE).strip()
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'([.,;:])\s+', r'\1 ', text)
        
        return text
    
    def extract_rating_drivers(self, file_path: str) -> Dict[str, Dict[str, str]]:
        """
        Main pipeline: Execute all 4 steps in sequence
        
        Args:
            file_path (str): Path to HTML file
            
        Returns:
            Dict: Final extracted results
        """
        try:
            print("🚀 Starting Generic Rating Drivers Extraction Pipeline")
            print("=" * 80)
            
            # Step 1: Read HTML
            html_content = self.step1_read_html(file_path)
            
            # Step 2: Extract target area
            target_content = self.step2_extract_target_area(html_content)
            
            # Step 3: Extract strengths and weaknesses sections
            sections = self.step3_extract_strengths_weaknesses(target_content)
            
            # Step 4: Process to dictionary
            result = self.step4_process_to_dictionary(sections)
            
            print("\n" + "=" * 80)
            print("🎉 Pipeline completed successfully!")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            self.logger.error(f"Pipeline error: {str(e)}")
            return {"strengths": {}, "weaknesses": {}}
    
    def save_results(self, results: Dict[str, Dict[str, str]], output_path: str = "extracted_results.json"):
        """Save results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")

# Example usage
def main():
    """
    Example usage of the generic extraction pipeline
    """
    extractor = GenericRatingDriversExtractor()
    
    # Replace with your actual file path
    file_path = "Rating Rationale.html"
    
    try:
        # Run the complete pipeline
        results = extractor.extract_rating_drivers(file_path)
        
        # Save results
        extractor.save_results(results)
        
        return results
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please make sure the file path is correct")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None

if __name__ == "__main__":
    main()
