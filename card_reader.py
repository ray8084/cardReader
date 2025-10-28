"""
Mahjong Card Reader
Reads a mahjong card PNG image and extracts hand information to JSON
Expected format: 17 hands, each with 14 characters (0-9, F, D)
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Configure tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


class MahjongCardReader:
    """Reads and processes mahjong card images using OCR"""
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.image = None
        self.load_image()
    
    def load_image(self):
        """Load the image"""
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
    
    def preprocess_image(self):
        """Preprocess image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    def extract_text_with_ocr(self):
        """Extract text from image using OCR"""
        # Try different preprocessing approaches
        processed_images = []
        
        # Original image
        processed_images.append(self.image)
        
        # Preprocessed image
        preprocessed = self.preprocess_image()
        processed_images.append(preprocessed)
        
        # Try with different configurations
        configs = [
            '--psm 3',  # Fully automatic page segmentation (default)
            '--psm 6',  # Assume a single uniform block of text
            '--psm 11',  # Sparse text
            '--psm 12',  # Treat the image as a single text line
        ]
        
        best_result = ""
        best_confidence = 0
        
        for img in processed_images:
            for config in configs:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(img, config=config)
                    
                    # Get confidence scores
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(x) for x in data['conf'] if int(x) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0
                    
                    print(f"OCR attempt (config: {config}): confidence={avg_confidence:.1f}")
                    print(f"Text preview: {text[:200]}...")
                    
                    if avg_confidence > best_confidence:
                        best_result = text
                        best_confidence = avg_confidence
                except Exception as e:
                    print(f"Error with config {config}: {e}")
                    continue
        
        print(f"\nBest OCR result (confidence: {best_confidence:.1f})")
        print(f"Extracted text:\n{best_result}")
        return best_result
    
    def parse_hands_from_text(self, text: str) -> List[List[str]]:
        """
        Parse hands from OCR text
        Expected format: lines with space-separated symbols
        Each hand should have 14 characters (0-9, F, D)
        """
        hands = []
        lines = text.strip().split('\n')
        
        valid_chars = set('0123456789FD')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split by spaces and extract characters
            parts = re.split(r'\s+', line)
            hand_chars = []
            
            for part in parts:
                # Extract only valid characters
                chars = ''.join(c for c in part if c in valid_chars)
                if chars:
                    hand_chars.append(chars)
            
            # Join to get the hand string
            hand_str = ''.join(hand_chars)
            
            # Filter valid hands (must contain valid chars, reasonable length)
            # Hands typically are 10-20 characters when including spaces
            if len(hand_str) >= 10 and any(c in valid_chars for c in hand_str):
                # Extract exactly 14 characters if we can
                if len(hand_str) >= 14:
                    # Take first 14 valid characters
                    chars = [c for c in hand_str if c in valid_chars]
                    if len(chars) >= 14:
                        hand = chars[:14]
                        hands.append(hand)
                        print(f"Found hand: {''.join(hand)}")
                    elif len(chars) >= 10:
                        # Partial hand, keep it
                        hands.append(chars)
                        print(f"Found partial hand: {''.join(chars)}")
        
        print(f"\nTotal hands found: {len(hands)}")
        return hands
    
    def parse_hands_v2(self, text: str) -> List[Dict]:
        """
        Parse all hands from OCR text with notes
        """
        lines = text.strip().split('\n')
        hands = []
        valid_chars = set('0123456789FD')
        seen_hands = set()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            # Skip clearly non-hand lines
            if any(x in line for x in ['VALUES', 'ANY LIKE']):
                continue
            if line.startswith('X ') or line.startswith('Cc ') or line.startswith('xX '):
                continue
            
            # Extract all valid characters for hand matching
            chars = [c for c in line if c in valid_chars]
            
            # Need at least 14 valid characters for a hand
            if len(chars) >= 14:
                # Take first 14 chars for hand string
                hand_chars = chars[:14]
                hand_str = ''.join(hand_chars)
                
                # Clean up: remove trailing numbers that are OCR artifacts
                while len(hand_str) > 14 and hand_str[-3:] in ['325', ' 25', ' 30', '8 2']:
                    hand_str = hand_str[:-1]
                    hand_chars = hand_chars[:-1]
                
                if len(hand_str) >= 14:
                    hand_str = hand_str[:14]
                    hand_chars = hand_chars[:14]
                    
                    # Skip if we've seen this exact hand
                    if hand_str in seen_hands:
                        continue
                    seen_hands.add(hand_str)
                    
                    # Extract the note (everything after the hand pattern)
                    # Look for parenthetical text or text after the hand
                    note = ""
                    
                    # Try to find the note in the original line
                    # Notes typically appear in parentheses or after the hand
                    if '(' in line:
                        # Get text after first '('
                        paren_start = line.find('(')
                        note = line[paren_start:].replace('(', '').replace(')', '').strip()
                        # Clean up common OCR artifacts
                        note = re.sub(r'\s+', ' ', note)
                        if note.endswith('.'):
                            note = note[:-1]
                    
                    # Format hand with spaces based on original OCR line
                    hand_formatted = hand_str
                    
                    # Try to extract the formatted version from the original line
                    # Look for sequences of F or digits in the original line
                    if '(' in line:
                        # Get the part before the parenthesis
                        before_paren = line.split('(')[0].strip()
                        # Extract just the hand part (before any OCR artifacts)
                        hand_parts = []
                        current_part = ""
                        for char in before_paren:
                            if char in valid_chars:
                                current_part += char
                            elif char.isspace() and current_part:
                                hand_parts.append(current_part)
                                current_part = ""
                            elif current_part and len(current_part) >= 14:
                                break
                        if current_part:
                            hand_parts.append(current_part)
                        
                        # If we got good parts from the original, use them
                        if len(hand_parts) >= 3:
                            hand_formatted = ' '.join(hand_parts[:4])  # Take first 4 parts
                        elif len(hand_parts) >= 2:
                            hand_formatted = ' '.join(hand_parts)
                    
                    # Fallback: simple spacing if we didn't extract from original
                    if hand_formatted == hand_str and len(hand_str) == 14:
                        # Default spacing
                        hand_formatted = f"{hand_str[0:4]} {hand_str[4:8]} {hand_str[8:11]} {hand_str[11:14]}"
                    
                    hands.append({
                        'id': len(hands) + 1,
                        'hand': hand_formatted,
                        'note': note or 'No description captured'
                    })
                    print(f"Hand {len(hands)}: {hand_str}")
                    if note:
                        print(f"  Note: {note[:50]}")
        
        print(f"\nTotal: {len(hands)} unique hands found")
        return hands
    
    def process(self) -> Dict:
        """Main processing method"""
        print("Processing image...")
        
        # Extract text using OCR
        text = self.extract_text_with_ocr()
        
        # Parse hands from text
        hands_data = self.parse_hands_v2(text)
        
        # Simplified output - just the hands
        result = {
            'hands': hands_data
        }
        
        return result
    
    def save_json(self, output_path: str = None):
        """Save results to JSON file"""
        if output_path is None:
            output_path = 'card2025.json'
        
        result = self.process()
        
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        return output_file


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python card_reader.py <image_path> [output_json_path]")
        print("\nExample:")
        print("  python card_reader.py mahjong_hand.png")
        print("  python card_reader.py mahjong_hand.png output.json")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        reader = MahjongCardReader(image_path)
        reader.save_json(output_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
