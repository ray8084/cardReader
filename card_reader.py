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
        Handles lines with multiple hands separated by "-01-"
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
            
            # Check if line contains multiple hands separated by "-01-"
            if '-01-' in line:
                # Split into multiple hands
                parts = line.split('-01-')
                for part in parts:
                    self._parse_single_hand(part, hands, valid_chars, seen_hands, line_num)
            else:
                self._parse_single_hand(line, hands, valid_chars, seen_hands, line_num)
        
        print(f"\nTotal: {len(hands)} unique hands found")
        return hands
    
    def _parse_single_hand(self, line: str, hands: List, valid_chars: set, seen_hands: set, line_num: int):
        """Parse a single hand from a line"""
        # Extract all valid characters
        chars = [c for c in line if c in valid_chars]
        
        # Need at least 13-14 valid characters for a hand
        if len(chars) < 13:
            return
        
        # Need exactly 14 chars - take what we have
        if len(chars) == 14:
            hand_str = ''.join(chars)
            hand_chars = chars
        elif len(chars) > 14:
            # Take first 14, but be smarter about cleaning artifacts
            hand_str = ''.join(chars[:14])
            hand_chars = chars[:14]
        elif len(chars) == 13:
            # Only 13 chars - pad with one more char or accept
            # Try to find one more character in surrounding context
            # For now, just use what we have
            hand_str = ''.join(chars)
            hand_chars = chars + ['0']  # Pad with dummy
        elif len(chars) == 15:
            # Has extra char, take first 14
            hand_str = ''.join(chars[:14])
            hand_chars = chars[:14]
        else:
            return
        
        # Skip if we've seen this exact hand
        if hand_str in seen_hands:
            return
        seen_hands.add(hand_str)
        
        # Extract the note
        note = ""
        if '(' in line:
            paren_start = line.find('(')
            note = line[paren_start:].replace('(', '').replace(')', '').strip()
            note = re.sub(r'\s+', ' ', note)
            if note.endswith('.'):
                note = note[:-1]
        
        # Format hand with spaces
        hand_formatted = hand_str
        if '(' in line:
            before_paren = line.split('(')[0].strip()
            hand_parts = []
            current_part = ""
            for char in before_paren:
                if char in valid_chars:
                    current_part += char
                elif char.isspace() and current_part:
                    hand_parts.append(current_part)
                    current_part = ""
            if current_part:
                hand_parts.append(current_part)
            
            if len(hand_parts) >= 2:
                hand_formatted = ' '.join(hand_parts)
        
        if hand_formatted == hand_str and len(hand_str) == 14:
            hand_formatted = f"{hand_str[0:4]} {hand_str[4:8]} {hand_str[8:11]} {hand_str[11:14]}"
        
        # Detect colors by actually sampling pixels
        color_mask = self.detect_colors_actual(line, hand_str)
        mask_with_spaces = self.add_spaces_to_mask(hand_formatted, color_mask)
        
        hands.append({
            'id': len(hands) + 1,
            'hand': hand_formatted,
            'mask': mask_with_spaces,
            'note': note or 'No description captured'
        })
        print(f"Hand {len(hands)}: {hand_str} (raw: {len(hand_str)} chars, formatted: {len(hand_formatted.replace(' ', ''))} chars)")
        if note:
            print(f"  Note: {note[:50]}")
        
        print(f"\nTotal: {len(hands)} unique hands found")
        return hands
    
    def detect_colors_actual(self, line_text: str, hand_str: str) -> str:
        """Detect colors by actually sampling the image at character positions"""
        # This will use HSV color segmentation from the full image
        # We already have color masks created in detect_colors_from_image
        # Just call that to get actual pixel sampling
        return self.detect_colors_from_image(line_text)
    
    def detect_colors_from_image(self, line_text: str) -> str:
        """Detect colors by creating color masks and sampling those regions"""
        valid_chars = set('0123456789FD')
        chars = [c for c in line_text if c in valid_chars]
        
        if len(chars) < 14:
            return '0' * 14
        
        chars = chars[:14]
        
        try:
            # Create HSV color masks for the entire image
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            # Red color ranges
            lower_red1 = np.array([0, 30, 30])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 30, 30])
            upper_red2 = np.array([180, 255, 255])
            mask_red = cv2.bitwise_or(
                cv2.inRange(hsv, lower_red1, upper_red1),
                cv2.inRange(hsv, lower_red2, upper_red2)
            )
            
            # Green color range  
            lower_green = np.array([35, 30, 30])
            upper_green = np.array([85, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            # Run OCR to get character positions
            config = '--oem 3 --psm 6'
            data = pytesseract.image_to_data(self.image, config=config, output_type=pytesseract.Output.DICT)
            
            # Build all character positions
            all_chars_with_pos = []
            for i in range(len(data["text"])):
                text = data["text"][i]
                if text and data["conf"][i] > 40:  # Only high confidence
                    for j, c in enumerate(text):
                        if c in valid_chars:
                            all_chars_with_pos.append({
                                'char': c,
                                'left': data["left"][i] + j * (data["width"][i] // len(text)) if len(text) > 0 else data["left"][i],
                                'top': data["top"][i],
                                'width': data["width"][i] // len(text) if len(text) > 0 else data["width"][i],
                                'height': data["height"][i]
                            })
            
            # For each character in our hand, find matching position and sample color
            color_mask = []
            ocr_match_idx = 0
            
            for i, char in enumerate(chars):
                best_match_idx = None
                best_distance = float('inf')
                
                # Find the best matching character in a small window
                search_window = 10
                for j in range(ocr_match_idx, min(ocr_match_idx + search_window, len(all_chars_with_pos))):
                    if all_chars_with_pos[j]['char'] == char:
                        best_match_idx = j
                        best_distance = abs(j - ocr_match_idx)
                        break
                
                if best_match_idx is not None:
                    pos = all_chars_with_pos[best_match_idx]
                    left, top = pos['left'], pos['top']
                    width, height = pos['width'], pos['height']
                    
                    # Sample a region around this character
                    center_x = left + width // 2
                    center_y = top + height // 2
                    
                    # Expand sampling region
                    y1 = max(0, center_y - 15)
                    y2 = min(hsv.shape[0], center_y + 15)
                    x1 = max(0, center_x - 20)
                    x2 = min(hsv.shape[1], center_x + 20)
                    
                    if y2 > y1 and x2 > x1:
                        # Sample green and red masks at this region
                        green_regions = np.sum(mask_green[y1:y2, x1:x2] > 0)
                        red_regions = np.sum(mask_red[y1:y2, x1:x2] > 0)
                        total_pixels = (y2 - y1) * (x2 - x1)
                        
                        green_ratio = green_regions / total_pixels if total_pixels > 0 else 0
                        red_ratio = red_regions / total_pixels if total_pixels > 0 else 0
                        
                        # Lower threshold for detection
                        if green_ratio > 0.08:
                            color_mask.append('g')
                        elif red_ratio > 0.08:
                            color_mask.append('r')
                        else:
                            color_mask.append('0')
                        
                        ocr_match_idx = best_match_idx + 1
                    else:
                        color_mask.append('0')
                else:
                    color_mask.append('0')
            
            # Ensure 14 chars
            while len(color_mask) < 14:
                color_mask.append('0')
            
            return ''.join(color_mask[:14])
            
        except Exception as e:
            print(f"Error in color detection: {e}")
            import traceback
            traceback.print_exc()
            return '0' * 14
    
    def detect_character_colors(self, line_text: str, line_y: int, image_width: int) -> str:
        """Detect colors for each character by analyzing pixels at that position"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges in HSV
        # Red (can wrap around hue=0)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Green
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Analyze each character position
        colors = []
        chars = [c for c in line_text if c not in ' ']
        
        # Estimate character width based on line
        if len(chars) > 0:
            estimated_char_width = image_width // len(chars)
            
            for i, char in enumerate(chars):
                # Estimate position
                char_x = i * estimated_char_width + estimated_char_width // 2
                
                # Sample a small region around this position
                if line_y < hsv.shape[0] and 0 <= char_x < hsv.shape[1]:
                    region = hsv[max(0, line_y-10):min(hsv.shape[0], line_y+10),
                                 max(0, char_x-20):min(hsv.shape[1], char_x+20)]
                    
                    if region.size > 0:
                        # Check for red
                        mask_red1 = cv2.inRange(region, lower_red1, upper_red1)
                        mask_red2 = cv2.inRange(region, lower_red2, upper_red2)
                        red_count = np.sum(mask_red1) + np.sum(mask_red2)
                        
                        # Check for green
                        mask_green = cv2.inRange(region, lower_green, upper_green)
                        green_count = np.sum(mask_green)
                        
                        # Decide color
                        if red_count > 50:
                            colors.append('r')
                        elif green_count > 50:
                            colors.append('g')
                        else:
                            colors.append('0')
                    else:
                        colors.append('0')
                else:
                    colors.append('0')
        else:
            colors = ['0'] * 14
        
        return ''.join(colors[:14])
    
    def add_spaces_to_mask(self, hand_formatted: str, mask: str) -> str:
        """Add spaces to mask to match spacing in hand string"""
        if not hand_formatted or not mask:
            return mask
        
        # Count non-space characters in hand
        non_space_chars = ''.join([c for c in hand_formatted if c != ' '])
        
        # Ensure mask has enough characters
        if len(mask) < len(non_space_chars):
            mask = mask.ljust(len(non_space_chars), '0')
        
        # Build result with spaces
        result = ""
        mask_idx = 0
        for char in hand_formatted:
            if char == ' ':
                result += ' '
            else:
                if mask_idx < len(mask):
                    result += mask[mask_idx]
                    mask_idx += 1
        
        return result
    
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
