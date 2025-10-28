"""
Mahjong Card Reader
Reads a mahjong card PNG image and extracts hand information to JSON
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image


class MahjongCardReader:
    """Reads and processes mahjong card images"""
    
    # Mahjong tile categories
    TILE_CATEGORIES = {
        'characters': ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m'],
        'dots': ['1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p'],
        'bamboos': ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s'],
        'winds': ['E', 'S', 'W', 'N'],  # East, South, West, North
        'dragons': ['D', 'H', 'C']  # Dragon, Haku (White), Chun (Red)
    }
    
    def __init__(self, image_path: str):
        """Initialize with image path"""
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.image = None
        self.gray_image = None
        self.load_image()
    
    def load_image(self):
        """Load and preprocess the image"""
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
        
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def detect_tiles(self) -> List[Dict]:
        """
        Detect mahjong tiles in the image
        This is a simplified version - you'll need to customize based on your specific card format
        """
        tiles = []
        
        # Apply threshold to find tile regions
        _, thresh = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (potential tile regions)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to find tile-like regions
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Adjust these thresholds based on your actual tile size
            if 500 < area < 50000:  # Filter by area
                aspect_ratio = w / h if h > 0 else 0
                
                # Tiles are roughly rectangular
                if 0.3 < aspect_ratio < 3.0:
                    tile_region = self.image[y:y+h, x:x+w]
                    tile_data = self.recognize_tile(tile_region)
                    
                    tiles.append({
                        'position': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                        'tile': tile_data,
                        'confidence': 0.8  # Placeholder
                    })
        
        return tiles
    
    def recognize_tile(self, tile_region: np.ndarray) -> str:
        """
        Recognize individual tile from image region
        This is a placeholder - you'll need to implement actual tile recognition
        Options:
        1. Template matching with reference images
        2. Deep learning model (CNN)
        3. OCR combined with pattern recognition
        """
        # TODO: Implement actual tile recognition
        # This could use:
        # - Template matching
        # - Feature detection (SIFT, ORB)
        # - Machine learning model
        # - OCR for numbers/characters
        
        return "unknown"  # Placeholder
    
    def parse_hands(self, tiles: List[Dict]) -> Dict:
        """
        Parse detected tiles into mahjong hands
        A mahjong hand typically has 13-14 tiles
        """
        hands = {
            'hand_count': 0,
            'hands': []
        }
        
        # Group tiles that might form hands
        # This is a simplified approach
        if len(tiles) >= 13:
            # For now, create a single hand with all detected tiles
            hand = {
                'tiles': [t['tile'] for t in tiles[:14]],  # Standard hand is 14 tiles
                'position': tiles[0]['position'] if tiles else None
            }
            hands['hands'].append(hand)
            hands['hand_count'] = 1
        
        return hands
    
    def process(self) -> Dict:
        """Main processing method"""
        print("Processing image...")
        tiles = self.detect_tiles()
        print(f"Detected {len(tiles)} potential tiles")
        
        hands = self.parse_hands(tiles)
        
        result = {
            'image_path': str(self.image_path),
            'tiles_detected': len(tiles),
            'hands': hands,
            'raw_tiles': tiles
        }
        
        return result
    
    def save_json(self, output_path: str = None):
        """Save results to JSON file"""
        if output_path is None:
            output_path = self.image_path.stem + '_hands.json'
        
        result = self.process()
        
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
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
        sys.exit(1)


if __name__ == "__main__":
    main()

