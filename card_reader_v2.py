#!/usr/bin/env python3
"""
Simplified mahjong card reader with pixel-accurate color detection
"""
import json
import re
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def detect_character_color(image, char_info, hsv, mask_red, mask_green):
    """Sample pixels around a character to determine its color"""
    x, y, w, h = char_info
    
    # Expand sampling region for better accuracy
    expand = 8
    y1 = max(0, y - expand)
    y2 = min(hsv.shape[0], y + h + expand)
    x1 = max(0, x - expand)
    x2 = min(hsv.shape[1], x + w + expand)
    
    # Sample color masks
    region_red = mask_red[y1:y2, x1:x2]
    region_green = mask_green[y1:y2, x1:x2]
    total_pixels = region_red.size
    
    if total_pixels == 0:
        return '0'
    
    red_pixels = np.sum(region_red > 0)
    green_pixels = np.sum(region_green > 0)
    
    red_ratio = red_pixels / total_pixels
    green_ratio = green_pixels / total_pixels
    
    # Determine color based on higher ratio with threshold
    if green_ratio > 0.05 and green_ratio > red_ratio:
        return 'g'
    elif red_ratio > 0.05 and red_ratio > green_ratio:
        return 'r'
    else:
        return '0'

def extract_hands_with_colors(image_path):
    """Extract mahjong hands from image with color detection"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    # Create HSV color masks
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red color ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    # Green color range
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Run OCR to get full text
    config = '--oem 3 --psm 11'
    text = pytesseract.image_to_string(image, config=config)
    
    # Run OCR with data for positions
    data = pytesseract.image_to_data(
        image, config=config, output_type=pytesseract.Output.DICT
    )
    
    # Parse text to find hands
    lines = text.split('\n')
    hands = []
    valid_chars = set('0123456789FD')
    
    for line in lines:
        # Extract only valid characters
        clean = ''.join([c for c in line if c in valid_chars])
        
        # Valid hands have 13-14 characters
        if 10 <= len(clean) <= 15:
            # Pad or truncate to 14
            if len(clean) < 14:
                clean = clean + '0' * (14 - len(clean))
            else:
                clean = clean[:14]
            
            # Now find the positions of these characters in the OCR data
            # and build color mask
            mask = ''
            char_positions = []
            
            # Get bounding boxes for this line
            line_boxes = []
            for i in range(len(data['text'])):
                text_item = data['text'][i].strip()
                if not text_item:
                    continue
                
                x, y = data['left'][i], data['top'][i]
                w, h = data['width'][i], data['height'][i]
                conf = data['conf'][i]
                
                if conf > 30:
                    for j, char in enumerate(text_item):
                        if char in valid_chars:
                            char_w = w // len(text_item)
                            line_boxes.append({
                                'char': char,
                                'x': x + j * char_w,
                                'y': y,
                                'w': char_w,
                                'h': h
                            })
            
            # For each character in our hand, find matching char and sample color
            for hand_char in clean:
                # Find this character in our positions (simple first-match)
                found = False
                for pos_idx, char_info in enumerate(line_boxes):
                    if char_info['char'] == hand_char and pos_idx < len(line_boxes):
                        color = detect_character_color(
                            image,
                            (char_info['x'], char_info['y'], char_info['w'], char_info['h']),
                            hsv, mask_red, mask_green
                        )
                        mask += color
                        found = True
                        break
                
                if not found:
                    mask += '0'
            
            # Ensure mask is 14 chars
            while len(mask) < 14:
                mask += '0'
            mask = mask[:14]
            
            # Format
            formatted_hand = f"{clean[0:4]} {clean[4:8]} {clean[8:11]} {clean[11:14]}"
            formatted_mask = f"{mask[0:4]} {mask[4:8]} {mask[8:11]} {mask[11:14]}"
            
            # Extract note (everything after first paren)
            note = ''
            if '(' in line and ')' in line:
                note_start = line.find('(')
                note_end = line.find(')')
                if note_start < note_end:
                    note = line[note_start+1:note_end].strip()
            
            hands.append({
                'id': len(hands) + 1,
                'hand': formatted_hand,
                'mask': formatted_mask,
                'note': note
            })
    
    return hands

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='2025p1.png')
    parser.add_argument('--out', default='card2025.json')
    args = parser.parse_args()
    
    print("Processing image...")
    hands = extract_hands_with_colors(args.image)
    
    print(f"Extracted {len(hands)} hands")
    
    # Display first few hands
    for h in hands[:5]:
        print(f"\nHand {h['id']}:")
        print(f"  Text: {h['hand']}")
        print(f"  Mask: {h['mask']}")
    
    # Save to JSON
    output = {'hands': hands}
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {args.out}")

