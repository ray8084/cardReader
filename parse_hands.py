#!/usr/bin/env python3
"""
Parse mahjong hands from OCR text and apply color detection
"""
import json
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def detect_char_color(image, x, y, w, h, mask_red, mask_green):
    """Detect if character is red or green by sampling pixels"""
    # Expand region for better sampling
    expand = 5
    y1 = max(0, y - expand)
    y2 = min(mask_red.shape[0], y + h + expand)
    x1 = max(0, x - expand)
    x2 = min(mask_red.shape[1], x + w + expand)
    
    region_red = mask_red[y1:y2, x1:x2]
    region_green = mask_green[y1:y2, x1:x2]
    total = region_red.size
    
    if total == 0:
        return '0'
    
    red_ratio = np.sum(region_red > 0) / total
    green_ratio = np.sum(region_green > 0) / total
    
    if green_ratio > 0.03 and green_ratio > red_ratio * 1.2:
        return 'g'
    elif red_ratio > 0.03 and red_ratio > green_ratio * 1.2:
        return 'r'
    return '0'

def extract_hands(image_path):
    """Extract hands with color detection"""
    
    # Load image and create color masks
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red masks
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    # Green mask
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Run OCR
    config = '--oem 3 --psm 11'
    text_lines = pytesseract.image_to_string(img, config=config).split('\n')
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    
    # Parse hands
    hands = []
    valid_chars = set('0123456789FD')
    
    for line_text in text_lines:
        # First try to extract patterns from the original line
        # Pattern: groups of chars separated by spaces
        # Look for patterns like "222 0000 222 5555" or "FFFF 2025 222 222"
        
        # Extract the hand pattern (everything up to the paren or end)
        hand_part = line_text.split('(')[0].strip() if '(' in line_text else line_text.strip()
        
        # Parse the original line to get spaced groups
        # Example: "222 0000 222 5555" -> ["222", "0000", "222", "5555"]
        parts = hand_part.split()
        hand_groups = []
        for part in parts:
            # Only take valid mahjong characters
            valid_part = ''.join([c for c in part if c in valid_chars])
            if valid_part:
                hand_groups.append(valid_part)
        
        # Combine groups into hand text
        full_hand = ''.join(hand_groups)
        
        # We need 14 characters total
        if len(full_hand) < 13:
            continue
            
        hand_text = full_hand[:14]
        
        # Build hand positions
        positions = []
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            if not word or data['conf'][i] < 30:
                continue
            
            x, y = data['left'][i], data['top'][i]
            w, h = data['width'][i], data['height'][i]
            
            # Split into chars
            char_w = w // len(word) if len(word) > 0 else w
            for j, char in enumerate(word):
                if char in valid_chars:
                    positions.append({
                        'char': char,
                        'x': x + j * char_w,
                        'y': y,
                        'w': char_w,
                        'h': h
                    })
        
        # Build mask by finding each character and sampling color
        mask = ''
        pos_idx = 0
        for char in hand_text:
            found_color = False
            # Try to find matching char in positions
            for i in range(pos_idx, len(positions)):
                if positions[i]['char'] == char:
                    color = detect_char_color(
                        img,
                        positions[i]['x'],
                        positions[i]['y'],
                        positions[i]['w'],
                        positions[i]['h'],
                        mask_red,
                        mask_green
                    )
                    mask += color
                    pos_idx = i + 1
                    found_color = True
                    break
            
            if not found_color:
                mask += '0'
        
        # Extract note
        note = ''
        if '(' in line_text and ')' in line_text:
            note = line_text[line_text.find('(')+1:line_text.find(')')].strip()
        
        # Preserve original spacing from hand_part
        # Build formatted hand and mask with original spacing
        formatted_parts = []
        mask_parts = []
        for i, part in enumerate(hand_groups):
            formatted_parts.append(part)
            # Get corresponding mask for this part
            start_idx = sum(len(p) for p in hand_groups[:i])
            end_idx = start_idx + len(part)
            mask_parts.append(mask[start_idx:end_idx])
        
        # Join with spaces matching the original
        formatted = ' '.join(formatted_parts)
        formatted_mask = ' '.join(mask_parts)
        
        hands.append({
            'id': len(hands) + 1,
            'hand': formatted,
            'mask': formatted_mask,
            'note': note
        })
    
    return hands

if __name__ == '__main__':
    hands = extract_hands('2025p1.png')
    
    print(f"Extracted {len(hands)} hands")
    for h in hands[:10]:
        print(f"{h['id']}. {h['hand']} | {h['mask']}")
    
    with open('card2025.json', 'w') as f:
        json.dump({'hands': hands}, f, indent=2)
    
    print(f"\nSaved to card2025.json")

