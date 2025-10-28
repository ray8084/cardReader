#!/usr/bin/env python3
"""
Parse mahjong hands from OCR text
"""
import json
import re
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def extract_hands(image_path):
    """Extract hands from OCR text"""
    
    # Load image
    img = cv2.imread(image_path)
    
    # Run OCR with better configuration for text with spaces
    config = '--oem 3 --psm 6'  # Uniform block of text
    text_lines = pytesseract.image_to_string(img, config=config).split('\n')
    
    hands = []
    valid_chars = set('0123456789FD')
    
    for line_text in text_lines:
        # Extract the hand pattern (everything up to the paren or end)
        hand_part = line_text.split('(')[0].strip() if '(' in line_text else line_text.strip()
        
        if not hand_part:
            continue
        
        # OCR can misread '1' as 'T' or 'I', normalize them
        hand_part = hand_part.replace('T', '1').replace('I', '1')
        
        # Extract note
        note = ''
        if '(' in line_text and ')' in line_text:
            note = line_text[line_text.find('(')+1:line_text.find(')')].strip()
        
        # Check for patterns that indicate multiple hands
        # Look for "-01-", "-or-", " or ", etc.
        
        # First try to split by "-01-" pattern
        split_pattern = re.compile(r'-\d+-')
        hand_variants = split_pattern.split(hand_part)
        
        # If that didn't split, try "or" patterns
        if len(hand_variants) == 1:
            or_pattern = re.compile(r'(?:\s|-)*or(?:\s|-)', re.IGNORECASE)
            hand_variants = or_pattern.split(hand_part)
        
        # If still no split, process as single hand
        if len(hand_variants) == 1:
            hand_variants = [hand_part]
        
        for variant in hand_variants:
            variant = variant.strip()
            if not variant:
                continue
            
            # Parse the hand variant
            parts = variant.split()
            
            # Filter to only valid mahjong characters and collect groups
            hand_groups = []
            char_count = 0
            
            for part in parts:
                # Filter to only valid characters (after T/I normalization)
                valid_part = ''.join([c for c in part if c in valid_chars])
                if not valid_part:
                    continue
                
                # Split concatenated characters (e.g., "D1111" -> ["D", "1111"])
                # Look for transitions from letter to number or vice versa
                split_part = []
                i = 0
                while i < len(valid_part):
                    # Find a run of digits
                    if valid_part[i] in '0123456789':
                        digits = ''
                        while i < len(valid_part) and valid_part[i] in '0123456789':
                            digits += valid_part[i]
                            i += 1
                        split_part.append(digits)
                    # Find a run of letters
                    elif valid_part[i] in 'FD':
                        letter = ''
                        while i < len(valid_part) and valid_part[i] in 'FD':
                            letter += valid_part[i]
                            i += 1
                        split_part.append(letter)
                    else:
                        i += 1
                
                for subpart in split_part:
                    # Check if adding this part would exceed 14 chars
                    if char_count + len(subpart) <= 14:
                        hand_groups.append(subpart)
                        char_count += len(subpart)
                    else:
                        # Need to truncate the last part
                        remaining = 14 - char_count
                        if remaining > 0:
                            hand_groups.append(subpart[:remaining])
                        break
                
                if char_count >= 14:
                    break
            
            # We need at least 14 characters total
            total_chars = sum(len(g) for g in hand_groups)
            
            if total_chars < 14:
                continue
            
            # Create colorMask that's all '0's matching the spacing of each hand
            colorMask_parts = []
            for part in hand_groups:
                colorMask_parts.append('0' * len(part))
            
            # Create jokerMask that's all '1's matching the spacing of each hand
            # EXCEPT pairs (2 identical chars) and singles (length 1) which become '0's
            jokerMask_parts = []
            for part in hand_groups:
                # Check if this is a pair (length 2 with same character) or single (length 1)
                if len(part) == 2 and len(set(part)) == 1:
                    # It's a pair, use '0's (cannot use jokers in pairs)
                    jokerMask_parts.append('0' * len(part))
                elif len(part) == 1:
                    # It's a single character, use '0's (cannot use jokers for singles)
                    jokerMask_parts.append('0')
                else:
                    # Not a pair or single, use '1's (can use jokers)
                    jokerMask_parts.append('1' * len(part))
            
            # Build formatted hand with proper spacing
            formatted = ' '.join(hand_groups)
            formatted_colorMask = ' '.join(colorMask_parts)
            formatted_jokerMask = ' '.join(jokerMask_parts)
            
            hands.append({
                'id': len(hands) + 1,
                'hand': formatted,
                'colorMask': formatted_colorMask,
                'jokerMask': formatted_jokerMask,
                'note': note
            })
    
    return hands

if __name__ == '__main__':
    hands = extract_hands('2025p1.png')
    
    print(f"Extracted {len(hands)} hands")
    for h in hands[:10]:
        print(f"{h['id']}. {h['hand']} | {h['colorMask']}")
    
    with open('card2025.json', 'w') as f:
        json_str = json.dumps({'hands': hands}, indent=2)
        # Add 6 spaces after colon for hand only
        lines = json_str.split('\n')
        result_lines = []
        for line in lines:
            if '"hand":' in line:
                # Replace ": " with ":      " (colon + 6 spaces) for hand only
                new_line = line.replace(': "', ':      "')
                result_lines.append(new_line)
            else:
                result_lines.append(line)
        f.write('\n'.join(result_lines))
    
    print(f"\nSaved to card2025.json")

