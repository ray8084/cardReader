#!/usr/bin/env python3
"""
Parse mahjong hands from OCR text using EasyOCR
"""
import json
import re
import cv2
import easyocr

def extract_hands(image_path):
    """Extract hands from OCR text using EasyOCR"""
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Load image
    img = cv2.imread(image_path)
    
    # Run OCR
    results = reader.readtext(img)
    
    # Convert results to text lines and combine fragmented lines
    text_lines = []
    for bbox, text, confidence in results:
        if confidence > 0.5:  # Only use high confidence results
            text_lines.append(text)
    
    # Combine fragmented lines (e.g., "FF 1111" followed by "6666 = 7777")
    combined_lines = []
    i = 0
    while i < len(text_lines):
        current_line = text_lines[i]
        
        # Check if this looks like a fragmented hand (short, contains FF/DDD/etc)
        if (len(current_line.split()) <= 2 and 
            any(char in current_line for char in 'FDNEWS') and
            i + 1 < len(text_lines)):
            
            # Look for continuation pattern: "FF 1111" + "6666 = 7777 (Any" + "Suit)" + "25"
            combined_line = current_line
            j = i + 1
            
            # Keep combining until we hit a complete hand or non-continuation
            while j < len(text_lines):
                next_line = text_lines[j]
                
                # Check if this looks like continuation
                if (any(char in next_line for char in '0123456789=') or
                    next_line.strip() in ['Suit)', 'Any', '25', 'X25']):
                    combined_line += ' ' + next_line
                    j += 1
                else:
                    break
            
            # Only combine if we found meaningful continuation and it's not too long
            if j > i + 1 and len(combined_line.split()) <= 8:
                combined_lines.append(combined_line)
                i = j  # Skip all combined lines
                continue
        
        combined_lines.append(current_line)
        i += 1
    
    text_lines = combined_lines
    
    hands = []
    valid_chars = set('0123456789FDNEWS')
    current_family = ''  # Track which section/family we're in
    
    # List of valid section headers
    section_headers = ['2025', '2468', 'ANY LIKE NUMBERS', '2024', 'ADDITION HANDS', 'LUCKY SEVENS']
    
    for line_text in text_lines:
        line_stripped = line_text.strip().upper()
        
        # Check if this is a section header
        # Headers are lines without '(' and without many hand characters
        # Exception: lines with parentheses that contain section header names
        is_header = False
        if '(' not in line_stripped:
            # Count hand characters
            hand_chars_count = sum(1 for c in line_stripped if c in valid_chars)
            for header in section_headers:
                if header.upper() in line_stripped and hand_chars_count < 8:
                    current_family = header
                    is_header = True
                    break
        else:
            # Check if line with parentheses contains a section header
            for header in section_headers:
                if header.upper() in line_stripped:
                    current_family = header
                    is_header = True
                    break
        
        # Skip header lines - don't process them as hands
        if is_header:
            continue
        
        # Extract the hand pattern (everything up to the paren or end)
        hand_part = line_text.split('(')[0].strip() if '(' in line_text else line_text.strip()
        
        if not hand_part:
            continue
        
        # OCR can misread '1' as 'T' or 'I', normalize them
        hand_part = hand_part.replace('T', '1').replace('I', '1')
        
        # Skip lines that are clearly not mahjong hands (like NOTE lines)
        if any(word in hand_part.upper() for word in ['NOTE:', 'DRAGON', 'USED', 'ZERO', 'MAY', 'BE', 'WITH', 'ADD111ON', 'VALUES']):
            continue
        
        # Extract note
        note = ''
        if '(' in line_text and ')' in line_text:
            note_match = re.search(r'\(([^)]+)\)', line_text)
            if note_match:
                note = note_match.group(1).strip()
        
        # Extract points value and concealed status from end of line
        points = ''
        concealed = False
        if line_text and ')' in line_text:
            text_after_note = line_text[line_text.rfind(')')+1:].strip()
            text_after_note_normalized = text_after_note.replace('€', 'C').replace('©', 'C')
            all_matches = list(re.finditer(r'([CcXx])\s*(\d+[DO]?)', text_after_note_normalized))
            if all_matches:
                last_match = all_matches[-1]
                concealed_char = last_match.group(1).upper()
                points = last_match.group(2)
                points = points.replace('D', '5').replace('O', '0')
                concealed = (concealed_char == 'C')
        elif line_text:
            text_normalized = line_text.replace('€', 'C').replace('©', 'C')
            all_matches = list(re.finditer(r'([CcXx])\s*(\d+[DO]?)', text_normalized))
            if all_matches:
                last_match = all_matches[-1]
                concealed_char = last_match.group(1).upper()
                points = last_match.group(2)
                points = points.replace('D', '5').replace('O', '0')
                concealed = (concealed_char == 'C')
        
        # If we couldn't extract points, default to 25
        if not points:
            points = '25'
        
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
                # Also remove common OCR artifacts like ']', '+', '='
                clean_part = part.replace(']', '').replace('+', '').replace('=', '')
                valid_part = ''.join([c for c in clean_part if c in valid_chars])
                if not valid_part:
                    continue
                
                # Split concatenated characters (e.g., "D1111" -> ["D", "1111"])
                # Look for transitions from letter to number or vice versa
                split_part = []
                i = 0
                while i < len(valid_part):
                    # Find a run of digits
                    if valid_part[i].isdigit():
                        digits = ''
                        while i < len(valid_part) and valid_part[i].isdigit():
                            digits += valid_part[i]
                            i += 1
                        split_part.append(digits)
                    # Find a run of letters
                    elif valid_part[i] in 'FDNEWS':
                        letter = ''
                        while i < len(valid_part) and valid_part[i] in 'FDNEWS':
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
            
            # Create jokerMask based on whether all chars are the same
            # - Pairs (length 2 with same char): '0's (cannot use jokers)
            # - Singles (length 1): '0's (cannot use jokers)
            # - All same chars (length 3+): '1's (can use jokers)
            # - Different chars: '0's (cannot use jokers)
            jokerMask_parts = []
            for part in hand_groups:
                if len(part) == 1:
                    jokerMask_parts.append('0')
                elif len(part) == 2 and len(set(part)) == 1:
                    jokerMask_parts.append('0' * len(part))
                elif len(set(part)) == 1:
                    jokerMask_parts.append('1' * len(part))
                else:
                    jokerMask_parts.append('0' * len(part))
            
            formatted_colorMask = ' '.join(colorMask_parts)
            formatted_jokerMask = ' '.join(jokerMask_parts)
            
            # Format the hand with proper spacing
            formatted = ' '.join(hand_groups)
            
            hands.append({
                'id': len(hands),
                'hand': formatted,
                'colorMask': formatted_colorMask,
                'jokerMask': formatted_jokerMask,
                'note': note,
                'family': current_family if current_family else '',
                'points': points,
                'concealed': concealed
            })
    
    return hands

if __name__ == '__main__':
    hands = extract_hands('2024p1.png')
    
    print(f"Extracted {len(hands)} hands")
    for h in hands[:10]:
        print(f"{h['id']}. {h['hand']} | {h['colorMask']}")
    
    # Save to JSON file
    with open('card2024_easyocr.json', 'w') as f:
        # Custom formatting for hand field alignment
        json_str = json.dumps({'hands': hands}, indent=2)
        lines = json_str.split('\n')
        result_lines = []
        for line in lines:
            if '"hand":' in line:
                new_line = line.replace(': "', ':      "')  # 6 spaces
                result_lines.append(new_line)
            else:
                result_lines.append(line)
        f.write('\n'.join(result_lines))
    
    print(f"\nSaved to card2024_easyocr.json")
