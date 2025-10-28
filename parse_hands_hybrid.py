#!/usr/bin/env python3
"""
Parse mahjong hands from OCR text using both pytesseract and EasyOCR
"""
import json
import re
import cv2
import pytesseract
import easyocr

def extract_hands_hybrid(image_path):
    """Extract hands using both OCR engines and combine results"""
    
    # Initialize EasyOCR reader
    easyocr_reader = easyocr.Reader(['en'])
    
    # Load image
    img = cv2.imread(image_path)
    
    # Run pytesseract OCR
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    config = '--oem 3 --psm 6'
    pytesseract_lines = pytesseract.image_to_string(img, config=config).split('\n')
    
    # Run EasyOCR
    easyocr_results = easyocr_reader.readtext(img)
    easyocr_lines = []
    for bbox, text, confidence in easyocr_results:
        if confidence > 0.5:
            easyocr_lines.append(text)
    
    # Combine fragmented lines for EasyOCR
    combined_lines = []
    i = 0
    while i < len(easyocr_lines):
        current_line = easyocr_lines[i]
        
        if (len(current_line.split()) <= 2 and 
            any(char in current_line for char in 'FDNEWS') and
            i + 1 < len(easyocr_lines)):
            
            combined_line = current_line
            j = i + 1
            
            while j < len(easyocr_lines):
                next_line = easyocr_lines[j]
                
                if (any(char in next_line for char in '0123456789=') or
                    next_line.strip() in ['Suit)', 'Any', '25', 'X25']):
                    combined_line += ' ' + next_line
                    j += 1
                else:
                    break
            
            if j > i + 1 and len(combined_line.split()) <= 8:
                combined_lines.append(combined_line)
                i = j
                continue
        
        combined_lines.append(current_line)
        i += 1
    
    easyocr_lines = combined_lines
    
    # Use pytesseract for main processing (better for 2024 family)
    # But add EasyOCR results for Addition Hands
    text_lines = pytesseract_lines
    
    # Add EasyOCR Addition Hands if not found in pytesseract
    addition_hands_found = False
    for line in easyocr_lines:
        if 'ADDITION HANDS' in line.upper():
            addition_hands_found = True
            break
    
    if addition_hands_found:
        # Add EasyOCR Addition Hands lines - handle fragmented hands properly
        i = 0
        while i < len(easyocr_lines):
            line = easyocr_lines[i]
            
            # Look for FF patterns that are fragmented
            if (len(line.split()) <= 2 and 
                any(char in line for char in 'FDNEWS') and
                'FF' in line and
                i + 1 < len(easyocr_lines)):
                
                # Combine this line with the next few lines to form a complete hand
                combined_line = line
                j = i + 1
                
                # Keep combining until we have a complete hand
                while j < len(easyocr_lines):
                    next_line = easyocr_lines[j]
                    
                    # Check if this looks like continuation of the same hand
                    if (any(char in next_line for char in '0123456789=') or
                        next_line.strip() in ['Suit)', 'Any', '25', 'X25']):
                        combined_line += ' ' + next_line
                        j += 1
                        
                        # Stop if we have a complete hand (ends with Suit) or points
                        if (next_line.strip() in ['Suit)', '25', 'X25'] or 
                            'Suit)' in combined_line):
                            break
                    else:
                        break
                
                # Only add if we have a reasonable hand length
                if len(combined_line.split()) >= 4 and len(combined_line.split()) <= 8:
                    text_lines.append(combined_line)
                    i = j  # Skip all combined lines
                    continue
            
            # Add other Addition Hands related lines
            if 'ADDITION HANDS' in line.upper():
                text_lines.append(line)
            
            # Also add complete Addition Hands that don't need combining
            if ('FF' in line and '=' in line and 'Suit)' in line and 
                len(line.split()) >= 4 and len(line.split()) <= 8):
                text_lines.append(line)
            
            i += 1
    
    hands = []
    valid_chars = set('0123456789FDNEWS')
    current_family = ''
    
    section_headers = ['2025', '2468', 'ANY LIKE NUMBERS', '2024', 'ADDITION HANDS', 'LUCKY SEVENS']
    
    for line_text in text_lines:
        line_stripped = line_text.strip().upper()
        
        # Check if this is a section header
        is_header = False
        if '(' not in line_stripped:
            hand_chars_count = sum(1 for c in line_stripped if c in valid_chars)
            for header in section_headers:
                if header.upper() in line_stripped and hand_chars_count < 8:
                    current_family = header
                    is_header = True
                    break
        else:
            # Check if line with parentheses contains a section header
            # But be more specific - only treat as header if it's clearly a header line
            for header in section_headers:
                if header.upper() in line_stripped:
                    # Only treat as header if it's a simple header line (not a hand with the header name in it)
                    # Be very restrictive - only match clear header patterns
                    if (line_stripped.startswith(header.upper()) or 
                        line_stripped.endswith(header.upper()) or
                        f"({header.upper()})" in line_stripped):
                        current_family = header
                        is_header = True
                        break
        
        # Debug: print lines that might be 2024 hands
        if '2024' in line_stripped or ('N' in line_stripped and 'E' in line_stripped and 'W' in line_stripped and 'S' in line_stripped):
            print(f"Processing potential 2024 line: {repr(line_text)} -> {repr(line_stripped)}")
            print(f"  is_header: {is_header}")
            if is_header:
                print(f"  Skipped as header")
                continue
        
        if is_header:
            continue
        
        # Extract the hand pattern
        hand_part = line_text.split('(')[0].strip() if '(' in line_text else line_text.strip()
        
        if not hand_part:
            continue
        
        # OCR normalization
        hand_part = hand_part.replace('T', '1').replace('I', '1')
        
        # Skip lines that are clearly not mahjong hands
        if any(word in hand_part.upper() for word in ['NOTE:', 'DRAGON', 'USED', 'ZERO', 'MAY', 'BE', 'WITH', 'ADD111ON', 'VALUES']):
            continue
        
        # Extract note
        note = ''
        if '(' in line_text and ')' in line_text:
            note_match = re.search(r'\(([^)]+)\)', line_text)
            if note_match:
                note = note_match.group(1).strip()
        
        # Extract points and concealed status
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
        
        if not points:
            points = '25'
        
        # Check for multiple hands
        split_pattern = re.compile(r'-\d+-')
        hand_variants = split_pattern.split(hand_part)
        
        if len(hand_variants) == 1:
            # Try "-1:-" pattern
            colon_pattern = re.compile(r'-\d+:-')
            hand_variants = colon_pattern.split(hand_part)
        
        if len(hand_variants) == 1:
            or_pattern = re.compile(r'(?:\s|-)*or(?:\s|-)', re.IGNORECASE)
            hand_variants = or_pattern.split(hand_part)
        
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
                # Filter to only valid characters
                clean_part = part.replace(']', '').replace('+', '').replace('=', '')
                valid_part = ''.join([c for c in clean_part if c in valid_chars])
                if not valid_part:
                    continue
                
                # Split concatenated characters
                split_part = []
                i = 0
                while i < len(valid_part):
                    if valid_part[i].isdigit():
                        digits = ''
                        while i < len(valid_part) and valid_part[i].isdigit():
                            digits += valid_part[i]
                            i += 1
                        split_part.append(digits)
                    elif valid_part[i] in 'FDNEWS':
                        letter = ''
                        while i < len(valid_part) and valid_part[i] in 'FDNEWS':
                            letter += valid_part[i]
                            i += 1
                        split_part.append(letter)
                    else:
                        i += 1
                
                for subpart in split_part:
                    if char_count + len(subpart) <= 14:
                        hand_groups.append(subpart)
                        char_count += len(subpart)
                    else:
                        remaining = 14 - char_count
                        if remaining > 0:
                            hand_groups.append(subpart[:remaining])
                        break
                
                if char_count >= 14:
                    break
            
            total_chars = sum(len(g) for g in hand_groups)
            
            if total_chars < 14:
                continue
            
            # Create masks
            colorMask_parts = ['0' * len(part) for part in hand_groups]
            formatted_colorMask = ' '.join(colorMask_parts)
            
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
            
            formatted_jokerMask = ' '.join(jokerMask_parts)
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
    hands = extract_hands_hybrid('2024p1.png')
    
    print(f"Extracted {len(hands)} hands")
    for h in hands[:10]:
        print(f"{h['id']}. {h['hand']} | {h['colorMask']}")
    
    # Save to JSON file with custom formatting
    with open('card2024.json', 'w') as f:
        f.write('[\n')
        for i, hand in enumerate(hands):
            f.write('  {\n')
            f.write(f'      "id": {hand["id"]},\n')
            f.write(f'      "hand":      "{hand["hand"]}",\n')
            f.write(f'      "colorMask": "{hand["colorMask"]}",\n')
            f.write(f'      "jokerMask": "{hand["jokerMask"]}",\n')
            f.write(f'      "note": "{hand["note"]}",\n')
            f.write(f'      "family": "{hand["family"]}",\n')
            f.write(f'      "points": "{hand["points"]}",\n')
            f.write(f'      "concealed": {str(hand["concealed"]).lower()}\n')
            f.write('  }')
            if i < len(hands) - 1:
                f.write(',')
            f.write('\n')
        f.write(']\n')
    
    print(f"\nSaved to card2024.json")
