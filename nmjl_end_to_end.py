#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        return None, e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, RuntimeError("OPENAI_API_KEY is not set")
    try:
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, e

@dataclass
class OCRChar:
    ch: str
    x: int
    y: int
    w: int
    h: int
    color: str

@dataclass
class OCRLine:
    text: str
    mask: str
    bbox: Tuple[int,int,int,int]
    note: str = ""

def deskew(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h,w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)

def color_masks_hsv(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 90, 80]); upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([170, 90, 80]); upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    lower_green = np.array([40, 60, 60]); upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    lower_black = np.array([0, 0, 0]);   upper_black = np.array([180, 90, 110])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((2,2),np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel, iterations=1)
    return {"r": mask_red, "g": mask_green, "0": mask_black}

def apply_mask_for_ocr(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray

def line_boxes_from_any(mask_dict: Dict[str, np.ndarray]) -> List[Tuple[int,int,int,int]]:
    combined = np.clip(mask_dict["r"] | mask_dict["g"] | mask_dict["0"], 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,3))
    dil = cv2.dilate(combined, kernel, iterations=1)
    cnts = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = [b for b in boxes if 12 <= b[3] <= 300 and b[2] > 25]
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def tesseract_to_chars(gray_img: np.ndarray) -> List[OCRChar]:
    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1 -l eng"
    data = pytesseract.image_to_data(gray_img, config=config, output_type=pytesseract.Output.DICT)
    chars: List[OCRChar] = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i]
        if not txt or not txt.strip(): continue
        x,y,w,h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        per_char = max(1, len(txt))
        cw = max(1, int(w/per_char))
        for j,ch in enumerate(txt):
            cx = x + j*cw
            chars.append(OCRChar(ch=ch, x=cx, y=y, w=cw, h=h, color="?"))
    return chars

def assign_colors(chars: List[OCRChar], mask_dict: Dict[str,np.ndarray]) -> None:
    for c in chars:
        cx = c.x + c.w//2
        cy = c.y + c.h//2
        best = 0; color = "0"
        for k in ("r","g","0"):
            m = mask_dict[k]
            cx_c = min(max(0,cx), m.shape[1]-1)
            cy_c = min(max(0,cy), m.shape[0]-1)
            val = m[cy_c, cx_c]
            if val > best:
                best = val; color = k
        c.color = color

def group_chars_by_line(chars: List[OCRChar], y_thresh: int=18) -> List[List[OCRChar]]:
    if not chars: return []
    chars_sorted = sorted(chars, key=lambda c: (c.y, c.x))
    lines = [[chars_sorted[0]]]
    for ch in chars_sorted[1:]:
        if abs(ch.y - lines[-1][-1].y) <= y_thresh:
            lines[-1].append(ch)
        else:
            lines.append([ch])
    for ln in lines: ln.sort(key=lambda c: c.x)
    return lines

def chars_to_text_and_mask(line_chars: List[OCRChar]) -> Tuple[str,str,Tuple[int,int,int,int]]:
    if not line_chars: return "","",(0,0,0,0)
    text_parts=[]; mask_parts=[]
    avg_w = np.mean([c.w for c in line_chars]) if line_chars else 8
    last_x2 = line_chars[0].x
    y_top = min(c.y for c in line_chars); y_bot = max(c.y+c.h for c in line_chars)
    for c in line_chars:
        gap = c.x - last_x2
        if gap > avg_w * 0.9:
            text_parts.append(" "); mask_parts.append(" ")
        text_parts.append(c.ch)
        mask_parts.append(c.color if c.color in ("r","g","0") else "0")
        last_x2 = c.x + c.w
    x1 = min(c.x for c in line_chars); x2 = max(c.x + c.w for c in line_chars)
    return "".join(text_parts), "".join(mask_parts), (x1, y_top, x2-x1, y_bot-y_top)

def normalize_text(s: str) -> str:
    t = s.strip()
    t = " ".join(t.split())
    if "2O25" in t or "20O5" in t: t = t.replace("O","0")
    return t

def extract_note(t: str) -> Tuple[str,str]:
    note = ""
    if "(" in t and ")" in t and t.rfind("(") < t.rfind(")"):
        i,j = t.rfind("("), t.rfind(")")
        note = t[i+1:j].strip()
        t = (t[:i] + t[j+1:]).strip()
    return t, note

def validate_hand(text: str, mask: str) -> Tuple[bool,str]:
    if len(text)!=len(mask): return False, "len mismatch"
    allowed = set("rg0 ")
    if any(c not in allowed for c in mask): return False, "bad mask char"
    for i,ch in enumerate(text):
        if ch==" " and mask[i]!=" ": return False, "space misalign"
        if ch!=" " and mask[i]==" ": return False, "mask space on char"
    return True, ""

def color_ocr_lines(image_path: str) -> List[OCRLine]:
    bgr = cv2.imread(image_path)
    if bgr is None: raise FileNotFoundError(image_path)
    bgr = denoise(deskew(bgr))
    masks = color_masks_hsv(bgr)
    boxes = line_boxes_from_any(masks)
    lines: List[OCRLine] = []
    for (x,y,w,h) in boxes:
        roi = bgr[y:y+h, x:x+w]
        combined = (masks["r"][y:y+h, x:x+w] | masks["g"][y:y+h, x:x+w] | masks["0"][y:y+h, x:x+w]).astype(np.uint8)
        gray = apply_mask_for_ocr(roi, combined)
        chars = tesseract_to_chars(gray)
        local_masks = {k: masks[k][y:y+h, x:x+w] for k in masks}
        assign_colors(chars, local_masks)
        grouped = group_chars_by_line(chars)
        for g in grouped:
            text, mask, bb = chars_to_text_and_mask(g)
            text = normalize_text(text)
            text, note = extract_note(text)
            mask = mask[:len(text)]
            ok, err = validate_hand(text, mask)
            if not ok:
                # Minimal repair: collapse spaces in text and compress mask accordingly
                t2 = " ".join(text.split())
                # rebuild mask for t2
                def realign(t, m):
                    out = []; mi=0
                    for ch in t:
                        if ch==" ": out.append(" ")
                        else:
                            while mi<len(m) and m[mi]==" ": mi+=1
                            out.append(m[mi] if mi<len(m) else "0"); mi+=1
                    return "".join(out)
                m2 = realign(t2, mask)
                text, mask = t2, m2
            lines.append(OCRLine(text=text, mask=mask, bbox=(x+bb[0], y+bb[1], bb[2], bb[3]), note=note))
    lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return lines

FULL_COLOR_SYSTEM_PROMPT = """You are an expert NMJL Card Reader performing pixel-aware normalization.
Task: Convert noisy OCR line items into perfectly structured JSON hands with aligned color masks and optional notes.
Rules:
- Return only: {"hands":[{"text":"...","mask":"...","note":"...","source":{"image":"image_1","region":[x,y,w,h]}}]}
- text: single spaces between groups; no leading/trailing spaces; keep original order/case.
- mask: same length as text; allowed chars: r,g,0, space; preserve space alignment 1:1.
- Colors: r=red, g=green, 0=black/navy (already precomputed; do not re-guess colors).
- Notes: move trailing descriptors like "(Any 1 Suit)" / "Any 1 Suit" into note (no parentheses) and remove from text.
- Two-hand rows: if a line obviously contains two separate hands, emit two entries.
- NMJL strict mode: American Mahjong only; do not introduce non-NMJL concepts; do not reorder or invent tokens.
Validation: ensure len(text)==len(mask); mask alphabet; space alignment. If invalid, fix and re-validate before returning.
"""

def batch_to_chatgpt(client, model: str, items: List[OCRLine], image_label: str) -> Dict[str, Any]:
    lines_block = []
    for i, ln in enumerate(items, 1):
        x,y,w,h = ln.bbox
        # Escape quotes
        tx = ln.text.replace('"','\\"')
        mk = ln.mask.replace('"','\\"')
        nt = ln.note.replace('"','\\"')
        lines_block.append(f'{i}) text="{tx}"\n   mask="{mk}"\n   note="{nt}"\n   region=[{x},{y},{w},{h}]')
    user_msg = "Normalize these NMJL OCR lines into JSON hands.\n\nReturn only the JSON object.\n\nLINES:\n" + "\n\n".join(lines_block)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":FULL_COLOR_SYSTEM_PROMPT},
                  {"role":"user","content":user_msg}],
        temperature=0
    )
    txt = resp.choices[0].message.content.strip()
    data = json.loads(txt)
    for h in data.get("hands", []):
        src = h.get("source", {})
        if "image" not in src: src["image"] = image_label
        h["source"] = src
    return data

def validate_final(data: Dict[str, Any]) -> List[str]:
    errs = []
    for idx, h in enumerate(data.get("hands", []), 1):
        text, mask = h.get("text",""), h.get("mask","")
        ok, err = validate_hand(text, mask)
        if not ok:
            errs.append(f"Hand {idx}: {err} :: {text} :: {mask}")
    return errs

def filter_valid_hands(lines: List[OCRLine]) -> List[Dict[str,Any]]:
    """Filter to only valid mahjong hands"""
    valid_chars = set('0123456789FD')
    hands = []
    
    for ln in lines:
        # Extract only valid characters
        clean_text = ''.join([c for c in ln.text if c in valid_chars])
        
        # Valid mahjong hands should have exactly 14 characters
        if len(clean_text) >= 14:
            # Take first 14 chars
            clean_text = clean_text[:14]
            
            # Get corresponding mask
            clean_mask = ""
            char_count = 0
            for i, c in enumerate(ln.text):
                if c in valid_chars:
                    if char_count < len(clean_text) and i < len(ln.mask):
                        clean_mask += ln.mask[i]
                    elif char_count < len(clean_text):
                        clean_mask += '0'
                    char_count += 1
                    if char_count >= len(clean_text):
                        break
            
            # Ensure mask is same length as text
            while len(clean_mask) < len(clean_text):
                clean_mask += '0'
            clean_mask = clean_mask[:len(clean_text)]
            
            hands.append({
                "id": len(hands) + 1,
                "hand": clean_text,
                "mask": clean_mask,
                "note": ln.note
            })
    
    print(f"Filtered {len(hands)} valid hands from {len(lines)} OCR lines")
    return hands

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to NMJL card image")
    ap.add_argument("--out", default="hands.json", help="Output JSON")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL","gpt-4o-mini"), help="ChatGPT model name")
    ap.add_argument("--batch", type=int, default=15, help="Lines per API batch")
    args = ap.parse_args()

    print("Processing image with color-aware OCR...")
    lines = color_ocr_lines(args.image)
    print(f"Found {len(lines)} OCR lines")

    client, err = get_openai_client()
    
    if client is not None:
        print("Using ChatGPT for normalization...")
        final_hands: List[Dict[str,Any]] = []
        for i in range(0, len(lines), args.batch):
            chunk = lines[i:i+args.batch]
            try:
                data = batch_to_chatgpt(client, args.model, chunk, "image_1")
                errs = validate_final(data)
                if errs:
                    print("[WARN] Validation errors:")
                    for e in errs: print(" -", e)
                final_hands.extend(data.get("hands", []))
            except Exception as e:
                print(f"[ERROR] ChatGPT error: {e}, falling back to filtered output")
                final_hands = filter_valid_hands(lines)
                break
    else:
        print(f"[INFO] OpenAI unavailable: {err}. Using filtered Python-only output.")
        final_hands = filter_valid_hands(lines)
    
    # Format for our expected output
    output_hands = []
    for h in final_hands:
        hand_text = h.get("text", h.get("hand", ""))
        # Format with spaces
        if len(hand_text) == 14:
            formatted = f"{hand_text[0:4]} {hand_text[4:8]} {hand_text[8:11]} {hand_text[11:14]}"
        else:
            formatted = hand_text
        
        mask_text = h.get("mask", "")
        # Add spaces to mask to match hand
        result = ""
        mask_idx = 0
        for char in formatted:
            if char == ' ':
                result += ' '
            else:
                if mask_idx < len(mask_text):
                    result += mask_text[mask_idx]
                    mask_idx += 1
                else:
                    result += '0'
        
        output_hands.append({
            "id": len(output_hands) + 1,
            "hand": formatted,
            "mask": result,
            "note": h.get("note", "")
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"hands": output_hands}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output_hands)} hands to {args.out}")

if __name__ == "__main__":
    main()
