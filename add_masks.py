#!/usr/bin/env python3
"""
add_masks.py - add mask entries to NMJL raw JSON files.

The script reads a JSON file containing a list of hands and adds a "mask" field
directly under each hand's "text" entry. The mask mirrors the spacing of the
text while converting all tile characters to "0". Special separator characters
("+", "=", "x", "X") are left blank (spaces) in the mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

SEPARATOR_CHARS = {"+", "=", "x", "X"}


def build_mask(text: str) -> str:
    """Return a color mask string aligned with the given text."""
    return "".join(
        " " if ch == " " or ch in SEPARATOR_CHARS else "0" for ch in text
    )


def build_joker_mask(text: str) -> str:
    """Return a joker mask string aligned with the given text."""
    return "".join(
        " " if ch == " " or ch in SEPARATOR_CHARS else "1" for ch in text
    )


def insert_mask(hand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new hand dict with the mask key inserted directly beneath text.

    If the hand already has a mask key, it will be replaced.
    """
    new_hand: Dict[str, Any] = {}
    for key, value in hand.items():
        if key in {"mask", "colorMask", "jokerMask"}:
            # Skip existing mask-style keys; we'll regenerate them.
            continue

        new_hand[key] = value
        if key == "text":
            mask_text = str(value)
            new_hand["colorMask"] = build_mask(mask_text)
            new_hand["jokerMask"] = build_joker_mask(mask_text)
    if "text" not in hand:
        # If no text field existed, just leave the dict untouched.
        new_hand.setdefault("colorMask", "")
        new_hand.setdefault("jokerMask", "")
    return new_hand


def process_file(filename: Path) -> None:
    """Load, update, and overwrite the JSON file."""
    with filename.open("r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)

    hands: List[Dict[str, Any]] = data.get("hands", [])
    updated_hands = [insert_mask(hand) for hand in hands]
    data["hands"] = updated_hands

    json_content = json.dumps(data, indent=2, ensure_ascii=False)
    json_content = json_content.replace('"text": "', '"text":      "')

    with filename.open("w", encoding="utf-8") as fh:
        fh.write(json_content)
        fh.write("\n")

    print(f"Updated {len(updated_hands)} hands in '{filename}' with mask fields.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: add_masks.py <path-to-json>")
        raise SystemExit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"File not found: {target}")
        raise SystemExit(1)

    process_file(target)


if __name__ == "__main__":
    main()

