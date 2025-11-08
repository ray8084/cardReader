#!/usr/bin/env python3
"""
generate.py - Shared helpers for Mahjong card generation.

Provides a reusable base class with helper methods used by the
year-specific generator scripts (e.g., generate2014.py).
"""

from __future__ import annotations

import json
from typing import List, Optional, Any, Dict

from hand import Hand


class CardGeneratorBase:
    """Base class containing reusable helpers for card generators."""

    def __init__(self, year: int):
        self.year = year
        self.hand_list: List[Hand] = []
        self._build_all_hands()

    def _build_all_hands(self) -> None:
        """Subclasses must implement the logic to construct all hands."""
        raise NotImplementedError("Subclasses must implement _build_all_hands()")

    def get_year(self) -> str:
        """Return the card's year as a string."""
        return str(self.year)

    def add_hand(
        self,
        hand_id: int,
        text: str,
        mask: str,
        joker_mask: str,
        note: str,
        family: str,
        concealed: bool,
        points: int,
    ) -> Hand:
        """Instantiate a Hand and append it to the generator's list."""
        hand = Hand(hand_id, text, mask, joker_mask, note, family, concealed, points)
        self.hand_list.append(hand)
        return hand

    def tile_id_to_name(self, tile_id: int) -> str:
        """Convert a tile ID back to a readable name."""
        from tile import TILE_MAPPINGS

        for tile_name, tile_ids in TILE_MAPPINGS.items():
            if tile_id in tile_ids:
                return tile_name
        return f"T{tile_id}"

    def condense_tile_set(self, tile_ids: List[int]) -> str:
        """Convert tile IDs to a compact, readable representation."""
        tile_counts: Dict[str, int] = {}
        for tile_id in tile_ids:
            tile_name = self.tile_id_to_name(tile_id)
            tile_counts[tile_name] = tile_counts.get(tile_name, 0) + 1

        parts: List[str] = []
        for tile_name, count in sorted(tile_counts.items()):
            if count <= 4:
                parts.append(tile_name * count)
            else:
                parts.append(f"{tile_name}x{count}")
        return " ".join(parts)

    def export_to_json(self, filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export the card data to a JSON file matching legacy card format."""
        target_file = filename or f"card{self.year}.json"

        json_hands: List[Dict[str, Any]] = []

        family_mapping = {
            "Consecutive Run": "Runs",
            "Winds - Dragons": "Winds",
            "Singles and Pairs": "Singles & Pairs",
        }

        for index, hand in enumerate(self.hand_list):
            hand_data: Dict[str, Any] = {
                "id": index,
                "text": hand.text,
                "mask": hand.mask,
                "jokerMask": hand.joker_mask,
                "note": hand.note,
                "family": family_mapping.get(hand.family, hand.family),
                "concealed": hand.concealed,
                "points": hand.points,
                "tileSets": [],
            }

            print(f"DEBUG: Hand {hand.id} ({hand.text}) has {len(hand.tile_sets)} tile sets")
            for i, tile_set in enumerate(hand.tile_sets):
                print(f"DEBUG: Adding tile set {i}: {tile_set}")
                hand_data["tileSets"].append(tile_set)

            json_hands.append(hand_data)

        valid_hands = sum(1 for hand in self.hand_list if hand.tile_sets)
        total_tiles = sum(len(hand.tile_sets) * 14 for hand in self.hand_list)

        with open(target_file, "w", encoding="utf-8") as fh:
            json_str = json.dumps(json_hands, indent=2, ensure_ascii=False, separators=(",", ": "))

            lines = json_str.split("\n")
            result_lines: List[str] = []
            i = 0
            in_tile_sets = False
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                if '"tileSets": []' in line:
                    result_lines.append(line)
                    i += 1
                    continue

                if '"tileSets": [' in line:
                    in_tile_sets = True
                    result_lines.append(line)
                    i += 1
                    continue

                if in_tile_sets and stripped.startswith("["):
                    indent = line[: line.index("[")]
                    values: List[str] = []
                    trailing_comma = ""
                    i += 1
                    while i < len(lines):
                        inner = lines[i].strip()
                        if inner.startswith("]"):
                            trailing_comma = "," if inner.endswith(",") else ""
                            i += 1
                            break
                        value = inner.rstrip(",")
                        if value:
                            values.append(value)
                        i += 1
                    joined = ", ".join(values)
                    result_lines.append(f"{indent}[{joined}]{trailing_comma}")
                    continue

                if in_tile_sets and stripped.startswith("]"):
                    in_tile_sets = False
                    result_lines.append(line)
                    i += 1
                    continue

                result_lines.append(line)
                i += 1

            fh.write("\n".join(result_lines))

        print(f"JSON file '{target_file}' exported successfully!")
        print(f"Total hands: {len(self.hand_list)}")
        print(f"Hands with valid tile sets: {valid_hands}")
        print(f"Hands without valid tile sets: {len(self.hand_list) - valid_hands}")
        print(f"Total tiles used: {total_tiles}")

        print("\nFamily breakdown:")
        family_counts: Dict[str, Dict[str, int]] = {}
        for hand in self.hand_list:
            stats = family_counts.setdefault(
                hand.family, {"total_hands": 0, "valid_hands": 0, "invalid_hands": 0}
            )
            stats["total_hands"] += 1
            if hand.tile_sets:
                stats["valid_hands"] += 1
            else:
                stats["invalid_hands"] += 1
        for family, stats in family_counts.items():
            print(f"  {family}: {stats['valid_hands']}/{stats['total_hands']} valid")

        return json_hands

    def print_hands_detailed(self) -> None:
        """Print detailed information about all hands."""
        print(f"{self.get_year()} Mahjong Card - All Hands")
        print("=" * 60)

        for hand in self.hand_list:
            print(f"\n{'=' * 60}")
            print(f"Hand #{hand.id + 1}: {hand.text}")
            print(f"Family: {hand.family}")
            print(f"Note: {hand.note}")
            print(f"Points: {hand.points}")
            print(f"Concealed: {'Yes' if hand.concealed else 'No'}")
            print(f"Valid tile combinations: {len(hand.tile_sets)}")

            if hand.tile_sets:
                print("\nTile Sets:")
                for i, tile_set in enumerate(hand.tile_sets):
                    print(f"  Set {i + 1}: {tile_set}")
                    readable = [self.tile_id_to_name(tile_id) for tile_id in tile_set]
                    print(f"         {readable}")
            else:
                print("  No valid tile combinations found")

        print(f"\n{'=' * 60}")
        print("SUMMARY:")
        print(f"Total hands: {len(self.hand_list)}")
        valid_hands = sum(1 for hand in self.hand_list if hand.tile_sets)
        print(f"Hands with valid tile combinations: {valid_hands}")
        print(f"Hands without valid combinations: {len(self.hand_list) - valid_hands}")
        total_tiles = sum(len(hand.tile_sets) * 14 for hand in self.hand_list)
        print(f"Total tiles used: {total_tiles}")


