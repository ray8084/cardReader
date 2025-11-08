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

    def export_to_json(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Export the card data to a JSON file with condensed tile sets."""
        target_file = filename or f"card{self.year}.json"

        json_data: Dict[str, Any] = {
            "year": self.get_year(),
            "total_hands": len(self.hand_list),
            "hands": [],
        }

        for hand in self.hand_list:
            hand_data: Dict[str, Any] = {
                "id": hand.id,
                "hand": hand.text,
                "colorMask": hand.mask,
                "jokerMask": hand.joker_mask,
                "note": hand.note,
                "family": hand.family,
                "points": str(hand.points),
                "concealed": hand.concealed,
                "tile_sets": [],
            }

            print(f"DEBUG: Hand {hand.id} ({hand.text}) has {len(hand.tile_sets)} tile sets")
            for i, tile_set in enumerate(hand.tile_sets):
                print(f"DEBUG: Adding tile set {i}: {tile_set}")
                hand_data["tile_sets"].append(tile_set)

            json_data["hands"].append(hand_data)

        valid_hands = sum(1 for hand in self.hand_list if hand.tile_sets)
        total_tiles = sum(len(hand.tile_sets) * 14 for hand in self.hand_list)

        json_data["summary"] = {
            "hands_with_valid_tiles": valid_hands,
            "hands_without_valid_tiles": len(self.hand_list) - valid_hands,
            "total_tiles_used": total_tiles,
            "families": {},
        }

        for hand in self.hand_list:
            family = hand.family
            family_stats = json_data["summary"]["families"].setdefault(
                family,
                {"total_hands": 0, "valid_hands": 0, "invalid_hands": 0},
            )
            family_stats["total_hands"] += 1
            if hand.tile_sets:
                family_stats["valid_hands"] += 1
            else:
                family_stats["invalid_hands"] += 1

        with open(target_file, "w", encoding="utf-8") as fh:
            json_str = json.dumps(json_data, indent=2, ensure_ascii=False, separators=(",", ": "))

            lines = json_str.split("\n")
            result_lines: List[str] = []
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                if stripped.startswith("[") and not stripped.startswith("[["):
                    tile_lines: List[str] = []
                    i += 1
                    while i < len(lines) and "]" not in lines[i]:
                        tile_lines.append(lines[i].strip().rstrip(","))
                        i += 1
                    if i < len(lines):
                        i += 1  # skip closing bracket

                    tile_ids_str = ", ".join(tile_lines)
                    is_last = True
                    j = i
                    while j < len(lines):
                        lookahead = lines[j].strip()
                        if lookahead.startswith("[") and not lookahead.startswith("[["):
                            is_last = False
                            break
                        if lookahead == "]":
                            break
                        j += 1

                    if is_last:
                        result_lines.append(f"          [{tile_ids_str}]")
                    else:
                        result_lines.append(f"          [{tile_ids_str}],")
                else:
                    result_lines.append(line)
                    i += 1

            fh.write("\n".join(result_lines))

        print(f"JSON file '{target_file}' exported successfully!")
        print(f"Total hands: {json_data['total_hands']}")
        print(f"Hands with valid tile sets: {json_data['summary']['hands_with_valid_tiles']}")
        print(f"Hands without valid tile sets: {json_data['summary']['hands_without_valid_tiles']}")
        print(f"Total tiles used: {json_data['summary']['total_tiles_used']}")

        print("\nFamily breakdown:")
        for family, stats in json_data["summary"]["families"].items():
            print(f"  {family}: {stats['valid_hands']}/{stats['total_hands']} valid")

        return json_data

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


