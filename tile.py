#!/usr/bin/env python3
"""
tile.py - Tile representation and utilities for Mahjong

This file contains classes and utilities for representing Mahjong tiles.
"""

from typing import List, Dict, Any, Optional


class Tile:
    """Represents a single Mahjong tile."""
    pass


# Wind tile mappings
WIND_MAPPINGS = {
    'N': [31, 31, 31],  # North - triplet
    'S': [32, 32, 32],  # South - triplet
    'W': [33, 33, 33],  # West - triplet
    'E': [34, 34, 34]   # East - triplet
}

# Number tile mappings (0-9 across 3 suits)
NUMBER_MAPPINGS = {
    0: [10, 10, 10],   # 0 (soap tile) across all 3 suits
    1: [1, 11, 21],    # 1 across all 3 suits
    2: [2, 12, 22],    # 2 across all 3 suits
    3: [3, 13, 23],    # 3 across all 3 suits
    4: [4, 14, 24],    # 4 across all 3 suits
    5: [5, 15, 25],    # 5 across all 3 suits
    6: [6, 16, 26],    # 6 across all 3 suits
    7: [7, 17, 27],    # 7 across all 3 suits
    8: [8, 18, 28],    # 8 across all 3 suits
    9: [9, 19, 29]     # 9 across all 3 suits
}

# Dragon tile mappings
DRAGON_MAPPINGS = {
    'D': [10, 20, 30],     # Dragons across all 3 types
    '0': [10, 10, 10],     # White dragon (soap tile) - triplet
    'Wh': [10, 10, 10],    # White dragon - triplet
    'R': [30, 30, 30],     # Red dragon - triplet
    'G': [20, 20, 20]      # Green dragon - triplet
}

# Special tile mappings
SPECIAL_MAPPINGS = {
    'F': [35],          # Flower tile
    'J': [36]           # Joker tile
}


if __name__ == "__main__":
    # Example usage
    print("Tile system initialized")
