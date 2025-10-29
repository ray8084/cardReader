#!/usr/bin/env python3
"""
tile.py - Tile representation and utilities for Mahjong

This file contains classes and utilities for representing Mahjong tiles.
"""

from typing import List, Dict, Any, Optional


class Tile:
    """Represents a single Mahjong tile."""
    pass


# All tile mappings in one structure
TILE_MAPPINGS = {
    # Wind tiles
    'N': [31, 31, 31],  # North - triplet
    'S': [32, 32, 32],  # South - triplet
    'W': [33, 33, 33],  # West - triplet
    'E': [34, 34, 34],  # East - triplet
    
    # Number tiles (0-9 across 3 suits)
    '0': [10, 10, 10],    # 0 (soap tile) across all 3 suits
    '1': [1, 11, 21],     # 1 across all 3 suits
    '2': [2, 12, 22],     # 2 across all 3 suits
    '3': [3, 13, 23],     # 3 across all 3 suits
    '4': [4, 14, 24],     # 4 across all 3 suits
    '5': [5, 15, 25],     # 5 across all 3 suits
    '6': [6, 16, 26],     # 6 across all 3 suits
    '7': [7, 17, 27],     # 7 across all 3 suits
    '8': [8, 18, 28],     # 8 across all 3 suits
    '9': [9, 19, 29],     # 9 across all 3 suits
    
    # Dragon tiles
    'D': [10, 20, 30],     # Dragons across all 3 types
    'Wh': [10, 10, 10],    # White dragon - triplet
    'R': [30, 30, 30],     # Red dragon - triplet
    'G': [20, 20, 20],     # Green dragon - triplet
    
    # Special tiles
    'F': [35, 35, 35],  # Flower tile - triplet
    'J': [36, 36, 36]   # Joker tile - triplet
}


if __name__ == "__main__":
    # Example usage
    print("Tile system initialized")
