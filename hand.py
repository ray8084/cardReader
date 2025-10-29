#!/usr/bin/env python3
"""
hand.py - Hand class for representing mahjong hand configurations

This module contains the Hand class that represents individual hand configurations
with their properties and methods for managing tile sets.
"""

from typing import List

class Hand:
    """Represents a single hand configuration."""
    
    def __init__(self, hand_id: int, text: str, mask: str, joker_mask: str, 
                 note: str, family: str, concealed: bool, points: int):
        """
        Initialize a hand configuration.
        
        Args:
            hand_id: Unique identifier for the hand
            text: Text representation of the hand pattern
            mask: Color mask for the hand
            joker_mask: Joker mask for the hand
            note: Additional notes about the hand
            family: Family/section name the hand belongs to
            concealed: Whether the hand is concealed
            points: Point value of the hand
        """
        self.id = hand_id
        self.text = text
        self.mask = mask
        self.joker_mask = joker_mask
        self.note = note
        self.family = family
        self.concealed = concealed
        self.points = points
        self.tile_sets = []  # Will be populated with valid tile combinations
    
    def add_tile_set(self, tiles: List[int]):
        """
        Add a valid tile combination to this hand.
        
        Args:
            tiles: List of tile IDs representing a valid combination
        """
        self.tile_sets.append(tiles)
    
    def get_family_string(self) -> str:
        """
        Get the family/section name.
        
        Returns:
            The family name as a string
        """
        return self.family
    
    def get_tile_count(self) -> int:
        """
        Get the total number of tile sets for this hand.
        
        Returns:
            Number of tile sets
        """
        return len(self.tile_sets)
    
    def __str__(self) -> str:
        """
        String representation of the hand.
        
        Returns:
            Formatted string with family, text, and note
        """
        return f"{self.family} {self.text} {self.note}"
    
    def generateTileSetStatic(self):
        """
        Generate static tile sets for this hand.
        
        This method creates 3 sets of 14 tiles by walking through the 3 suits,
        keeping winds and soap tiles the same in every pass.
        """
        from tile import TILE_MAPPINGS
        
        # Parse the hand text into individual characters (tiles)
        tiles = []
        for group in self.text.split():
            if group not in ['+', '=']:  # Skip special characters
                tiles.extend(list(group))  # Add each individual tile
        
        # Generate 3 sets of tiles (one for each suit)
        for suit_index in range(3):
            tile_set = []
            
            # Process each individual tile
            for tile in tiles:
                if tile in TILE_MAPPINGS:
                    tile_ids = TILE_MAPPINGS[tile]
                    # Use the tile ID at the current suit index
                    tile_id = tile_ids[suit_index]
                    tile_set.append(tile_id)
            
            # Add the tile set to this hand
            self.add_tile_set(tile_set)
    
    def generateTileSetsMixedSuit(self):
        """
        Generate tile sets with mixed suits for this hand.
        
        This method will create tile combinations using different suits
        for different parts of the hand pattern.
        """
        pass
