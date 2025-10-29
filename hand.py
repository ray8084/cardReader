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
    
    def __repr__(self) -> str:
        """
        Detailed string representation for debugging.
        
        Returns:
            Detailed string with all hand properties
        """
        return (f"Hand(id={self.id}, text='{self.text}', family='{self.family}', "
                f"concealed={self.concealed}, points={self.points}, "
                f"tile_sets={len(self.tile_sets)})")
