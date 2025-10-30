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
    
    def addTileSets(self, hand_text=None):
        """
        Generate tile sets with mixed suits for this hand.
        
        Uses the mask to determine which tiles get which suit combinations:
        - 'r' (red): outer loop determines suit
        - 'g' (green): middle loop determines suit  
        - '0' (black): inner loop determines suit
        
        Args:
            hand_text (str, optional): Custom hand text to use instead of self.text
        """
        from tile import TILE_MAPPINGS
        
        # Use provided hand text or default to self.text
        text_to_use = hand_text if hand_text is not None else self.text
        
        # Parse the hand text into individual characters (tiles)
        tiles = []
        for group in text_to_use.split():
            if group not in ['+', '=']:  # Skip special characters
                tiles.extend(list(group))  # Add each individual tile
        
        # Parse the mask into individual characters
        mask_chars = []
        for group in self.mask.split():
            mask_chars.extend(list(group))  # Add each mask character
        
        # Generate tile sets with mixed suits
        # Outer loop: determines suit for red ('r') tiles
        for red_suit in range(3):
            # Middle loop: determines suit for green ('g') tiles
            for green_suit in range(3):
                # Inner loop: determines suit for black ('0') tiles
                for black_suit in range(3):
                    # Skip combinations where any two suits are the same
                    if black_suit == green_suit or black_suit == red_suit or green_suit == red_suit:
                        continue
                        
                    tile_set = []
                    
                    # Process each individual tile
                    for i, tile in enumerate(tiles):
                        if tile in TILE_MAPPINGS:
                            tile_ids = TILE_MAPPINGS[tile]
                            
                            # Determine which suit to use based on mask
                            if i < len(mask_chars):
                                mask_char = mask_chars[i]
                                if mask_char == '0':  # Black - use inner loop suit
                                    tile_id = tile_ids[black_suit]
                                elif mask_char == 'g':  # Green - use middle loop suit
                                    tile_id = tile_ids[green_suit]
                                elif mask_char == 'r':  # Red - use outer loop suit
                                    tile_id = tile_ids[red_suit]
                                else:  # Default to first suit
                                    tile_id = tile_ids[0]
                            else:
                                # Default to first suit if no mask
                                tile_id = tile_ids[0]
                            
                            tile_set.append(tile_id)
                    
                    # Add the tile set to this hand
                    self.add_tile_set(tile_set)
        
        # Post-processing: remove duplicates
        self.removeDuplicates()
    
    def removeDuplicates(self):
        """
        Remove duplicate tile sets from this hand.
        Tile sets are considered duplicates if they contain the same tiles regardless of order.
        Also sorts each tile set so single digits come before teens.
        """
        unique_tile_sets = []
        for tile_set in self.tile_sets:
            # Sort the tile set so single digits (1-9) come before teens (10-19)
            sorted_tile_set = sorted(tile_set)
            is_duplicate = False
            for existing_set in unique_tile_sets:
                if sorted(existing_set) == sorted_tile_set:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tile_sets.append(sorted_tile_set)
        
        self.tile_sets = unique_tile_sets
    
    def addTileSets_LikeNumbers(self, hand_text=None):
        """
        Generate tile sets for Like Numbers hands.
        
        This method creates tile sets where all tiles have the same number
        across different suits (like 1-1-1, 2-2-2, etc.)
        
        Args:
            hand_text (str, optional): Custom hand text to use instead of self.text
        """
        # Use provided hand text or default to self.text
        text_to_use = hand_text if hand_text is not None else self.text
        
        # Run for numbers 1 through 9
        for number in range(1, 10):
            # Replace the number in the hand text
            modified_text = text_to_use.replace('1', str(number))
            
            # Call the main addTileSets method with the modified hand text
            self.addTileSets(modified_text)
