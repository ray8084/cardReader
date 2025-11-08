#!/usr/bin/env python3
"""generate2014.py - Generated hand builder for 2014 card."""

from generate import CardGeneratorBase


class Card2014(CardGeneratorBase):
    """Card builder for 2014 with all hand configurations."""

    def __init__(self):
        super().__init__(2014)
    
    def _build_all_hands(self):
        """Build all hands by calling section-specific methods."""
        self.add2014()
        self.add2468()
        self.addLikeNumbers()
        self.addAdditionHands()
        self.addQuints()
        self.addConsecutiveRun()
        self.add13579()
        self.addWindsDragons()
        self.add369()
        self.addSinglesAndPairs()

        # Print summary
        total_tiles = 0
        for hand in self.hand_list:
            total_tiles += len(hand.tile_sets) * 14  # Assuming 14 tiles per hand
            print(f"{hand.id+1} count:{len(hand.tile_sets)} - {hand}")
            # Dump tile sets for hands that have them
            if len(hand.tile_sets) > 0:
                for i, tile_set in enumerate(hand.tile_sets):
                    print(f"  Set {i+1}: {tile_set}")
        print(f"Total tiles: {total_tiles}")
    
    def add2014(self):
        p0 = self.add_hand(0, "NNNN EW SSSS 2014", "0000 00 0000 0000", "1111 00 1111 0000", "Any 1 Suit", "2014", False, 25)
        p0.addTileSets()
        
        p1 = self.add_hand(1, "222 000 1111 4444", "ggg ggg rrrr rrrr", "111 111 1111 1111", "Any 2 Suits", "2014", False, 25)
        p1.addTileSets()
    
        p2 = self.add_hand(2, "FFFF 2222 0000 14", "0000 0000 0000 00", "1111 1111 1111 00", "Any 1 Suit", "2014", False, 30)
        p2.addTileSets()
        
        p3 = self.add_hand(3, "FF 2014 1111 4444", "00 gggg rrrr 0000", "00 0000 1111 1111", "Any 3 Suits", "2014", False, 25)
        p3.addTileSets()
        
        p4 = self.add_hand(4, "FFFF DDD 2014 DDD", "0000 ggg rrrr 000", "1111 111 0000 111", "Any 2 Dragons, 2 or 3 Suits", "2014", True, 35)
        p4.addTileSets()

    def add2468(self):
        p0 = self.add_hand(0, "22 44 666 888 DDDD", "00 00 000 000 0000", "00 00 111 111 1111", "", "2468", False, 25)
        p0.addTileSets()

        p1 = self.add_hand(1, "2222 44 6666 88 88", "gggg gg gggg rr 00", "1111 00 1111 00 00", "Any 3 Suits, Pairs 8s Other 2 Suits", "2468", False, 30)
        p1.addTileSets()

        p2 = self.add_hand(2, "22 44 444 666 8888", "gg gg rrr rrr 0000", "00 00 111 111 1111", "Any 3 Suits, Kong 8s", "2468", False, 25)
        p2.addTileSets()

        p3 = self.add_hand(3, "222 444 6666 8888", "ggg ggg rrrr rrrr", "111 111 1111 1111", "Any 2 Suits", "2468", False, 25)
        p3.addTileSets()

        p4 = self.add_hand(4, "222 888 DDDD DDDD", "ggg ggg rrrr 0000", "111 111 1111 1111", "Any 3 Suits", "2468", False, 25)
        p4.addTileSets()

        p5 = self.add_hand(5, "FF 222 444 666 888", "00 000 000 000 000", "00 111 111 111 111", "", "2468", True, 30)
        p5.addTileSets()

    def addLikeNumbers(self):
        p0 = self.add_hand(0, "FFFF 1111 11 1111", "0000 gggg rr 0000", "1111 1111 00 1111", "Any Like No.", "Like Numbers", False, 25)
        p0.addTileSets_LikeNumbers()
        

    def addAdditionHands(self):
        p0 = self.add_hand(0, "FFFF 4444 + 7777 = 11", "0000 0000 0000 00", "1111 1111 1111 00", "Any 1 Suit", "Addition Hands", False, 25)
        p0.addTileSets()
        
        p1 = self.add_hand(1, "FFFF 4444 + 7777 = 11", "0000 gggg rrrr 00", "1111 1111 1111 00", "Any 3 Suits", "Addition Hands", False, 25)
        p1.addTileSets()
        
        p2 = self.add_hand(2, "FFFF 5555 + 7777 = 12", "0000 0000 0000 00", "1111 1111 1111 00", "Any 1 Suit", "Addition Hands", False, 25)
        p2.addTileSets()
        
        p3 = self.add_hand(3, "FFFF 5555 + 7777 = 12", "0000 gggg rrrr 00", "1111 1111 1111 00", "Any 3 Suits", "Addition Hands", False, 25)
        p3.addTileSets()
        
        p4 = self.add_hand(4, "FFFF 6666 + 7777 = 13", "0000 0000 0000 00", "1111 1111 1111 00", "Any 1 Suit", "Addition Hands", False, 25)
        p4.addTileSets()
        
        p5 = self.add_hand(5, "FFFF 6666 + 7777 = 13", "0000 gggg rrrr 00", "1111 1111 1111 00", "Any 3 Suits", "Addition Hands", False, 25)
        p5.addTileSets()
        

    def addQuints(self):
        p0 = self.add_hand(0, "22 333 4444 55555", "00 000 0000 00000", "00 111 1111 11111", "These Nos. Only", "Quints", False, 35)
        p0.addTileSets()
        
        p1 = self.add_hand(1, "11111 2222 33333", "ggggg rrrr 00000", "11111 1111 11111", "Any 3 Suits, Any 3 Consec. Nos. Kong Middle No. Only", "Quints", False, 40)
        p1.addTileSets_Run()
        
        p2 = self.add_hand(2, "FFFF NNNNNN 11111", "0000 000000 rrrrr", "1111 111111 11111", "Quint Any Wind & Any No. in Any Suit (INVALID: 15 tiles)", "Quints", False, 40)
        p2.addTileSets_AnyWindAnyNumber()
        
        p3 = self.add_hand(3, "11111 DDDD 11111", "ggggg rrrr 00000", "11111 1111 11111", "Quint Any Like No., Kong Dragon 3rd Suit", "Quints", False, 45)
        p3.addTileSets_LikeNumbers()

    def addConsecutiveRun(self):
        p0 = self.add_hand(0, "11 22 333 444 5555", "00 00 000 000 0000", "00 00 111 111 1111", "", "Consecutive Run", False, 25)
        p0.addTileSets()
        
        p1 = self.add_hand(1, "55 66 777 888 9999", "00 00 000 000 0000", "00 00 111 111 1111", "", "Consecutive Run", False, 25)
        p1.addTileSets()
        
        p2 = self.add_hand(2, "111 2222 333 4444", "ggg gggg rrr rrrr", "111 1111 111 1111", "Any 2 Suits, Any 4 Consec. Nos.", "Consecutive Run", False, 25)
        p2.addTileSets_Run()
        
        p3 = self.add_hand(3, "1111 22 22 22 3333", "gggg rr gg 00 gggg", "1111 00 00 00 1111", "Any 3 Consec. Nos. Like Pairs Middle No. Only", "Consecutive Run", False, 30)
        p3.addTileSets_Run()
        
        p4 = self.add_hand(4, "11 22 33 4444 4444", "gg gg gg rrrr 0000", "00 00 00 1111 1111", "Any 3 Suits, Any 3 Consec. Prs., Like Kongs Ascending No", "Consecutive Run", False, 30)
        p4.addTileSets_Run()

        p5 = self.add_hand(5, "FFFF 1111 2222 DD", "0000 0000 0000 00", "1111 1111 1111 00", "Any 2 Consec. Nos.", "Consecutive Run", False, 25)
        p5.addTileSets_Run()

        p6 = self.add_hand(6, "11 22 111 222 3333", "gg gg rrr rrr 0000", "00 00 111 111 1111", "Any 3 Suits, Any 3 Consec. Nos.", "Consecutive Run", False, 25)
        p6.addTileSets_Run()

        p7 = self.add_hand(7, "111 22 333 DDD DDD", "ggg gg ggg rrr 000", "111 00 111 111 111", "Any 3 Suits, Any 3 Consec. Nos.", "Consecutive Run", True, 30)
        p7.addTileSets_Run()

    def add13579(self):
        p0 = self.add_hand(0, "11 33 555 777 9999", "00 00 000 000 0000", "00 00 111 111 1111", "", "13579", False, 25)
        p0.addTileSets()

        p1 = self.add_hand(1, "111 3333 333 5555", "ggg gggg rrr rrrr", "111 1111 111 1111", "Any 2 Suits", "13579", False, 25)
        p1.addTileSets()
        
        p2 = self.add_hand(2, "555 7777 777 9999", "ggg gggg rrr rrrr", "111 1111 111 1111", "Any 2 Suits", "13579", False, 25)
        p2.addTileSets()
        
        p3 = self.add_hand(3, "FFFF 1111 33 5555", "0000 0000 00 0000", "1111 1111 00 1111", "", "13579", False, 25)
        p3.addTileSets()
        
        p4 = self.add_hand(4, "FFFF 5555 77 9999", "0000 0000 00 0000", "1111 1111 00 1111", "", "13579", False, 25)
        p4.addTileSets()
        
        p5 = self.add_hand(5, "11 33 111 333 5555", "gg gg rrr rrr 0000", "00 00 111 111 1111", "Any 3 Suits", "13579", False, 25)
        p5.addTileSets()
        
        p6 = self.add_hand(6, "55 77 555 777 9999", "gg gg rrr rrr 0000", "00 00 111 111 1111", "Any 3 Suits", "13579", False, 25)
        p6.addTileSets()
        
        p7 = self.add_hand(7, "FF 1111 9999 DDDDD", "00 0000 0000 00000", "00 1111 1111 11111", "Any 1 Suit", "13579", False, 25)
        p7.addTileSets()
        
        p8 = self.add_hand(8, "FF 1111 9999 DDDDD", "00 gggg rrrr 00000", "00 1111 1111 11111", "Any 3 Suits", "13579", False, 25)
        p8.addTileSets()
        
        p9 = self.add_hand(9, "111 3 555 555 7 999", "ggg g ggg rrr r rrr", "111 0 111 111 0 111", "Any 2 Suits", "13579", True, 30)
        p9.addTileSets()
        

    def addWindsDragons(self):
        p0 = self.add_hand(0, "NNNN EEEE WWWW SS", "0000 0000 0000 00", "1111 1111 1111 00", "", "Winds - Dragons", False, 25)
        p0.addTileSets()

        p1 = self.add_hand(1, "FFFF NNNN RR SSSS", "0000 0000 rr 0000", "1111 1111 00 1111", "Red Dragon Only", "Winds - Dragons", False, 25)
        p1.addTileSets()

        p2 = self.add_hand(2, "FFFF EEEE GG WWWWW", "0000 0000 gg 00000", "1111 1111 00 11111", "Green Dragon Only (INVALID: 15 tiles)", "Winds - Dragons", False, 25)
        p2.addTileSets()

        p3 = self.add_hand(3, "NN 11 SSS 111 1111", "00 gg 000 rrr 0000", "00 00 111 111 1111", "Any Like Odd No.", "Winds - Dragons", True, 30)
        p3.addTileSets_LikeNumbersOdd()
        
        p4 = self.add_hand(4, "EE 22 WWW 222 2222", "00 gg 000 rrr 0000", "00 00 111 111 1111", "Any Like Even No.", "Winds - Dragons", True, 30)
        p4.addTileSets_LikeNumbersEven()
        
        p5 = self.add_hand(5, "FFFF DDDD DD DDDD", "0000 gggg rr 0000", "1111 1111 00 1111", "Any 3 Suits", "Winds - Dragons", False, 30)
        p5.addTileSets()

    def add369(self):
        p0 = self.add_hand(0, "FF 3333 66 9999 DD", "00 0000 00 0000 00", "00 1111 00 1111 00", "", "369", False, 30)
        p0.addTileSets()

        p1 = self.add_hand(1, "333 666 6666 9999", "ggg ggg 0000 0000", "111 111 1111 1111", "Any 2 Suits", "369", False, 25)
        p1.addTileSets()

        p2 = self.add_hand(2, "33 66 99 3333 3333", "gg gg gg rrrr 0000", "00 00 00 1111 1111", "Any 3 Suits, Like Kongs 3, 6 or 9", "369", False, 30)
        p2.addTileSets("33 66 99 3333 3333")
        p2.addTileSets("33 66 99 6666 6666")
        p2.addTileSets("33 66 99 9999 9999")

        p3 = self.add_hand(3, "FF 3333 6666 9999", "00 gggg rrrr 0000", "00 1111 1111 1111", "Any 3 Suits", "369", False, 25)
        p3.addTileSets()

        p4 = self.add_hand(4, "333 66 999 333 333", "ggg gg ggg rrr rrr", "111 00 111 111 111", "Any 2 Suits, Like Pungs 3, 6 or 9", "369", False, 30)
        p4.addTileSets("333 66 999 333 333")
        p4.addTileSets("333 66 999 666 666")
        p4.addTileSets("333 66 999 999 999")

        p5 = self.add_hand(5, "333 6 999 333 6 999", "ggg g ggg rrr r rrr", "111 0 111 111 0 111", "Any 2 Suits", "369", True, 30)
        p5.addTileSets()

    def addSinglesAndPairs(self):
        p0 = self.add_hand(0, "NN EE WW SS 11 11 11", "00 00 00 00 gg rr 00", "00 00 00 00 00 00 00", "Any Like No.", "Singles and Pairs", True, 50)
        p0.addTileSets_LikeNumbers()

        p1 = self.add_hand(1, "FF 11 22 33 44 55 DD", "00 00 00 00 00 00 00", "00 00 00 00 00 00 00", "Any 5 Consec. Nos.", "Singles and Pairs", True, 50)
        p1.addTileSets_Run()

        p2 = self.add_hand(2, "11 33 55 77 99 11 11", "gg gg gg gg gg rr 00", "00 00 00 00 00 00 00", "Any Like Odd No. in Other 2 Suits", "Singles and Pairs", True, 50)
        p2.addTileSets("11 33 55 77 99 11 11")
        p2.addTileSets("11 33 55 77 99 33 33")
        p2.addTileSets("11 33 55 77 99 55 55")
        p2.addTileSets("11 33 55 77 99 77 77")
        p2.addTileSets("11 33 55 77 99 99 99")

        p3 = self.add_hand(3, "FF 22 46 88 22 46 88", "00 gg gg gg rr rr rr", "00 00 00 00 00 00 00", "Any 2 Suits", "Singles and Pairs", True, 50)
        p3.addTileSets()
        
        p4 = self.add_hand(4, "FF 11 22 11 22 11 22", "00 gg gg rr rr 00 00", "00 00 00 00 00 00 00", "Any 2 Consec. Nos. in 3 Suits", "Singles and Pairs", True, 50)
        p4.addTileSets_Run()

        p5 = self.add_hand(5, "336 33669 336699", "ggg rrrrr 000000", "000 00000 000000", "Any 3 Suits", "Singles and Pairs", True, 50)
        p5.addTileSets()
        
        p6 = self.add_hand(6, "FF 2014 2014 2014", "00 gggg rrrr 0000", "00 0000 0000 0000", "3 Suits", "Singles and Pairs", True, 75)
        p6.addTileSets()


if __name__ == "__main__":
    card = Card2014()
    print(f"Generated {len(card.hand_list)} hands for {card.get_year()}")
    
    # Example usage of new methods:
    # card.print_hands_detailed()  # Print detailed hand information
    card.export_to_json()       # Export to JSON file
