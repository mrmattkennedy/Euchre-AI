import os
import sys
import random
from pathlib import Path

from player import Player

#Append to use player class in parent directory
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
from card import Card

class AI_dumb(Player):
    """
    Player class
    """

    def __init__(self, player_id, partner_id):
        super().__init__(player_id, partner_id)

    def play(self, trump, lead_card=None):
        #If no lead card, just pick one
        if not lead_card:
            card = random.choice(self.cards)

        #If there is a lead, pick a legal card
        else:
            #Find cards of same suit
            lead_suit = lead_card.suit
            if lead_card.is_left(trump):
                lead_suit = trump
                
            viable = []
            for c in self.cards:
                if c.suit == lead_suit or c.is_left(trump):
                    viable.append(c)

            #If a viable card, choose randomly. Otherwise, choose randomly from all
            if viable:
                card = random.choice(viable)
            else:
                card = random.choice(self.cards)

        self.cards.remove(card)
        return card
