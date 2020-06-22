import os
import sys
import random
from pathlib import Path
from collections import Counter
from abc import ABCMeta, abstractmethod

#Append to use player class in parent directory
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
from card import Card

class Player(metaclass=ABCMeta):
    """
    Player class
    """
    #Assign from game class
    cards = []
    
    def __init__(self, player_id, partner_id):
        self.id = player_id
        self.partner_id = partner_id
        
        self.score = 0
        self.tricks = 0
        self.pickup_threshold = 17
        self.cards = []
        
    def __str__(self):
        return str(self.id)

    def reset(self):
        self.cards.clear()
        self.score = 0
        self.tricks = 0

    def clear_hand(self):
        self.cards.clear()

    def reset_tricks(self):
        self.tricks = 0
        
    def get_hand_value(self, trump, dealer=None, pickup_card=None):
        """
        Used to check value of hand at the beginning of each round
        Checks if dealer
            If dealer, replace a card with pickup card
                Replacement card is either lowest single card suit, or just lowest card

        Next, check value of hand. Add up scores for cards, then divide by a value based on # of suits (more suits is bad)
        """
        self.hand_value = 0
        suits = []

        if dealer:
            replacement_card = self.get_replacement_card(pickup_card)
            #Replace temporarily as lowest card
            self.cards[self.cards.index(replacement_card)] = pickup_card
            
        #Get value in hand       
        for c in self.cards:
            #Determine number of suits in hand
            if c.suit not in suits and not c.is_left(trump):
                suits.append(c.suit)
            elif c.is_left(trump) and trump not in suits:
                suits.append(trump)

            #Get card value if trump
            if c.suit == trump:
                if c.value == 11:
                    self.hand_value += 16
                else:
                    self.hand_value += c.value
                    
            #Get card value if not trump
            else:
                if c.is_left(trump):
                    self.hand_value += 15
                else:
                    self.hand_value += c.get_offsuit_value()

        #Divide by # of suits
        self.hand_value /= 1.5**len(suits)

        if dealer:
            self.cards[self.cards.index(pickup_card)] = replacement_card


    def get_replacement_card(self, pickup_card):
        """
        Check if dealer
        If so, replace any single card suits
        If none, replace lowest non trump card
        """
        trump = pickup_card.suit
        
        #Count cards in each suit
        suit_count = Counter()
        for c in self.cards:
            if c.suit == trump:
                continue
            suit_count[c.suit] += 1

        #See if any single card suits
        if any(value == 1 for value in suit_count.values()):
            lowest_card = None
            for c in self.cards:
                if suit_count[c.suit] == 1:
                    #If not the left, not an offsuit ace, and lower than already picked, replace as lowest
                    if not (c.is_left(trump)) and (lowest_card and c.value < lowest_card.value and c.value < 14) or (not lowest_card):
                        lowest_card = c
            

        #No single suits, just get the lowest at this point
        else:
            lowest_card = None
            for c in self.cards:
                #If not the left, not an offsuit ace, and lower than already picked, replace as lowest
                if not (c.is_left(trump)) and (lowest_card and c.value < lowest_card.value) or (not lowest_card):
                    lowest_card = c

        #Replace temporarily as lowest card
        return lowest_card

    def add_card(self, card):
        self.cards.append(card)
        
    def call_pickup(self):
        return self.hand_value >= self.pickup_threshold

    def pickup(self, pickup_card):
        replacement_card = self.get_replacement_card(pickup_card)
        self.cards[self.cards.index(replacement_card)] = pickup_card
        return replacement_card

    def legal_card(self, card, lead, trump):
        """
        if card is same suit as lead, legal
        if card is lead is trump and card is left, legal
        if card is diff suit and have no cards that follow lead, legal
        """
        if card.suit == lead:
            return True
        if lead == trump and card.is_left(trump):
            return True

        #If suit isn't lead suit, or if lead and trump are same and card isn't lead and isn't left
        elif ((card.suit != lead and lead != trump) or
              (lead == trump and (card.suit != lead and not card.is_left(trump)))):

            #If any cards in hand are legal cards, return False
            for c in self.cards:
                if c.suit == lead or (lead == trump and c.is_left(trump)):
                    return False

        return True

    @staticmethod
    def get_card(idx):
        return Player.cards[idx]

    
    @abstractmethod
    def play(self, trump, lead_card=None):
        pass


    
if __name__ == "__main__":
    suits = ["H", "D", "S", "C"]
    card_nums = [i for i in range(9, 15)]

    #Create cards list
    cards = []
    for s in suits:
        for n in card_nums:
            cards.append(Card(s, n))
    p = Player(1, 3, cards)

    #for i, c in enumerate(cards):
    #    print(i, c)
        
    for _ in range(4):
        p.add_card(random.choice(cards[5:]))
    p.add_card(cards[8])
    #p.add_card(cards[7])
    #p.add_card(cards[8])
    #p.add_card(cards[9])
    #p.add_card(cards[10])
    lead ='H'
    trump = 'H'
    for c in p.cards:
        print(c)
    print()
    for c in cards:
        print(c, p.legal_card(c, lead, trump))
