from collections import Counter

from card import Card

class Player:
    """
    Player class
    """

    def __init__(self, player_id, partner_id):
        self.id = player_id
        self.partner_id = partner_id
        self.score = 0
        self.pickup_threshold = 17
        self.cards = []
        
    def __str__(self):
        return str(self.id)

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

        '''
        Check if dealer
        If so, replace any single card suits
        If none, replace lowest non trump card
        '''
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

        print(self.hand_value)

    def get_replacement_card(self, pickup_card):
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

    def player_on_left(self, num_players=4):
        return (self.id + 1) % num_players

    
