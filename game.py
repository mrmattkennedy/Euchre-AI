import random

from card import Card
from player import Player

class Game:
    """
    Game class
    """

    def __init__(self):
        self.create_cards()
        self.create_players()
        self.get_first_dealer()
        self.play()

    def create_cards(self):
        """
        There are 4 suits and 6 cards
        4 suits: (H)earts, (D)iamonds, (S)pades, (C)lubs
        6 cards: 9, 10, 11, 12, 13, 14 (9, 10, Jack, Queen, King, Ace)

        Create cards and put them in a list
        """

        self.num_cards = 24
        self.suits = ["H", "D", "S", "C"]
        card_nums = [i for i in range(9, 15)]
        
        self.cards = []
        for s in self.suits:
            for n in card_nums:
                self.cards.append(Card(s, n))

    def create_players(self):
        """
        4 players, each with a partner
        Player 0 is partnered with player 2
        Player 1 is partnered with player 3
        """

        self.num_players = 4
        self.players = [Player(i, (i+2) % self.num_players) for i in range(0, self.num_players)]


    def get_first_dealer(self):
        """
        Hands out cards from deck in order until black jack received
        """
        cards_available = self.cards.copy()
        curr_player = 1

        #Get dealer
        while cards_available:
            card = random.choice(cards_available)
            cards_available.remove(card)
            
            if (card.suit == "S" or card.suit == "C") and card.value == 11:
                break
            curr_player = (curr_player+1) % self.num_players

        #Assign player as dealer
        self.dealer = curr_player



    def play(self):
        """
        Plays the game
        Starts by dealing cards out
        Assigns score values to each player, based on the face up card
        Pickup round 1 - see if anyone wants dealer to pick up
        Pickup round 2 - see if anyone wants to call
        Pass - move on
        If called, play
        """

        self.deal_cards()
        called = self.pickup()
        if called:
            pass

    def deal_cards(self):
        """
        Dealer hands out cards in order of 2,3,2,3,3,2,3,2
        """

        #Get deal order based on first player to the left of the dealer
        self.get_deal_order()

        #Deal out cards
        cards_available = self.cards.copy()
        rnd_1 = [2, 3, 2, 3]
        rnd_2 = [3, 2, 3, 2]

        #First round
        for num, i in enumerate(self.order):
            for _ in range(rnd_1[num]):
                card = random.choice(cards_available)
                cards_available.remove(card)
                self.players[i].add_card(card)
                
        #Second round 
        for num, i in enumerate(self.order):
            for _ in range(rnd_2[num]):
                card = random.choice(cards_available)
                cards_available.remove(card)
                self.players[i].add_card(card)

        #Assign pickup card to choose
        self.pickup_card = cards_available[0]        
        
    def pickup(self):
        """
        Assigns total point values based on pickup card
        Sees if any players want to pickup the presented card
        If no pickup, go around and check if anyone wants to call for any suit
        """
        #print("Trump: {}\tDealer: {}".format(self.pickup_card, self.dealer))

        #Round 1 - for each player, get value, see if want to call
        for p in self.order:
            self.players[p].get_hand_value(self.pickup_card.suit, self.dealer==self.players[p].id, self.pickup_card)

            #If called, assign caller and have dealer pickup
            if self.players[p].call_pickup():
                self.trump = self.pickup_card.suit
                self.caller = self.players[p].id
                self.players[self.dealer].pickup(self.pickup_card)
                return True

        #Round 2 - go in order and check value for every possible trump
        #Remove pickup card suit as available call
        available_suits = self.suits.copy()
        available_suits.remove(self.pickup_card.suit)
        
        for p in self.order:
            for s in available_suits:
                print(p, s)
                self.players[p].get_hand_value(s)
                if self.players[p].call_pickup():
                    self.caller = self.players[p].id
                    self.trump = s
                    return True

        return False
        
    def get_deal_order(self):
        """
        Gets the order for dealing out cards and calling pickups
        at the start of each round, starting with first to left of dealer
        """
        self.order = []
        for i in range(self.num_players):
            self.order.append((i+self.dealer+1) % self.num_players)

    
g = Game()
