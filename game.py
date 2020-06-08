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
        suits = ["H", "D", "S", "C"]
        card_nums = [i for i in range(9, 15)]
        
        self.cards = []
        for s in suits:
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
        Pickup round 1 - see if anyone wants dealer to pick up
        Pickup round 2 - see if anyone wants to call
        Pass - move on
        If called, play
        """

        self.deal_cards()


    def deal_cards(self):
        """
        Dealer hands out cards in order of 2,3,2,3,3,2,3,2
        """

        #Get deal order based on first player to the left of the dealer
        deal_order = []
        curr_player = self.dealer
        for i in range(self.num_players):
            deal_order.append(self.players[curr_player].player_on_left())
            curr_player = (curr_player + 1) % self.num_players

        #Deal out cards
        cards_available = self.cards.copy()
        rnd_1 = [2, 3, 2, 3]
        rnd_2 = [3, 2, 3, 2]

        #First round
        for num, i in enumerate(deal_order):
            for _ in range(rnd_1[num]):
                card = random.choice(cards_available)
                cards_available.remove(card)
                self.players[i].add_card(card)
                
        #Second round 
        for num, i in enumerate(deal_order):
            for _ in range(rnd_2[num]):
                card = random.choice(cards_available)
                cards_available.remove(card)
                self.players[i].add_card(card)

        #Assign pickup card to choose
        self.pickup_card = cards_available[0]
        
g = Game()
