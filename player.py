from card import Card

class Player:
    """
    Player class
    """

    def __init__(self, player_id, partner_id):
        self.id = player_id
        self.partner_id = partner_id
        self.score = 0
        self.cards = []
        
    def __str__(self):
        return str(self.id)


    def add_card(self, card):
        self.cards.append(card)

    def player_on_left(self, num_players=4):
        return (self.id + 1) % num_players

    
