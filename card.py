class Card:
    """
    Card class
    Has a suit and a value
    Also has a per-trick value, depending on lead card and trump
    """

    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.pairs = {"H": "D", "D": "H", "S": "C", "C": "S"}
        self.offsuit_values = {9: 0, 10: 0, 11: 0, 12: 1, 13: 2, 14: 5}

    def __str__(self):
        return "{}{}".format(self.value, self.suit)

    def is_left(self, suit):
        return self.suit == self.pairs[suit] and self.value == 11

    def get_offsuit_value(self):
        return self.offsuit_values[self.value]

    def trick_value(self, trick_value):
        self.trick_value = trick_value
