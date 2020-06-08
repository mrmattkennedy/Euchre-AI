class Card:
    """
    Card class
    Has a suit and a value
    Also has a per-trick value, depending on lead card and trump
    """

    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return "{}{}".format(self.value, self.suit)


    def trick_value(self, trick_value):
        self.trick_value = trick_value
