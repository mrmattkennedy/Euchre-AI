import os
import sys
import pdb
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath(__file__)))))
from players.player import Player
from card import Card

class Human(Player):
    def __init__(self, player_id, partner_id):
        super().__init__(player_id, partner_id, AI=False) 

    def play(self):
        pass
