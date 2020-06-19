import pdb
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from card import Card
from player import Player
from AI.play import DQL_Play_TF1, DQL_Play_TF2

class Game:
    """
    Game class
    TODO:
    Add decision tree for pickup calls?
    """

    def __init__(self, AI=True, AI_id=0):
        self.state = {}
        self.training = True
        self.AI = AI
        self.AI_id = AI_id

        self.create_cards()
        self.create_players()
        self.state = self.reset_state()
        
        if self.AI:
            self.create_AI()
            
        self.get_first_dealer()
        self.play()
        self.tricks_won = 0

    def create_AI(self):
        self.AI_play = DQL_Play_TF1(state_size=len(self.state),
                                action_size=self.num_cards,
                                hidden_sizes=[50,50])
        self.reward = 0
        self.terminated = False
        self.current_state = self.reset_state()
        self.next_state = self.reset_state()

    def reset(self):
        self.state = self.reset_state()
        self.current_state = self.reset_state()
        self.next_state = self.reset_state()
        for p in self.players:
            p.reset()

        self.tricks_won = 0
        
    def reset_state(self):
        """
        Reset state
            Each card
                0-3 means played by that player
                4 in my hand
                5 in play,
                6 kitty
                7 not seen
            My score
            Opponent score
            Current tricks I have
            Current tricks opponent has
            Tricks remaining
            My order to play
            What trump is
            What the lead is (5 for no lead)
            Who called
            Who currently has the hand (3 for no one)

        """
        state = {}
        for c in self.cards:
            state[str(c)] = 7
        state['my_tricks'] = 0
        state['op_tricks'] = 0
        state['rem_tricks'] = 5
        state['play_pos'] = 0
        state['trump'] = 0
        state['lead'] = 4
        state['caller'] = 0
        state['have_trick'] = 2
        return state

    def create_cards(self):
        """
        There are 4 suits and 6 cards
        4 suits: (H)earts, (D)iamonds, (S)pades, (C)lubs
        6 cards: 9, 10, 11, 12, 13, 14 (9, 10, Jack, Queen, King, Ace)

        Create cards and put them in a list
        """

        self.size_hand = 5
        self.num_cards = 24
        self.suits = ["H", "D", "S", "C"]
        card_nums = [i for i in range(9, 15)]

        #Create cards list and state dictionary
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
        tricks_to_play = 5000000
        batch_size = 10
        counter = 0
        start = time.time()
        
        for i in range(tricks_to_play):
            self.reset()
            play_game = True

            if i % 1000 == 0:
                print(i)
                
            while not any(p.score >= 10 for p in self.players) and play_game:
                self.deal_cards()
                called = self.pickup()
                if called:
                    play_game = self.play_trick()
                    counter += 1
                    
                else:
                    #print('no call')
                    for p in self.players:
                        p.clear_hand()

                #update AI target network
                #self.AI_play.align_models()
                if counter % batch_size == 0 and counter != 0:
                    self.AI_play.retrain(batch_size)
                    counter += 1


        print("Won {}".format(self.tricks_won))
        total_time = time.time() - start
        print("Time: {}s, {}m".format(round(total_time, 2), round(total_time / 60, 2)))
        
        c_list = self.AI_play.get_cost_data(1000)
        plt.figure(figsize=(14,7))
        plt.plot(range(len(c_list)),c_list)
        plt.xlabel('Trainings')
        plt.ylabel('Cost')
        plt.show()

        called = False
        while not called:
            self.reset()
            self.deal_cards()
            called = self.pickup()
            if called:
                play_game = self.play_trick(True)
        
        
    def play_trick(self, verbose=False):
        #Play, starting with lead and following with others
        for _ in range(self.size_hand):
            if self.AI:
                self.state['play_pos'] = self.order.index(self.AI_id)

                #AI is lead
                if self.AI_id == self.order[0]:
                    #Update state
                    self.state['lead'] = 4
                    self.state['have_trick'] = 2
                    
                    #Let AI play
                    action, reset = self.AI_action()

                    #Illegal card, reset
                    if reset:
                        self.next_state = self.reset_state()
                        self.terminated = True
                        self.AI_play.store(self.get_state(self.state),
                                           action,
                                           self.reward,
                                           self.get_state(self.state),
                                           self.terminated)
                        if verbose:
                            print("Action {}, {} failed".format(action, self.cards[action]))
                        return False
                    
                    #Legal move, save current state
                    self.current_state = self.state.copy()

                    lead_card = self.cards[action]
                    self.players[self.AI_id].play_card(lead_card)
                    trick = {self.order[0]: lead_card}
                    
                    #Update state
                    self.state[str(lead_card)] = 5
                    self.state['lead'] = self.suits.index(lead_card.suit)

                #AI is not lead
                else:
                    #Get lead card
                    lead_card = self.players[self.order[0]].play_random(self.trump)
                    trick = {self.order[0]: lead_card}

                    #Update state
                    self.state[str(lead_card)] = 5
                    self.state['lead'] = self.suits.index(lead_card.suit)
                    
                    #Every other player plays until AI
                    for p in self.order[1:self.state['play_pos']]:
                        trick[p] = self.players[p].play_random(self.trump, lead_card=lead_card)
                        self.state[str(trick[p])] = 5

                    #Let AI play
                    current_winner = self.evaluate_trick(trick, verbose=False)
                    self.state['have_trick'] = int(current_winner == self.players[self.AI_id].partner_id)
                    action, reset = self.AI_action()

                    #Illegal card, reset
                    if reset:
                        self.next_state = self.reset_state()
                        self.terminated = True
                        self.AI_play.store(self.get_state(self.state),
                                           action,
                                           self.reward,
                                           self.get_state(self.state),
                                           self.terminated)
                        if verbose:
                            print("Action {}, {} failed".format(action, self.cards[action]))
                        return False
                    
                    #Legal move, save current state
                    self.current_state = self.state.copy()
                    card = self.cards[action]
                    self.players[self.AI_id].play_card(card)
                    trick[self.AI_id] = card

                    self.state[str(trick[self.AI_id])] = 5
                    
                #Let rest of players play
                for p in self.order[self.state['play_pos']+1:]:
                    trick[p] = self.players[p].play_random(self.trump, lead_card=lead_card)
                    self.state[str(trick[p])] = 5
                    
            #No AI     
            else:
                #Get lead card
                lead_card = self.players[self.order[0]].play_random(self.trump)
                trick = {self.order[0]: lead_card}
                #Every other player plays
                for p in self.order[1:]:
                    trick[p] = self.players[p].play_random(self.trump, lead_card=lead_card)

            #Get winner of trick and assign next trick order
            winner = self.evaluate_trick(trick, verbose=False)
            if verbose:
                print("Winner: {}\nTrick: {}\n".format(winner, trick))
            self.players[winner].tricks += 1
            self.players[self.players[winner].partner_id].tricks += 1
            self.get_order((winner-1)%4)

            #Update and save state
            if self.AI:
                #Update tricks
                if self.AI_id == winner or self.players[self.AI_id].partner_id == winner:
                    self.state['my_tricks'] += 1
                    self.reward = 1
                    self.tricks_won += 1
                else:
                    self.state['op_tricks'] += 1
                    self.reward = -1

                #Reset card states
                for p, c in trick.items():
                    self.state[str(c)] = p
                    
                #Update remaining tricks
                self.state['rem_tricks'] -= 1

                #Check if terminated, reset next state if so
                self.terminated = self.state['rem_tricks'] == 0
                if self.terminated:
                    self.next_state = self.reset_state()

                #Save
                self.AI_play.store(self.get_state(self.current_state),
                                           action,
                                           self.reward,
                                           self.get_state(self.next_state),
                                           self.terminated)
        
        #Assign score after all 5 tricks
        self.assign_score()
        
        #Reset tricks
        for p in self.players:
            p.reset_tricks()

        if self.AI:
            self.state['my_tricks'] = 0
            self.state['op_tricks'] = 0
            self.state['rem_tricks'] = 5
            self.state['lead'] = 4
            self.state['have_trick'] = 2
            for c in self.cards:
                self.state[str(c)] = 7

        #Successfully finished
        return True

                        
    def deal_cards(self):
        """
        Dealer hands out cards in order of 2,3,2,3,3,2,3,2
        """
        #Seed random
        #seed = int(time.time())
        #random.seed(seed)
        
        #Get deal order based on first player to the left of the dealer
        self.get_order(self.dealer)

        #Deal out cards
        cards_available = self.cards.copy()
        random.shuffle(cards_available)
        rnd_1 = [2, 3, 2, 3]
        rnd_2 = [3, 2, 3, 2]

        #First round
        for num, i in enumerate(self.order):
            for _ in range(rnd_1[num]):
                #card = random.choice(cards_available)
                card = cards_available[-1]
                cards_available.remove(card)
                self.players[i].add_card(card)
                
        #Second round 
        for num, i in enumerate(self.order):
            for _ in range(rnd_2[num]):
                #card = random.choice(cards_available)
                card = cards_available[-1]
                cards_available.remove(card)
                self.players[i].add_card(card)

        #Assign pickup card to choose
        self.pickup_card = cards_available[0]

        #If AI is involved, update the state of the game
        if self.AI:
            for c in self.players[self.AI_id].cards:
                self.state[str(c)] = self.AI_id
            self.state[str(self.pickup_card)] = 6
        
    def pickup(self):
        """
        Assigns total point values based on pickup card
        Sees if any players want to pickup the presented card
        If no pickup, go around and check if anyone wants to call for any suit
        """
        #Round 1 - for each player, get value, see if want to call
        for p in self.order:
            self.players[p].get_hand_value(self.pickup_card.suit, self.dealer==self.players[p].id, self.pickup_card)

            #If called, assign caller and have dealer pickup
            if self.players[p].call_pickup():
                self.trump = self.pickup_card.suit
                self.caller = self.players[p].id
                replacement_card = self.players[self.dealer].pickup(self.pickup_card)

                #If AI is involved, update the state of the game
                if self.AI:
                    self.state['trump'] = self.suits.index(self.trump)
                    self.state['caller'] = self.caller
                    if self.players[self.AI_id].id == self.dealer:
                        self.state[str(replacement_card)] = 6
                        self.state[str(self.pickup_card)] = self.AI_id
                        
                return True

        #Round 2 - go in order and check value for every possible trump
        #Remove pickup card suit as available call
        available_suits = self.suits.copy()
        available_suits.remove(self.pickup_card.suit)
        
        for p in self.order:
            for s in available_suits:
                self.players[p].get_hand_value(s)
                if self.players[p].call_pickup():
                    self.caller = self.players[p].id
                    self.trump = s
                    
                    #If AI is involved, update the state of the game
                    if self.AI:
                        self.state['trump'] = self.suits.index(self.trump)
                        self.state['caller'] = self.caller
                        
                    return True

        return False
    
    def get_order(self, starter):
        """
        Gets order for dealing, calling, or playing, based on starter
        """
        self.order = []
        for i in range(self.num_players):
            self.order.append((i+starter+1) % self.num_players)

    def evaluate_trick(self, trick, verbose=True):
        """
        Evaluates trick to see which card won
        Check if same trick but higher
        Check if trump
        """
        lead_card = trick[self.order[0]]
        lead_suit = lead_card.suit

        #Set winning player to first card
        winner = self.order[0]
        highest = lead_card

        for p, c in list(trick.items())[1:]:
            #Highest is trump, this card is not
            if (highest.suit == self.trump or highest.is_left(self.trump)) and (c.suit != self.trump and not c.is_left(self.trump)):
                continue
            
            #Highest is trump, this card is trump. Check value
            elif (highest.suit == self.trump or highest.is_left(self.trump)) and (c.suit == self.trump or c.is_left(self.trump)):
                highest_value = highest.trump_order.index(highest.value)
                c_value = c.trump_order.index(c.value)
                
                #matching if left and right
                if highest_value == c_value:
                    if c.suit == self.trump:
                        winner = p
                        highest = c
                        
                elif highest_value < c_value:
                    winner = p
                    highest = c

            #Highest is not trump, this card is
            elif (highest.suit != self.trump and not highest.is_left(self.trump)) and (c.suit == self.trump or c.is_left(self.trump)):     
                winner = p
                highest = c

            #Highest is not trump, this card is lead suit
            elif (highest.suit != self.trump and not highest.is_left(self.trump)) and (c.suit == lead_suit and not c.is_left(self.trump)):
                if c.value > highest.value:
                    winner = p
                    highest = c
        if verbose:
            print("Winner: {}, card: {}".format(winner, highest))
        return winner

    def assign_score(self):
        """
        Return the score for winning team
        Possibilities:
        Winning team called
        Winning team didn't call
        Winning team called and got all 5
        Loner
        """
        winners = (self.players[0], self.players[2]) if self.players[0].tricks > self.players[1].tricks else (self.players[1], self.players[3])
        if any(p.id == self.caller for p in winners):
            reward = 1
        elif not any(p.id == self.caller for p in winners):
            reward = 2
        elif any(p.id == self.caller for p in winners) and all(p.tricks == 5 for p in winners):
            reward = 2

        for p in winners:
            p.score += reward
        for p in self.players:
            print(p, p.score)
                
        return reward


    def get_state(self, state=None):
        """
        Gets the current state of the game
        See notes in reset_state method
        If provided a state, return as expanded dim numpy array, otherwise get state
        """
        if not state:
            return np.expand_dims(np.array(list(self.state.values())), axis=0)
        else:
            return np.expand_dims(np.array(list(state.values())), axis=0)

    def AI_action(self):
        action = self.AI_play.act(self.get_state())
        reset = False
        #check if card is in hands - if not, terminate game and start over
        if self.cards[action] not in self.players[self.AI_id].cards:
            self.reward = -10
            self.terminated = True
            reset = True

        return action, reset
            
g = Game()
