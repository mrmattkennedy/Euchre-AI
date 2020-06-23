import os
import pdb
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from card import Card
from players.player import Player
from players.AI import DQL_Play_TF1
from players.AI_dumb import AI_dumb
from players.human import Human

class Game:
    """
    Game class
    TODO:
    Add decision tree for pickup calls?
    """

    def __init__(self, p_types=None):
        self.create_cards()
        self.create_players(p_types)
        #self.state = self.reset_state()
        self.get_first_dealer()

    def create_AI(self):
        self.AI_play = DQL_Play_TF1(state_size=len(self.state),
                                action_size=self.num_cards,
                                hidden_sizes=[50,50])
        self.reward = 0
        self.terminated = False
        self.current_state = self.reset_state()
        self.next_state = self.reset_state()
        self.tricks_won = []
        self.AI_legal = False
        
    def reset(self):
        for p in self.players:
            p.reset()
        

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

        #Create cards list
        self.cards = []
        for s in self.suits:
            for n in card_nums:
                self.cards.append(Card(s, n))
    
    def create_players(self, p_types):
        """
        4 players, each with a partner
        Player 0 is partnered with player 2
        Player 1 is partnered with player 3
        """
        #Set Player class variable to copy of cards
        Player.cards = self.cards.copy()
        self.num_players = 4
        self.players = []
        
        for x, i in enumerate(p_types):
            p = None
            if i == 0:
                p = AI_dumb(x, (x+2) % self.num_players)
                
            elif i == 1:
                p = DQL_Play_TF1(x, (x+2) % self.num_players,
                                state_size=34, #Replace with len(state)
                                action_size=self.num_cards)
                
            elif i == 2:
                p = Human(x, (x+2) % self.num_players)
                
            self.players.append(p)

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


    def train(self, tricks_to_play=10000001):
        """
        Trains AI
        Given specified tricks to play, will try and play that many tricks
        Each attempt, reset and get new dealer
        Every 20000 tricks, save session
        Try and play a full game. If bad card, reset everything
        """
        counter = 0
        batch_size = 10
        start = time.time()
        
        for i in range(tricks_to_play):
            #Reset and get new dealer
            self.reset()
            self.get_first_dealer()
            play_game = True

            #Print every 1000
            if i % 1000 == 0:
                print(i)

            #Save every 20000
            if i % 20000 == 0 and i > 0:
                for p in self.players:
                    if p.AI:
                        p.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'players', 'saves', 'DQL_Play_{}_{}.ckpt'.format(p.id, i)))

            #Try and play full game. If bad call, exit the loop  
            while not any(p.score >= 10 for p in self.players) and play_game:
                self.deal_cards()
                called = self.pickup()
                if called:
                    play_game = self.play_trick()
                    counter += 1
                    if play_game:
                        i+=1
                    
                else:
                    #print('no call')
                    for p in self.players:
                        p.clear_hand()

            #update AI target network
            good_to_continue=True
            if counter % batch_size == 0 and counter != 0:
                counter += 1
                for p in self.players:
                    if p.AI and len(p.memory) >= batch_size:
                        good_cost = p.retrain(batch_size)
                        if not good_cost:
                            good_to_continue = False
    

            if not good_to_continue:
                break

        #Display time and cost data
        total_time = time.time() - start
        print("Time: {}s, {}m".format(round(total_time, 2), round(total_time / 60, 2)))

        plt.figure(figsize=(14,7))
        #Display cost data
        for p in self.players:
            if p.AI:
                c_list = p.get_cost_data(10000)
                plt.plot(range(len(c_list)),c_list)
                
        plt.xlabel('Trainings')
        plt.ylabel('Cost')
        plt.show()

    def play(self, ids, save_count, games_to_play=1):
        counter = 0
        total_wins = []

        for p in ids:
            self.players[p].legal=True
            self.players[p].load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'players', 'saves', 'DQL_Play_{}_{}.ckpt'.format(p, save_count)))

        for _ in range(games_to_play):
            self.reset()
            self.get_first_dealer()

            while not any(p.score >= 10 for p in self.players):
                self.deal_cards()
                called = self.pickup()
                if called:
                    self.play_trick(verbose=True)
                    
                else:
                    for p in self.players:
                        p.clear_hand()
            
        
    def play_trick(self, verbose=False):
        #Play, starting with lead and following with others
        if verbose:
            for p in self.players:
                print(p, p.AI)
                for c in p.cards:
                    print(c)
                print()
            print('\n' + str(self.trump))
            
        for _ in range(self.size_hand):
            
            #If lead is AI, reset state for that AI
            if self.players[self.order[0]].AI:
                self.players[self.order[0]].set_state(k='lead', v=4)
                self.players[self.order[0]].set_state(k='have_trick', v=2)
                
            self.lead_suit=None
                
            #Get lead card
            lead_card = self.players[self.order[0]].play(trump=self.trump)

            #If AI, evaluate if legal
            if self.players[self.order[0]].AI:
                self.players[self.order[0]].current_state = self.players[self.order[0]].get_state_as_dict() 
                reset = self.AI_action(lead_card, self.order[0])

                if reset:
                    terminated = True
                    self.players[self.order[0]].store(self.players[self.order[0]].get_state_as_array(self.players[self.order[0]].current_state),
                                                      self.cards.index(lead_card),
                                                      self.reward,
                                                      self.players[self.order[0]].get_state_as_array(self.players[self.order[0]].reset_state()),
                                                      terminated) 
                    if verbose:
                        print("Action {}, ID {} failed".format(lead_card, self.players[self.order[0]]))
                    return False
                
                #Legal move, save current state
                else:
                    self.players[self.order[0]].cards.remove(lead_card)
                    for p in self.players:
                        if p.AI:
                            p.set_state(k=str(lead_card), v=5)
                            p.set_state(k='lead', v=self.suits.index(lead_card.suit))
                            
            #Legal move     
            trick = {self.order[0]: lead_card}
            if lead_card.is_left(self.trump):
                self.lead_suit = self.trump
            else:
                self.lead_suit = lead_card.suit
            
            #Every other player plays
            for p in self.order[1:]:
                #See if currently winning
                if self.players[p].AI:
                    current_winner = self.evaluate_trick(trick, verbose=False)
                    self.players[p].set_state(k='have_trick', v=int(current_winner == self.players[p].partner_id))
                    
                action = self.players[p].play(trump=self.trump, lead_suit=self.lead_suit)
                #If AI, evaluate if legal
                if self.players[p].AI:
                    
                    self.players[p].current_state = self.players[p].get_state_as_dict()
                    reset = self.AI_action(action, p)

                    if reset:
                        terminated = True
                        self.players[p].store(self.players[p].get_state_as_array(self.players[p].current_state),
                                                          self.cards.index(lead_card),
                                                          self.reward,
                                                          self.players[p].get_state_as_array(self.players[p].reset_state()),
                                                          terminated) 
                        if verbose:
                            print("Action {}, ID {} failed".format(action, p))
                        return False

                    #Legal move
                    else:
                        self.players[p].cards.remove(action)
                        #Update state for all AI
                        for player in self.players:
                            if player.AI:
                                player.set_state(k=str(action), v=5)
      
                trick[p] = action

                
            #Get winner of trick and assign next trick order
            winner = self.evaluate_trick(trick, verbose=False)
            if verbose:
                print("Winner: {}\nTrick: {}\n".format(winner, trick))
            self.players[winner].tricks += 1
            self.players[self.players[winner].partner_id].tricks += 1
            self.get_order((winner-1)%4)

            for p in self.players:
                if p.AI:
                    #Update state
                    if p.id == winner or p.partner_id == winner:
                        p.set_state(k='my_tricks', v=p.get_state_key('my_tricks') + 1)
                        self.reward = 1
                    else:
                        p.set_state(k='op_tricks', v=p.get_state_key('op_tricks') + 1)
                        self.reward = -1

                    #Reset card states
                    for pid, c in trick.items():
                        p.set_state(k=str(c), v=pid)

                    #Update remaining tricks
                    p.set_state(k='rem_tricks', v=p.get_state_key('rem_tricks') - 1)
                    terminated = p.get_state_key('rem_tricks') == 0
                    
                    if terminated:
                        p.state = p.reset_state()
                        p.store(p.get_state_as_array(p.current_state),
                                  self.cards.index(lead_card),
                                  self.reward,
                                  p.get_state_as_array(p.state),
                                  terminated) 
        
        #Assign score after all 5 tricks
        self.assign_score(verbose)
        
        #Reset tricks
        for p in self.players:
            p.reset_tricks()

            if p.AI:
                p.set_state(k='my_tricks', v=0)
                p.set_state(k='op_tricks', v=0)
                p.set_state(k='rem_tricks', v=0)
                p.set_state(k='lead', v=0)
                p.set_state(k='have_trick', v=2)
                
                for c in self.cards:
                    p.set_state(k=str(c), v=7)

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

        #If AI is involved, update the state of the game for AI
        for p in self.players:
            if p.AI:
                for c in p.cards:
                    p.set_state(k=str(c), v=4)
                p.set_state(k=str(self.pickup_card), v=6)
        
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
                for p in self.players:
                    if p.AI:
                        p.set_state(k='trump', v=self.suits.index(self.trump))
                        p.set_state(k='caller', v=self.caller)
                        p.set_state(k=str(replacement_card), v=6)
                        
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
                    for p in self.players:
                        if p.AI:
                            p.set_state(k='trump', v=self.suits.index(self.trump))
                            p.set_state(k='caller', v=self.caller)
                        
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

    def assign_score(self, verbose):
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

        if verbose:
            for p in self.players:
                print(p, p.score)
            print()
                
        return reward

    def AI_action(self, action, pid):                
        reset = False
        #check if card is in hands - if not, terminate game and start over
        if not self.players[pid].legal_card(action, self.lead_suit, self.trump):
            self.reward = -10
            self.terminated = True
            reset = True

        return reset

    def get_tricks_avg(self, chunk_size=1):
        return [sum(self.tricks_won[i:i+chunk_size]) / chunk_size for i in range(0,len(self.tricks_won),chunk_size)]

    def test_saved_models(self, start, stop, step, ids=[]):
        sizes = [i for i in range(start, stop, step)]
        total_wins = []
        games_to_play = 100

        for p in ids:
            self.players[p].legal=True
                
        for i in range(start, stop, step):
            for p in ids:
                self.players[p].load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'players', 'saves', 'DQL_Play_{}_{}.ckpt'.format(p, i)))

            wins = [0, 0, 0, 0]
            for _ in range(games_to_play):
                self.reset()
                self.get_first_dealer()

                while not any(p.score >= 10 for p in self.players):
                    self.deal_cards()
                    called = self.pickup()
                    if called:
                        self.play_trick()
                        
                    else:
                        for p in self.players:
                            p.clear_hand()


                for x, p in enumerate(self.players):
                    if p.score >= 10:
                        wins[x] += 1

            total_wins.append([wins[0], wins[1]])
            print("{}: {}".format(i, wins))
        
        plt.figure(figsize=(14,7))
        plt.plot(range(start, stop, step), [x[0] for x in total_wins])
        plt.plot(range(start, stop, step), [x[1] for x in total_wins])
        plt.xlabel('Tricks played when saved')
        plt.ylabel('Wins')
        plt.show()


if __name__ == "__main__":
    p_types = input("Enter types for player (0=AI_Dumb, 1=AI, 2=Human):\n")
    p_types = [int(p) for p in "".join(p_types.split()).split(',')]
    p_types = [1,0,1,0]
    g = Game(p_types)
    #g.play([0,2], 10000000)
    tricks = 100000000
    save_step = 20000
    g.train(tricks+1)
    
    g.test_saved_models(save_step, tricks+1, save_step, [0,1,2,3])
    
    p_types = [1,0,1,0]
    g = Game(p_types)
    g.test_saved_models(save_step, tricks+1, save_step, [0,2])
    
    p_types = [0,1,0,1]
    g = Game(p_types)
    g.test_saved_models(save_step, tricks+1, save_step, [1,3])

