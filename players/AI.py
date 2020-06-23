import os
import sys
import pdb
import time
import random
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
#from player import Player

#Append to use player class in parent directory
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath(__file__)))))
from players.player import Player
from card import Card


class DQL_Play_TF1(Player):
    def __init__(self, player_id, partner_id, state_size, action_size, seed=0, gamma=0.99, epsilon=0.1, hidden_sizes=[], alpha=0.001):
        super().__init__(player_id, partner_id, AI=True)
        
        tf.compat.v1.disable_eager_execution()
        seed = int(time.time()) if seed == 0 else seed
        self.memory = deque(maxlen=2000)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.state = self.reset_state()
        self.current_state = self.reset_state()
        self.legal = False
        self.cost_list = []

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(seed)

        self.q_target = tf.compat.v1.placeholder(shape=(None,action_size), dtype=tf.float32)
        self.r = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)
        self.states = tf.compat.v1.placeholder(shape=(None, state_size), dtype=tf.float32)
        self.enum_actions = tf.compat.v1.placeholder(shape=(None,2), dtype=tf.int32) 
        layer = self.states
        for l in hidden_sizes:
            layer = tf.compat.v1.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed))
        self.output = tf.compat.v1.layers.dense(inputs=layer, units=action_size,
                                        kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed))
        self.predictions = tf.gather_nd(self.output,indices=self.enum_actions)
        self.labels = self.r + gamma * tf.reduce_max(input_tensor=self.q_target, axis=1)
        self.cost = tf.reduce_mean(input_tensor=tf.compat.v1.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=alpha).minimize(self.cost)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=5000)
        
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def play(self, trump=None, lead_suit=None):
        
        #If legal action required, get most likely card from hand
        if self.legal:
            pred = np.squeeze(self.sess.run(self.output,feed_dict={self.states: self.get_state_as_array()}))
            action = -1

            #Loop through actions from most likely to least
            for i in pred.argsort()[::-1]:

                #If card in hand, check it
                if super().get_card(i) in self.cards:
                    
                    #If card is legal, use this and break
                    if not lead_suit or (lead_suit and self.legal_card(super().get_card(i), lead_suit, trump)):
                        action = i
                        break
                
        #If not legal action required, play according to DQL strategy   
        elif not self.legal:
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size-1)
            else:
                pred = np.squeeze(self.sess.run(self.output,feed_dict={self.states: self.get_state_as_array()}))
                action = np.argmax(pred)

        card = super().get_card(action)
        return card

    
    def store(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))
        
    def retrain(self, batch_size):
        # Network training
        batch = random.sample(self.memory, batch_size)

        state = 0
        action = 1
        reward = 2
        next_state = 3
        terminated = 4


        q_target = self.sess.run(self.output,feed_dict={self.states: np.array(list(map(lambda x: np.squeeze(x[next_state]), batch)))})
        terminals = np.array(list(map(lambda x: x[terminated], batch)))
        for i in range(terminals.size):
            if terminals[i]:
                q_target[i] = np.zeros(self.action_size)
        
        _, cost = self.sess.run([self.optimizer, self.cost], 
                           feed_dict={self.states: np.array(list(map(lambda x: np.squeeze(x[state]), batch))),
                           self.r: np.array(list(map(lambda x: np.squeeze(x[reward]), batch))),
                           self.enum_actions: np.array(list(enumerate(map(lambda x: np.squeeze(x[action]), batch)))),
                           self.q_target: q_target})
        
        if np.isnan(cost) or cost == np.inf:
            self.cost_list.append(1.0e+300)
            return False
        else:
            self.cost_list.append(cost)
            return True

    def get_cost_data(self, chunk_size = 1):
        return [sum(self.cost_list[i:i+chunk_size]) / chunk_size for i in range(0,len(self.cost_list),chunk_size)]

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
        for c in Player.cards:
            state[str(c)] = 7
        state['my_tricks'] = 0
        state['op_tricks'] = 0
        state['rem_tricks'] = 5
        state['play_pos'] = 0
        state['trump'] = 0
        state['lead'] = 4
        state['caller'] = 0
        state['have_trick'] = 2
        state['id'] = self.id
        state['p_id'] = self.partner_id
        
        return state
        
    def set_state(self, k, v):
        self.state[k] = v

    def get_state_key(self, k):
        return self.state[k]
    
    def get_state_as_dict(self, state=None):
        if not state:
            state  = self.state
        return state.copy()

    def get_state_as_array(self, state=None):
        if not state:
            state  = self.state
        return np.expand_dims(np.array(list(state.values())), axis=0)

    def reset(self):
        super().reset()
        self.state = self.reset_state()
        self.current_state = self.reset_state()
        
    def save(self, name):
        save_path = self.saver.save(self.sess, name)

    def load(self, name):
        self.saver.restore(self.sess, name)
        
if __name__ == '__main__':   
    q = DQL_Play_TF1(20, 24, hidden_sizes = [50, 50])

    for _ in range(20):
        state = np.random.randn(1, 20)
        act = q.act(state)
        q.store(state, act, 1, state, False)
    q.retrain(10)
