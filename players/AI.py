import os
import sys
import pdb
import time
import random
import numpy as np
import tensorflow as tf
from collections import deque
from player import Player

#Append to use player class in parent directory
sys.path.insert(0, str(Path(os.path.dirname(os.path.realpath(__file__))).parent))
from card import Card


class DQL_Play_TF2(Player):
    def __init__(self, player_id, partner_id, state_size, action_size, gamma=0.99, epsilon=0.1, hidden_sizes=[], alpha=0.01):
        super().__init__(player_id, partner_id)
        
        self.state_size = state_size
        self.action_size = action_size

        #Create replay
        self.memory = deque(maxlen=2000)

        #Set gamma and epsilon
        self.gamma = gamma
        self.epsilon = epsilon

        #Create networks
        self.q_network = self.create_model(hidden_sizes, alpha)
        self.target_network = self.create_model(hidden_sizes, alpha)
        self.align_models()
        
        self.cost = []
        
    def create_model(self, hidden_sizes, alpha):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.state_size), dtype= tf.float32))
        
        #Add hidden layers
        for size in hidden_sizes:
            model.add(tf.keras.layers.Dense(units=size, activation="relu"))

        #Add output layer and compile
        model.add(tf.keras.layers.Dense(units=self.action_size))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(alpha))
        return model
        

    def align_models(self):
        self.target_network.set_weights(self.q_network.get_weights())


    def store(self, state, action, reward, next_state, terminated):
        self.memory.append((state, action, reward, next_state, terminated))

    def play(self, state, trump=None, lead_suit=None, legal=False):
        #If legal action required, get most likely card from hand
        if legal:
            pred = self.q_network.predict(state)
            action = -1

            #Loop through actions from most likely to least
            for i in pred.argsort()[::-1]:

                #If card in hand, check it
                if super().get_card(i) in self.cards:
                    
                    #If card is legal, use this and break
                    if not lead_suit or (lead_suit and self.legal_card(self.cards_list[i], lead_suit, trump)):
                        action = i
                        break
                
        #If not legal action required, play according to DQL strategy   
        elif not legal:
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size-1)
            else:
                pred = self.q_network.predict(state)
                action = np.argmax(pred)

        return action

    def retrain(self, batch_size):        
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)

            #Reset the state
            if terminated:
                target[0] = np.zeros(self.action_size)
            else:
                t = self.target_network.predict(np.expand_dims(next_state, axis=0))
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)



class DQL_Play_TF1(Player):
    def __init__(self, player_id, partner_id, cards_list, state_size, action_size, seed=0, gamma=0.99, epsilon=0.1, hidden_sizes=[], alpha=0.001):
        super().__init__(player_id, partner_id, cards_list)
        
        tf.compat.v1.disable_eager_execution()
        seed = int(time.time()) if seed == 0 else seed
        self.memory = deque(maxlen=2000)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
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

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def play(self, state, legal):
        
        #If legal action required, get most likely card from hand
        if legal:
            pred = np.squeeze(self.sess.run(self.output,feed_dict={self.states: state}))
            action = -1

            #Loop through actions from most likely to least
            for i in pred.argsort()[::-1]:

                #If card in hand, check it
                if super().get_card(i) in self.cards:
                    
                    #If card is legal, use this and break
                    if not lead_suit or (lead_suit and self.legal_card(self.cards_list[i], lead_suit, trump)):
                        action = i
                        break
                
        #If not legal action required, play according to DQL strategy   
        elif not legal:
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size-1)
            else:
                pred = np.squeeze(self.sess.run(self.output,feed_dict={self.states: state}))
                action = np.argmax(pred)

        return action

    
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

    def save(self, name):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(self.sess, name)

    def load(self, name):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, name)
        
if __name__ == '__main__':   
    q = DQL_Play_TF1(20, 24, hidden_sizes = [50, 50])

    for _ in range(20):
        state = np.random.randn(1, 20)
        act = q.act(state)
        q.store(state, act, 1, state, False)
    q.retrain(10)
