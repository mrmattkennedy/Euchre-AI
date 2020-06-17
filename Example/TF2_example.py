import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time

tf.compat.v1.disable_eager_execution()

seed = 1546847731  # or try a new seed by using: seed = int(time())
random.seed(seed)
print('Seed: {}'.format(seed))

class Game:
    board = None
    board_size = 0
    
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.reset()
    
    def reset(self):
        self.board = np.zeros(self.board_size)
    
    def play(self, cell):
        # returns a tuple: (reward, game_over?)
        if self.board[cell] == 0:
            self.board[cell] = 1
            game_over = len(np.where(self.board == 0)[0]) == 0
            return (1,game_over)
        else:
            return (-1,False)

def state_to_str(state):
    return str(list(map(int,state.tolist())))


class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size=4, output_size=4):
        self.q_target = tf.compat.v1.placeholder(shape=(None,output_size), dtype=tf.float32)
        self.r = tf.compat.v1.placeholder(shape=None,dtype=tf.float32)
        self.states = tf.compat.v1.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.enum_actions = tf.compat.v1.placeholder(shape=(None,2), dtype=tf.int32) 
        layer = self.states
        for l in hidden_layers_size:
            layer = tf.compat.v1.layers.dense(inputs=layer, units=l, activation=tf.nn.relu,
                                    kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed))
        self.output = tf.compat.v1.layers.dense(inputs=layer, units=output_size,
                                      kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=seed))
        self.predictions = tf.gather_nd(self.output,indices=self.enum_actions)
        self.labels = self.r + gamma * tf.reduce_max(input_tensor=self.q_target, axis=1)
        self.cost = tf.reduce_mean(input_tensor=tf.compat.v1.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)



class ReplayMemory:
    memory = None
    counter = 0

    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n):
        return random.sample(self.memory, n)

    

all_states = list()
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                s = np.array([i,j,k,l])
                all_states.append(state_to_str(s))
                
print('All possible states:')
for s in all_states:
    print(s)


game = Game()
num_of_games = 2000
epsilon = 0.1
gamma = 0.99
batch_size = 10
memory_size = 2000


tf.compat.v1.reset_default_graph()
tf.compat.v1.set_random_seed(seed)
qnn = QNetwork(hidden_layers_size=[20,20], gamma=gamma, learning_rate=0.001)
memory = ReplayMemory(memory_size)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


r_list = []  
c_list = []  # same as r_list, but for the cost

counter = 0  # will be used to trigger network training

import pdb

for g in range(num_of_games):
    game_over = False
    game.reset()
    total_reward = 0
    while not game_over:
        counter += 1
        state = np.copy(game.board)
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            pred = np.squeeze(sess.run(qnn.output,feed_dict={qnn.states: np.expand_dims(game.board,axis=0)}))
            action = np.argmax(pred)
        reward, game_over = game.play(action)
        total_reward += reward
        next_state = np.copy(game.board)
        memory.append({'state':state,'action':action,'reward':reward,'next_state':next_state,'game_over':game_over})
        if counter % batch_size == 0:
            # Network training
            batch = memory.sample(batch_size)
            q_target = sess.run(qnn.output,feed_dict={qnn.states: np.array(list(map(lambda x: x['next_state'], batch)))})
            terminals = np.array(list(map(lambda x: x['game_over'], batch)))
            for i in range(terminals.size):
                if terminals[i]:
                    # Remember we use the network's own predictions for the next state while calculatng loss.
                    # Terminal states have no Q-value, and so we manually set them to 0, as the network's predictions
                    # for these states is meaningless
                    q_target[i] = np.zeros(game.board_size)
                    
            _, cost = sess.run([qnn.optimizer, qnn.cost], 
                               feed_dict={qnn.states: np.array(list(map(lambda x: x['state'], batch))),
                               qnn.r: np.array(list(map(lambda x: x['reward'], batch))),
                               qnn.enum_actions: np.array(list(enumerate(map(lambda x: x['action'], batch)))),
                               qnn.q_target: q_target})
            c_list.append(cost)
    r_list.append(total_reward)
print('Final cost: {}'.format(c_list[-1]))


for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                b = np.array([i,j,k,l])
                if len(np.where(b == 0)[0]) != 0:
                    pred = np.squeeze(sess.run(qnn.output,feed_dict={qnn.states: np.expand_dims(b,axis=0)}))
                    pred = list(map(lambda x: round(x,3),pred))
                    action = np.argmax(pred)
                    print('board: {b}\tpredicted Q values: {p} \tbest action: {a}\tcorrect action? {s}'
                          .format(b=b,p=pred,a=action,s=b[action]==0))


plt.figure(figsize=(14,7))
plt.plot(range(len(c_list)),c_list)
plt.xlabel('Trainings')
plt.ylabel('Cost')
plt.show()
