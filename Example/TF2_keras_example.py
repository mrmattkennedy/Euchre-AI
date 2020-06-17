import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time


#x_train = np.random.normal(size=(3, 2))
#astensor = tf.convert_to_tensor(x_train)
#logits = tf.keras.layers.Dense(2)(astensor)
#print(logits.numpy())
#temp = tf.keras.Input(shape=(), dtype=tf.float32)
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

input_ = np.random.normal(size=(3, 2))
x = tf.placeholder(tf.float32, shape=(None, 2))
logits = tf.keras.layers.Dense(2)(x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(logits, feed_dict={x:input_}))
"""
seed = int(time())
random.seed(seed)

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

class QNetwork:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.01, hidden_sizes=[], learning_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size

        #Create replay
        self.experience_replay = deque(maxlen=2000)

        #Set gamma and epsilon
        self.gamma = gamma
        self.epsilon = epsilon
        
        #Create model
        self.q_network = self.create_model(hidden_sizes, learning_rate)
        self.target_network = self.create_model(hidden_sizes, learning_rate)

        self.cost = []

    def create_model(self, hidden_sizes, learning_rate):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.action_size), dtype= tf.float32))
        
        #Add hidden layers
        for size in hidden_sizes:
            model.add(tf.keras.layers.Dense(units=size, activation="relu"))

        #Add output layer and compile
        model.add(tf.keras.layers.Dense(units=self.action_size))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate))
        return model

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def store(self, state, action, reward, next_state, terminated):
        self.experience_replay.append((state, action, reward, next_state, terminated))
        
    def act(self, state):
        if random.random() < epsilon:
            action = random.randint(0, self.action_size-1)
        else:
            pred = self.q_network.predict(state)
            action = np.argmax(pred)

        return action

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, terminated in minibatch:
            target = self.q_network.predict(state)
            if terminated:
                target[0] = np.zeros(game.board_size)
            else:
                t = self.target_network.predict(np.expand_dims(next_state, axis=0))
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)
            
            states.append(state)
            targets.append(target)

        states = np.array(states).reshape(len(states), self.action_size)
        targets = np.array(targets).reshape(len(targets), self.action_size)
        self.cost.append(np.sum(np.square(states-target)))

game = Game()
epsilon = 0.1
gamma = 0.99

agent = QNetwork(4, 4, hidden_sizes=[20,20], gamma=gamma, epsilon=epsilon, learning_rate=0.01)
batch_size = 10
num_of_episodes = 10000
timesteps_per_episode = 1000
timesteps = []

import time
start = time.time()
for e in range(0, num_of_episodes):
    game_over = False
    game.reset()
    
    for timestep in range(timesteps_per_episode):
        # Run Action
        state = np.expand_dims(game.board,axis=0)
        action = agent.act(state)
        
        # Take action    
        reward, terminated = game.play(action)
        next_state = np.copy(game.board)

        #Store
        agent.store(state, action, reward, next_state, terminated)
        state = next_state

        if terminated:
            agent.align_target_model()
            timesteps.append(timestep)
            break
            
        if timestep % batch_size == 0 and timestep != 0:
            agent.retrain(batch_size)
    
    if (e + 1) % 10 == 0:
        print("Episode: {}".format(e + 1))
        



for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                b = np.array([i,j,k,l])
                if len(np.where(b == 0)[0]) != 0:
                    pred = np.squeeze(agent.q_network.predict(np.expand_dims(b,axis=0)))
                    pred = list(map(lambda x: round(x,3),pred))
                    action = np.argmax(pred)
                    print('board: {b}\tpredicted Q values: {p} \tbest action: {a}\tcorrect action? {s}'
                          .format(b=b,p=pred,a=action,s=b[action]==0))

print(time.time() - start)

plt.figure(figsize=(14,7))
plt.plot(range(len(agent.cost)),agent.cost)
plt.xlabel('Trainings')
plt.ylabel('Cost')
plt.show()

plt.figure(figsize=(14,7))
plt.plot(range(len(timesteps)),timesteps)
plt.xlabel('Trainings')
plt.ylabel('Steps')
plt.show()

