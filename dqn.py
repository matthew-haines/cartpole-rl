import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

random.seed(16) # Allows for remakability

class Network:

    def __init__(self, state_space=4, action_space=2, learning_rate=0.0025):

        self.model = Sequential()
        self.model.add(Dense(32, input_dim=state_space, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_space, activation='relu'))

        self.model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    def predict(self, x):
        y_hat = self.model.predict(x)
        return y_hat

    def train(self, x, y):
        self.model.fit(x, y, epochs=1, verbose=0)  # We dont want much output


class Agent:

    def __init__(self, state_space, action_space, **kwargs):

        self.params = {  # Default parameters
            "replay_length": 1000,
            "initial_exploration": 1.0,
            "exploration_decay": 0.995,
            "min_exploration": 0.005,
            "discount": 0.95,
            "learning_rate": 0.0025
        }

        # replaces defaults parameters with user defined if exist
        for (param, val) in (self.params.items()):
            setattr(self, param, kwargs.get(param, val))

        self.state_space = state_space
        self.action_space = action_space

        self.exploration = self.params['initial_exploration']

        self.model = Network(self.state_space, self.action_space,
                             learning_rate=self.params['learning_rate'])
        self.replay_memory = deque(maxlen=self.params['replay_length'])

    def replay(self, batch_size):
        """ 
        We use a variant of mean absolute error as our function to update the network:
        |r+\gamma \arg\max_{a^\prime}Q(s^\prime,a^\prime)-Q(s, a)|
        Minibatches improve training speed 
        """

        batch = []
        for i in range(batch_size):
            batch.append(random.choice(self.replay_memory))

        for state, action, reward, next_state, done in batch:

            # if the simulation is done, next state does not matter
            y = reward

            if not done:
                y = reward + self.params['discount'] * \
                    np.argmax(self.model.predict(next_state))

            predicted_y = self.model.predict(state)
            # Now we map the predicted target with the target at the action taken by the agent
            predicted_y[0][action] = y

            self.model.train(state, predicted_y)

        # decay exploration rate
        self.exploration = np.max(
            [self.params['min_exploration'], self.params['exploration_decay'] * self.exploration])

    def add_to_memory(self, state, action, reward, next_state, done):
        # Adds scenario to memory
        self.replay_memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if (random.random()-0.1) / 0.9 > self.exploration:
            return random.randint(0, self.state_space-1)

        return np.argmax(self.model.predict(state)[0])

env = gym.make('CartPole-v0')

agent = Agent(env.observation_space.shape[0], env.action_space.n)

episodes = 1000
time_range = 5000

batch_size = 32

for episode in range(episodes):
    done = False
    state = env.reset()
    state = np.reshape(state, [1, agent.state_space])

    for timestep in range(time_range):
        action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        if done: # penalize losing
            reward = -5

        agent.add_to_memory(state, action, reward, next_state, done)
        next_state = np.reshape(next_state, [1, agent.state_space])
        state = next_state # step forward

        if done:
            print("Episode: {}, Time Survived: {}".format(episode, timestep)) 
            break 

    if len(agent.replay_memory) > batch_size:
        agent.replay(batch_size)

