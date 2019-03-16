import gym
import random
import numpy as np

env = gym.make('CartPole-v0')

for i in range(100):
    fitness = 0
    state = env.reset()
    for _ in range(100):
        action = np.random.randint(0, 2)
        next_state, reward, done, _ = env.step(action)
        fitness += reward
        state = next_state

        if done:
            break


    print(fitness)