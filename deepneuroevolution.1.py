import gym
import numpy as np
from copy import deepcopy
import random
import pickle

np.random.seed(16)


def RELU(x):
    return x * (x > 0)  # Fastest RELU implementation, same as max(x, 0)


def Linear(x):
    return x


def evaluate_network(layers, x):

    value = x
    for layer in layers:
        value = layer.compute(value)

    return value


class Layer:

    def __init__(self, input_dimension, output_dimension, activation):

        # Xavier Initialization
        self.weight = np.random.randn(input_dimension, output_dimension)
        self.bias = np.zeros((1, output_dimension))

        self.activation = activation

    def compute(self, x):
        y = self.activation(np.dot(x, self.weight) + self.bias)
        return y


class Individual:

    def __init__(self, state_space, action_space, update_coefficient=0.005):

        self.layers = []
        self.layers.append(Layer(state_space, 32, RELU))
        self.layers.append(Layer(32, 16, RELU))
        self.layers.append(Layer(16, action_space, Linear))

        self.update_coefficient = update_coefficient

    def mutate(self):

        for layer in self.layers:
            layer.weight += np.random.randn(*layer.weight.shape) * \
                self.update_coefficient
            layer.bias += np.random.randn(*layer.bias.shape) * \
                self.update_coefficient

    def compute(self, x):

        value = x
        for layer in self.layers:
            value = layer.compute(value)

        return value


class Population:

    def __init__(self, state_size, action_size, individual_count=100, update_coefficient=0.005):

        self.state_size = state_size
        self.action_size = action_size

        self.individual_count = individual_count

        self.update_coefficient = update_coefficient

        self.individuals = []
        for i in range(self.individual_count):
            self.individuals.append(Individual(
                self.state_size, self.action_size, self.update_coefficient))

        self.fitnesses = np.zeros((self.individual_count))

    def evaluate(self, env, selected_individuals, time_limit, render=False):

        i = 0

        for individual in self.individuals:

            done = False
            state = env.reset()

            fitness = 0

            for step in range(time_limit):
                action = np.argmax(individual.compute(state))
                if render:
                    env.render()

                next_state, reward, done, _ = env.step(action)

                fitness += reward

                if done:
                    break

                state = next_state
                if step == time_limit - 1:
                    print("survived")

            self.fitnesses[i] = fitness

        # Find the top individuals
        best_individual = np.argmax(self.fitnesses)
        print("Best individual of population had fitness of {}".format(
            self.fitnesses[best_individual]))
        self.fitnesses[best_individual] = 0
        top_individuals = []
        for i in range(selected_individuals):
            top_individuals.append(np.argmax(self.fitnesses))
            # Remove the highest score to allow for others
            self.fitnesses[top_individuals[i]] = 0

        new_population = []
        new_population.append(self.individuals[best_individual])

        for i in range(self.individual_count):
            new_individual = deepcopy(
                self.individuals[random.choice(top_individuals)])
            new_individual.mutate()
            new_population.append(new_individual)

        self.individuals = new_population

    def save(self, name):
        with open('saved_models/{}_dne.pkl'.format(name), 'wb') as output_file:
            # This is the best individual
            pickle.dump(self.individuals[0], output_file)


# Actual training
env = gym.make('CartPole-v0')

population = Population(env.observation_space.shape[0], env.action_space.n)

mutation_count = 1000
time_limit = 5000
selected_individuals = 10

for _ in range(mutation_count):

    population.evaluate(env, selected_individuals, time_limit)

    if _ % 5:
        population.save('population_{}'.format(_))

env.close()
