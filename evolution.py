import os
import json
from player import Player
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import copy
import random
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.records = {
            "generation": [],
            "max": [],
            "min": [],
            "avg": []
        }
        try:
            os.remove('plot/records.json')
        except OSError:
            pass


    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]


    def gaussian_noise(self, x: np.ndarray, mu = 0.0, std = 0.1):
        return x + np.random.normal(mu, std, size = x.shape)



    def mutate(self, child):

        # TODO
        # child: an object of class `Player` 
        wthresh = 0.35
        bthresh = 0.7
        mu = 0.0
        std = 0.3
          
        for i in range(0, 2):
            prob = np.random.uniform(0, 1)
            if prob <= wthresh:
                child.nn.weights[i] = self.gaussian_noise(child.nn.weights[i], mu, std)
                
            prob = np.random.uniform(0, 1)
            if prob <= bthresh:
                child.nn.biases[i] = self.gaussian_noise(child.nn.biases[i], mu, std)
            
        return child


    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:

            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects
            # parents = sorted(prev_players, key = lambda x: x.fitness, reverse = True)
            # new_players = copy.deepcopy(parents[:num_players])

            # TODO (additional): a selection method other than `fitness proportionate`
            parents = self.q_tournament_selection(prev_players, num_players, 15)
            # TODO (additional): implementing crossover
            
            new_players = []
            for i in range(0, num_players, 2):
                mother = parents[i]
                father = parents[i + 1]
                # crossover
                new_players.append(self.crossover(mother, father)) # child 1
                new_players.append(self.crossover(father, mother)) # child 2

            for player in new_players:
                if np.random.rand() > 0.3:
                    player = self.mutate(player)

            return new_players


    def crossover(self, mother: Player, father: Player) -> Player:
        child = copy.deepcopy(mother)

        # Weights 1 (horizontal split replacement of mother/father)
        w0_split_point = mother.nn.weights[0].shape[0] // 2
        if random.choice([0, 1]) == 0:
            child.nn.weights[0][:w0_split_point, :] = mother.nn.weights[0][:w0_split_point, :]
            # child.nn.weights[0][:w0_split_point, :] = random.choice([mother, father]).nn.weights[0][:w0_split_point, :]
        else:
            child.nn.weights[0][w0_split_point:, :] = father.nn.weights[0][w0_split_point:, :]
            # child.nn.weights[0][w0_split_point:, :] = random.choice([mother, father]).nn.weights[0][w0_split_point:, :]

        # Weights 2 (vertical split replacement of mother/father)
        w1_split_point = mother.nn.weights[1].shape[0] // 2
        if random.choice([0, 1]) == 0:
            child.nn.weights[1][:, w1_split_point:] = mother.nn.weights[1][:, w1_split_point:]
            # child.nn.weights[1][:, w1_split_point:] = random.choice([mother, father]).nn.weights[1][:, w1_split_point:]
        else:
            child.nn.weights[1][:, :w1_split_point] = father.nn.weights[1][:, :w1_split_point]
            # child.nn.weights[1][:, :w1_split_point] = random.choice([mother, father]).nn.weights[1][:, :w1_split_point]

        return child


    def next_population_selection(self, players: list, num_players: int):

        # TODO Top-k selection
        # players = self.k_top_selection(players, num_players)

        # TODO (additional): Roulette wheel selection
        players = self.roulette_wheel_selection(players, num_players)

        # TODO (additional): plotting

        self.save_fitness_history(players)  # Save evo
        return players[:num_players]


    def k_top_selection(self, players: list, num_players: int):
        return sorted(players, key = lambda x: x.fitness, reverse = True)


    def q_tournament_selection(self, players: list, num_players: int, q: int):
        result = []
        for i in range(num_players):
            batch = []
            for j in range(q):
                batch.append(np.random.choice(players))
            result.append(copy.deepcopy(sorted(batch, key=lambda x: x.fitness, reverse=True)[0]))
        return result


    def roulette_wheel_selection(self, players: list, num_players: int):
        fitnesses = np.array([p.fitness for p in players])
        fitnesses = fitnesses / np.sum(fitnesses)
        players = np.random.choice(players, num_players, p = fitnesses).tolist()

        return sorted(players, key = lambda x: x.fitness, reverse = True)


    def save_fitness_history(self, players: list) -> None:
        """
            Plot the data and save it to a file

            Parameters:
                players (list): list of players
        """
        fitnesses = [player.fitness for player in players]
        self.records["generation"].append(len(self.records["max"]))
        self.records["max"].append(max(fitnesses))
        self.records["min"].append(min(fitnesses))
        self.records["avg"].append(sum(fitnesses) / len(players))

        with open('plot/records.json', 'w') as f:
            json.dump(self.records, f)


    def plot_fitness_history(self) -> None:
        """
            Plot the data and save it to a file
        """
        pd.DataFrame(data = self.records).plot()
        plt.show()