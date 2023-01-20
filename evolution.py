from player import Player
import numpy as np
import copy
import random
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode
        self.records = []


    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]


    def gaussian_noise(self, x: np.ndarray, mu = 0.0, std = 0.1):
        noise = np.random.normal(mu, std, size = x.shape)

        threshold = 0.3
        def random_chocie(x):
            return 0 if x < threshold else x

        x_noisy = x + np.vectorize(random_chocie)(noise)
        return x_noisy


    def mutate(self, child):

        # TODO
        # child: an object of class `Player`
        child.nn.biases[0] = self.gaussian_noise(child.nn.biases[0])
        child.nn.biases[1] = self.gaussian_noise(child.nn.biases[1])
        child.nn.weights[0] = self.gaussian_noise(child.nn.weights[0])
        child.nn.weights[1] = self.gaussian_noise(child.nn.weights[1])
        
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
            child.nn.weights[0][:w0_split_point, :] = random.choice([mother, father]).nn.weights[0][:w0_split_point, :]
        else:
            child.nn.weights[0][w0_split_point:, :] = random.choice([mother, father]).nn.weights[0][w0_split_point:, :]

        # Weights 2 (vertical split replacement of mother/father)
        w1_split_point = mother.nn.weights[1].shape[0] // 2
        if random.choice([0, 1]) == 0:
            child.nn.weights[1][:, w1_split_point:] = random.choice([mother, father]).nn.weights[1][:, w1_split_point:]
        else:
            child.nn.weights[1][:, :w1_split_point] = random.choice([mother, father]).nn.weights[1][:, :w1_split_point]

        return child


    def next_population_selection(self, players, num_players):

        # TODO Top-k selection
        # players = self.k_top_selection(players, num_players)

        # TODO (additional): Roulette wheel selection
        players = self.roulette_wheel_selection(players, num_players)

        # TODO (additional): plotting
        self.records.append({
            'generation': len(self.records),
            'best_fitness': players[0].fitness,
            'worst_fitness': players[-1].fitness,
            'avg_fitness': np.mean([p.fitness for p in players])
        })

        print(self.records[-1])

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