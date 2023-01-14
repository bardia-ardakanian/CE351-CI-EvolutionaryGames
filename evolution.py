from player import Player
import numpy as np
import copy
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

        threshold = 0.5
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
            parents = self.roulette_wheel_selection(prev_players, num_players)
            # TODO (additional): implementing crossover
            # make sure we have enough parents
            if num_players % 2 == 1:
                parents.append(parents[-1])
            
            new_players = []
            for i in range(0, num_players, 2):
                mother = parents[i]
                father = parents[i + 1]
                # crossover
                # first child
                new_players.append(self.crossover(mother, father))
                # second child
                new_players.append(self.crossover(father, mother))

            for player in new_players:
                if np.random.rand() > 0.5:
                    player = self.mutate(player)

            return new_players


    def crossover(self, mother: Player, father: Player) -> Player:
        child = copy.deepcopy(mother)
        wmid = mother.nn.weights[0].size // 2
        bmid = mother.nn.biases[0].size // 2

        child.nn.weights[0][wmid:] = father.nn.weights[0][wmid:]
        child.nn.biases[0][bmid:] = father.nn.biases[0][bmid:]
        
        child.nn.weights[1][wmid:] = father.nn.weights[1][wmid:]
        child.nn.biases[1][bmid:] = father.nn.biases[1][bmid:]
        
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

    def roulette_wheel_selection(self, players: list, num_players: int):
        fitnesses = np.array([p.fitness for p in players])
        fitnesses = fitnesses / np.sum(fitnesses)
        return np.random.choice(players, num_players, p = fitnesses).tolist()