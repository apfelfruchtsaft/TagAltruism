import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


class Model:
    def __init__(self, N, cost, benefit, pairings, mutation_rate, min_tolerance, cheater_mutation_rate, n_neighbors,
                 generations, mu, sigma, seed=None):
        self.N = N
        self.cost = cost
        self.benefit = benefit
        self.pairings = pairings
        self.mutation_rate = mutation_rate
        self.min_tolerance = min_tolerance
        self.cheater_mutation_rate = cheater_mutation_rate
        self.n_neighbors = n_neighbors
        assert self.n_neighbors % 2 == 0
        self.generations = generations
        self.mu = mu
        self.sigma = sigma
        self.seed = seed
        self.tags = np.zeros((generations, N))
        self.tolerances = np.zeros((generations, N))
        self.cheater_flags = np.zeros((generations, N), dtype=bool)
        self.donation_rates = np.zeros(generations)
        self.output = []

    def interaction(self, i, partner, fitnesses, cheater_flag, tags, tolerance,
                    interactions_attempted, interactions_made):
        interactions_attempted += 1
        if cheater_flag == False and abs(tags[i] - tags[partner]) <= tolerance:
            fitnesses[i] -= self.cost
            fitnesses[partner] += self.benefit
            interactions_made += 1
        return fitnesses, interactions_attempted, interactions_made

    def reproduction(self, i, mate, fitnesses, tags, tolerances, cheater_flags):
        rng = np.random.default_rng(self.seed)
        fitness = fitnesses[i]
        fitness_mate = fitnesses[mate]

        if fitness > fitness_mate:
            parent = i
        elif fitness < fitness_mate:
            parent = mate
        else:
            parent = rng.choice([i, mate])

        child_tag = tags[parent]
        child_tolerance = tolerances[parent]
        child_cheater_flag = cheater_flags[parent]

        if rng.random() < self.mutation_rate:
            child_tag = rng.random()

        if rng.random() < self.mutation_rate:
            noise = rng.normal(self.mu, self.sigma)
            child_tolerance += noise
            if child_tolerance < self.min_tolerance: child_tolerance = self.min_tolerance
        if rng.random() <= self.mutation_rate:
            child_cheater_flag = not (child_cheater_flag)

            return child_tag, child_tolerance, child_cheater_flag

        return child_tag, child_tolerance, child_cheater_flag

    def simulate(self):
        # generate initial data
        rng = np.random.default_rng(self.seed)
        child_tags = rng.uniform(low=0, high=1, size=self.N)
        child_tolerances = rng.uniform(low=self.min_tolerance, high=1, size=self.N)
        child_cheater_flags = np.zeros(self.N)

        self.tags[0, :] = child_tags
        self.tolerances[0, :] = child_tolerances

        for g in tqdm(range(1, self.generations + 1)):
            tags = child_tags.copy()
            tolerances = child_tolerances.copy()
            cheater_flags = child_cheater_flags.copy()
            fitnesses = np.zeros(self.N)

            interactions_made, interactions_attempted = 0, 0
            G = nx.circulant_graph(n=self.N, offsets=[i + 1 for i in range(int(self.n_neighbors / 2))])
            neighbors = [list(nx.all_neighbors(G, i)) for i in range(self.N)]

            for i in range(self.N):
                for p in range(self.pairings):
                    partner = rng.choice(neighbors[i])
                    fitnesses, interactions_attempted, interactions_made = self.interaction(i, partner, fitnesses,
                                                                                            cheater_flags[i], tags,
                                                                                            tolerances[i],
                                                                                            interactions_attempted,
                                                                                            interactions_made)
            for i in range(self.N):
                mate = rng.choice(neighbors[i])
                child_tags[i], child_tolerances[i], child_cheater_flags[i] = self.reproduction(i, mate, fitnesses, tags,
                                                                                               tolerances,
                                                                                               cheater_flags)
            self.tags[g - 1, :] = child_tags
            self.tolerances[g - 1, :] = child_tolerances
            self.cheater_flags[g - 1, :] = child_cheater_flags
            self.donation_rates[g - 1] = interactions_made / interactions_attempted
            self.output.append(
                [g, interactions_attempted, interactions_made, tags, tolerances, cheater_flags, fitnesses])

    def save(self, simulation_name, directory):
        output_df = pd.DataFrame(self.output, columns=(
            'generation', 'interactions_attempted', 'interactions_made', 'tags', 'tolerances', 'cheater_flags',
            'fitnesses')).explode(['tags', 'tolerances', 'cheater_flags', 'fitnesses'])

        output_df.to_csv(directory + simulation_name + '.csv', index=False)

    def mean_donation_rate(self, cutoff):
        return np.mean(self.donation_rates[cutoff:]), np.std(self.donation_rates[cutoff:])

    def plot_donation_rate(self):
        fig, axs = plt.subplots(2)
        axs[0].plot(range(self.generations), self.donation_rates, label=r'Donation Rate')
        axs[0].set_ylim((0, 1))
        axs[0].grid(True)
        axs[1].plot(range(self.generations), np.count_nonzero(self.cheater_flags, axis=1) / self.N,
                    label=r'Cheater Ratio')
        axs[1].grid(True)
        axs[1].set_ylim((0, 1))
        axs[0].legend()
        axs[1].legend()
        plt.show()


class Statistics:
    def __init__(self, number_of_runs, N, cost, benefit, pairings, mutation_rate, min_tolerance, cheater_mutation_rate,
                 n_neighbors, generations, mu, sigma, ):
        self.n_runs = number_of_runs
        self.N = N
        self.cost = cost
        self.benefit = benefit
        self.pairings = pairings
        self.mutation_rate = mutation_rate
        self.min_tolerance = min_tolerance
        self.cheater_mutation_rate = cheater_mutation_rate
        self.n_neighbors = n_neighbors
        self.generations = generations
        self.mu = mu
        self.sigma = sigma
        self.output = []
        self.models = [
            Model(N, cost, benefit, pairings, mutation_rate, min_tolerance, cheater_mutation_rate, n_neighbors,
                  generations, mu,
                  sigma) for _ in range(number_of_runs)]
        self.tags = np.zeros((generations, N))
        self.tolerances = np.zeros((generations, N))
        self.cheater_tags = np.zeros(N, dtype=bool)

    def simulate(self, n_processes=-2):
        """
                Runs the simulations in parallel to speed up thinks a little bit.

                Parameters
                ----------
                n_proceses: int Number of processes to be started simultaneously Default: Use all but one
        """

        def run_sim(model):
            model.simulate()
            return model

        self.models = Parallel(n_jobs=n_processes)(
            delayed(run_sim)(model) for model in tqdm(self.models, leave=False))

    def save(self):
        for model in self.models:
            output_df = pd.DataFrame(model.output, columns=(
                'generation', 'interactions_attempted', 'interactions_made', 'tags', 'tolerances', 'cheater_flags',
                'fitnesses')).explode(['tags', 'tolerances', 'cheater_flags', 'fitnesses'])

            output_df.to_csv(directory + simulation_name + '.csv', index=False, mode='a')

    def mean_donation_rate(self, last_n_generations):
        mean = 0
        for model in self.models:
            mean += model.donation_rates[-last_n_generations:]
        return mean

def parameter_sweep_pairings(N: int, cost: float, benefit: float, max_pairings: int, mutation_rate: float,
                             n_neighbors: int, generations: int, cutoff: int, min_tolrance=-10E-6, mu=0, sigma=0.01):
    pairings = np.arange(1, max_pairings+1)
    def run(p):
        model = Model(N=N,
                      cost=cost,
                      benefit=benefit,
                      pairings=p,
                      mutation_rate=mutation_rate,
                      min_tolerance=min_tolrance,
                      cheater_mutation_rate=1,  # social parasite type, no changes made
                      n_neighbors=n_neighbors,  # neighbor radius = neighbors / 2, max is n - 1
                      generations=generations,
                      mu=mu,
                      sigma=sigma)
        model.simulate()
        return model.donation_rates[cutoff:]

    donation_rates_time_series = np.array(Parallel(n_jobs=-1)(
            delayed(run)(p) for p in pairings)).transpose()
    print(np.array(donation_rates_time_series).transpose()[0:5].shape)
    #mean_donation_rates[p-1], stdvs_donation_rate[p-1] = model.mean_donation_rate(cutoff)
    mean_donation_rates = np.mean(donation_rates_time_series, axis=0)
    assert len(mean_donation_rates) == len(pairings)
    stdvs_donation_rate = np.std(donation_rates_time_series, axis=0)
    assert len(stdvs_donation_rate) == len(pairings)
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(pairings, mean_donation_rates, color='k')
    axs[0, 0].scatter(pairings, mean_donation_rates, color='k')
    #axs[0].plot(pairings, mean_donation_rates + stdvs_donation_rate, color='tab:blue', ls='--')
    #axs[0].plot(pairings, mean_donation_rates - stdvs_donation_rate, color='tab:blue', ls='--')
    #axs[0].set_ylim((0, .4))
    axs[0, 0].grid(True)
    axs[0, 0].set_ylabel('Mean Donation Rate')
    axs[0, 0].set_title(f'N={N}, c={cost}, b = {benefit}, m={mutation_rate},R={int(0.5*n_neighbors)} T_min={min_tolrance}, g={generations}, cutoff={cutoff}',
                 fontsize=10)
    axs[1, 0].plot(pairings, stdvs_donation_rate, color='k')
    axs[1, 0].scatter(pairings, stdvs_donation_rate, color='k')
    axs[1, 0].set_xlabel('Pairings')
    axs[1, 0].set_ylabel('StDv of Mean Donation Rate')
    axs[1, 0].grid(True)
    jump1 = [np.array(donation_rates_time_series).transpose()[i] for i in [8,9,10]]
    axs[0, 1].violinplot(jump1)
    jump2 = [np.array(donation_rates_time_series).transpose()[i] for i in [18, 19, 20]]
    axs[0, 2].violinplot(jump2)
    plt.show()



if __name__ == "__main__":
    pairings = np.arange(1, 15)
    parameter_sweep_pairings(N=90, cost=0.1, benefit=1, max_pairings=25, mutation_rate=0.01, n_neighbors=2,
                             generations=8_0, cutoff=1_0)


    model = Model(N=90,
                      cost=0.1,
                      benefit=1,
                      pairings=12,
                      mutation_rate=0.01,
                      min_tolerance=-10E-6,
                      cheater_mutation_rate=1,  # social parasite type, no changes made
                      n_neighbors=2,  # neighbor radius = neighbors / 2, max is n - 1
                      generations=8_000,
                      mu=0,
                      sigma=0.01)

    directory = './'
    simulation_name = f'test'

    model.simulate()
    model.plot_donation_rate()
    #model.save(simulation_name, directory)
