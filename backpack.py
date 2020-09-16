from ea.ga import genetic_algorithm, create_crossover, create_mutator, roulette
from ea.utils import PlotterMaster
import numpy as np
import random


def create_population(size, n):
    population = list()
    for i in range(size):
        specimen = list()
        for i in range(n):
            gene = 1 if random.random() < 0.2 else 0
            specimen.append(gene)
        population.append(specimen)
    return population


n = 50
pop_size = 100
max_weight = 13
weights = np.random.rand(n)
utilities = np.random.rand(n)
population = create_population(pop_size, n)
K = 3  # utilities.max() / weights.min()


def fitness(specimen):
    size = len(specimen)
    weight = 0
    utility = 0
    for i in range(size):
        if specimen[i] == 1:
            weight += weights[i]
            utility += utilities[i]
    return utility - K * max(weight - max_weight, 0)


def mutate(specimen, p_mutate):
    size = len(specimen)
    for i in range(size):
        rand = random.random()
        gene = specimen[i]
        if rand <= p_mutate:
            specimen[i] = 0 if gene == 1 else 1
    return specimen


def cross(p1, p2):
    size = len(p1)
    cross_point = random.randint(1, size - 1)
    o1 = p1[:cross_point] + p2[cross_point:]
    o2 = p2[:cross_point] + p1[cross_point:]
    return o1.copy(), o2.copy()


master = PlotterMaster()
operators = [create_crossover(cross), create_mutator(mutate)]
best_backpack = genetic_algorithm(
    population, fitness, roulette, operators, on_data=master.update, max_nochange=600, max_iterations=5000
)
print("BEST: {}".format(fitness(best_backpack)))
master.join()
