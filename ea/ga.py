import random
import math
import numpy as np
from time import sleep


def roulette(population, fitness):
    min_fitness = min(fitness)
    population_fitness = sum(fitness) - min_fitness * len(population)
    pr = list(map(lambda x: (x - min_fitness) / population_fitness, fitness))
    cdf = list()
    cum_pr = 0
    for specimen_pr in pr:
        cum_pr += specimen_pr
        cdf.append(cum_pr)
    successors = list()
    for i in range(len(population)):
        winner = random.choices(population, cum_weights=cdf)[0]
        successors.append(winner)
    return successors


def create_crossover(operator, p_cross=0.7):
    def crossover(population):
        random.shuffle(population)
        for i in range(0, len(population) - 1, 2):
            rand = random.random()
            if rand <= p_cross:
                p1 = population[i]
                p2 = population[i + 1]
                o1, o2 = operator(p1.copy(), p2.copy())
                population[i] = o1
                population[i] = o2
        return population

    return crossover


def create_mutator(operator, p_mutate=0.02):
    def mutator(population):
        for i in range(len(population)):
            mutant = operator(population[i].copy(), p_mutate)
            population[i] = mutant
        return population

    return mutator


def log_statistics(population, it, buffer, on_data):
    size = len(population)
    max_fitness = max(population)
    min_fitness = min(population)
    sum_fitness = sum(population)
    mean_fitness = sum_fitness / size
    var_fitness = sum(map(lambda x: (x - mean_fitness) ** 2, population)) / size
    sd_fitness = math.sqrt(var_fitness)

    buffer[0].append(min_fitness)
    buffer[1].append(mean_fitness)
    buffer[2].append(max_fitness)
    buffer[3].append(sd_fitness)
    buffer[4] = population
    buffer[5] = it
    if it % 10 == 0:
        on_data(buffer)
        buffer = list([[], [], [], [], [], 0])
        print(
            "Fitness: max={}, sum={}, mean={}, min={}, var={}, sd={}, It:{}".format(
                max_fitness,
                sum_fitness,
                mean_fitness,
                min_fitness,
                var_fitness,
                sd_fitness,
                it,
            )
        )
        sleep(0.1)

    return buffer


def noop():
    pass


def genetic_algorithm(
    initial_population,
    fitness,
    reproduction,
    operators,
    on_data=noop,
    max_iterations=10000,
    max_nochange=100,
):
    it = 0
    nochange = 0
    best = None
    population = initial_population.copy()
    buffer = list([[], [], [], [], [], 0])
    while it < max_iterations and nochange < max_nochange:
        specimen_fitness = list(map(fitness, population))
        pop_best = np.argmax(specimen_fitness)
        if best is None or best < specimen_fitness[pop_best]:
            best = specimen_fitness[pop_best]
            best_specimen = population[pop_best]
            nochange = 0
        nochange += 1
        buffer = log_statistics(specimen_fitness, it, buffer, on_data)
        T = reproduction(population.copy(), specimen_fitness)
        for operator in operators:
            O = operator(T.copy())
        population = O
        it += 1
    return best_specimen
