import statistics as stats
import Reporter
import numpy as np
import random

from matplotlib import pyplot as plt
from datetime import datetime
from typing import *

class Individual:
    def __init__(self, N, alpha, order=None):
        self.N = N
        self.alpha = alpha

        if order is None:
            self.order = 2 + np.arange(N)
            np.random.shuffle(self.order)
        else:
            self.order = order


class TSP:
    def __init__(self, N, cost_matrix, max_elem):
        self.N = N - 1  # exclude 1 from the matrix
        self.cost_matrix = cost_matrix
        self.max_elem = max_elem

    def c(self, x, y):
        return self.cost_matrix[x - 1][y - 1]

    def fitness(self, ind: Individual) -> float:
        value = self.c(1, ind.order[0])
        for i in range(1, len(ind.order)):
            value += self.c(ind.order[i - 1], ind.order[i])

        return value

    def get_path(self, ind: Individual):
        return np.concatenate((np.array([1]), ind.order))


class TspEa:
    def __init__(self, tsp: TSP):
        self.lamda = 200
        self.mu = self.lamda * 2
        self.tsp = tsp
        self.k = 3
        self.iteration = 100
        self.init_alpha = 0.05  # probability of applying mutation
        # self.init_alpha = max(0.01, 0.1 + 0.02 * np.random.randn())
        self.population = self.initialize()
        self.objf = tsp.fitness

    def initialize(self) -> List[Individual]:
        return [Individual(self.tsp.N, self.init_alpha) for _ in range(self.lamda)]

    def one_iteration(self):
        offsprings: List[Individual] = list()

        for j in range(self.mu):
            p1 = self.selection()
            p2 = self.selection()
            c = self.recombination(p1, p2)
            offsprings.append(c)

        # Apply mutation to offsprings
        map(self.mutation, offsprings)

        # Apply mutation to population
        map(self.mutation, self.population)

        # Elimination
        self.population = self.elimination(offsprings)

    def get_results(self):
        population_fitness = [self.tsp.fitness(x) for x in self.population]
        min_val = self.tsp.max_elem * (self.tsp.N + 1)
        min_elem = None
        for i in range(len(population_fitness)):
            if population_fitness[i] < min_val:
                min_val = population_fitness[i]
                min_elem = self.population[i]
        return stats.mean(population_fitness), self.tsp.fitness(min_elem), self.tsp.get_path(min_elem)

    def mutation(self, ind: Individual) -> Individual:
        if random.random() < ind.alpha:
            # for x in range(int(len(ind.order) * 0.1)):
            #     idx1 = random.randint(0, len(ind.order) - 1)
            #     idx2 = random.randint(0, len(ind.order) - 1)
            #     ind.order[idx1], ind.order[idx2] = ind.order[idx2], ind.order[idx1]

            idx1 = random.randint(0, len(ind.order) - 2)
            idx2 = random.randint(idx1 + 1, len(ind.order) - 1)
            np.random.shuffle(ind.order[idx1, idx2])

        return ind

    def selection(self) -> Individual:
        ri = random.choices(range(np.size(self.population, 0)), k=self.k)
        rif = np.array([self.tsp.fitness(self.population[e]) for e in ri])
        idx = np.argmin(rif)
        return self.population[ri[idx]]

    def elimination(self, offsprings: List[Individual]) -> List[Individual]:
        combined = np.concatenate((self.population, offsprings))
        # combined = offsprings
        l = list(combined)
        l.sort(key=lambda x: self.tsp.fitness(x))
        return l[:self.lamda]

    def recombination(self, ind3: Individual, ind4: Individual):

        size = len(ind3.order)

        beta = 2 * np.random.random() - 0.5
        alpha = ind3.alpha + beta * (ind4.alpha - ind3.alpha)

        # ind1 = Individual(ind3.N, alpha, np.array([0] * size))
        # ind2 = Individual(ind4.N, alpha, np.array([0] * size))

        ind1 = Individual(ind3.N, ind3.alpha, np.array([0] * size))

        csp1 = random.randint(0, size - 2)
        csp2 = random.randint(csp1 + 1, size)
        # csp1 = 3
        # csp2 = 6

        pos_ind3 = dict()
        pos_ind4 = dict()

        for i in range(len(ind3.order)):
            pos_ind3[ind3.order[i]] = i
            pos_ind4[ind4.order[i]] = i

        for i in range(csp1, csp2):
            ind1.order[i] = ind3.order[i]

        for i in range(csp1, csp2):

            pos_i = None
            t = ind4.order[i]

            if t in ind1.order:
                continue

            for _ in range(len(ind3.order)):
                v = ind3.order[pos_ind4[t]]
                pos_i = pos_ind4[v]
                if ind1.order[pos_i] == 0:
                    break
                t = ind4.order[pos_i]

            if pos_i is not None and ind1.order[pos_i] == 0:
                ind1.order[pos_i] = ind4.order[i]

        for i in range(0, len(ind1.order)):
            if ind1.order[i] == 0:
                ind1.order[i] = ind4.order[i]

        return ind1


class r0896104:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iterations = 100
        self.max_int = 1000000000.0
        self.theta = 1

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)

        distance_matrix = np.loadtxt(file, delimiter=",")
        # print(distance_matrix[0])
        max_elem = self.max_int / np.shape(distance_matrix)[0]
        distance_matrix[distance_matrix <= 0.0] = max_elem
        distance_matrix[distance_matrix == np.inf] = max_elem
        # print(distanceMatrix[0])
        file.close()

        # Your code here.
        N = np.shape(distance_matrix)[0]
        tsp = TSP(N, distance_matrix, max_elem)
        tsp_ea = TspEa(tsp)

        meanObjective, bestObjective, bestSolution = tsp_ea.get_results()
        print("Iteration=", -1,
              "population size= ", len(tsp_ea.population),
              "Mean fitness= ", meanObjective,
              # "variance fitness= ", stats.variance(population_fitness),
              "Best fitness= ", bestObjective,
              # "Best Solution= ", bestSolution
              # "Best KS= ", self.population[0].order,
              )

        meanObjectives = []
        bestObjectives = []

        tick = datetime.now()

        # while( yourConvergenceTestsHere ):
        for i in range(self.iterations):
            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            if abs(meanObjective - bestObjective) < self.theta:
                break

            tsp_ea.one_iteration()
            meanObjective, bestObjective, bestSolution = tsp_ea.get_results()

            meanObjectives.append(meanObjective)
            bestObjectives.append(bestObjective)

            print("Iteration=", i,
                  "Mean fitness= ", meanObjective,
                  # "variance fitness= ", stats.variance(population_fitness),
                  "Best fitness= ", bestObjective,
                  # "Best Solution= ", bestSolution
                  # "Best KS= ", self.population[0].order,
                  )
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        tock = datetime.now()
        print("duration: ", (tock - tick).microseconds / 1000, "ms")

        # Your code here.
        return meanObjectives, bestObjectives


if __name__ == '__main__':
    best_obj_in_runs = []

    for i in range(1):
        f = r0896104()
        mean_objs, best_objs = f.optimize("tours/tour29.csv")
        x = list(np.arange(len(mean_objs)))
        plt.plot(x, mean_objs, label="Mean objective")
        plt.plot(x, best_objs, label="Best objective")
        plt.xlabel("Iterations")
        plt.ylabel("Cycle length")
        plt.legend()
        plt.show()

        best_obj = best_objs[len(best_objs) - 1]
        best_obj_in_runs.append(best_obj)

    # x = list(np.arange(len(best_obj_in_runs)))
    # plt.plot(x, best_obj_in_runs, label="Best objective")
    # # plt.plot(x, best_objs, label="Best objective")
    # plt.xlabel("Runs")
    # plt.ylabel("Cycle length")
    # plt.legend()
    # plt.show()
