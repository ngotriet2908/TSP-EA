import statistics as stats
import Reporter
import numpy as np
import random
import time
import scipy.stats as sp_stats
import threading

from multiprocessing import Pool, Process
from matplotlib import pyplot as plt
from typing import *
from collections import deque


class Individual:
    def __init__(self, N, alpha, order=None):
        self.N = N
        self.alpha = alpha
        if order is None:
            self.order = 2 + np.arange(N)
            np.random.shuffle(self.order)
        else:
            self.order = order

    def __copy__(self):
        return Individual(self.N, self.alpha, self.order.copy())


class TSP:
    def __init__(self, N, cost_matrix, max_elem):
        print("N = " + str(N))
        self.N = N - 1  # exclude 1 from the matrix
        self.cost_matrix = cost_matrix
        self.max_elem = max_elem

    def c(self, x, y):
        return self.cost_matrix[x - 1][y - 1]

    def fitness(self, ind: Individual) -> float:
        value = self.c(1, ind.order[0])
        for i in range(1, len(ind.order)):
            value += self.c(ind.order[i - 1], ind.order[i])

        value += self.c(ind.order[len(ind.order) - 1], 1)
        return value

    def valid_path(self, ind: Individual):
        if len(ind.order) != self.N:
            return False

        values = [x for x in range(2, self.N + 1)]
        order_vals = list(ind.order)
        for x in values:
            if x not in order_vals:
                return False

        return True

    def fitness_arr(self, arr):
        value = self.c(1, arr[0])
        for i in range(1, len(arr)):
            value += self.c(arr[i - 1], arr[i])

        value += self.c(arr[len(arr) - 1], 1)
        return value

    def get_path(self, ind: Individual):
        return np.concatenate((np.array([1]), ind.order))


class TspEa:
    def __init__(self, tsp: TSP):
        # self.lamda = 50
        # self.lamda = 40
        self.multi_threading = True
        self.lamda = 30
        # self.lamda = 20

        # self.mu = self.lamda
        self.mu = self.lamda * 2

        self.tsp = tsp
        self.k = 2
        self.init_alpha = 0.01  # probability of applying mutation
        # self.objf = tsp.fitness
        # self.objf = lambda xx, pop=None, beta_init=0: self.shared_fitness_wrapper(tsp.fitness, xx, pop, beta_init)
        # self.objf = self.shared_fitness_wrapper
        self.true_fitness = tsp.fitness

        self.adaptive_mutation_step = 0.3

        # self.init_population_multiplier = 10
        self.init_population_multiplier = 10

        self.distance = self.hamming_distance
        # self.distance = self.kendal_tau_distance

        self.mutation = self.shuffle_mutation

        # self.initialize = self.nearest_neighbor_init
        self.initialize = self.nearest_neighbor_buffer_init

        self.selection = self.k_tour_selection
        # self.selection = self.shared_selection

        # self.elimination = self.normal_elimination
        self.elimination = self.shared_elimination

        # self.recombination = self.pmx_recombination
        self.recombination = self.order_crossover

        # self.local_search = self.two_opt_local_search_opt
        self.local_search = self.two_opt_local_search_opt_np
        # self.local_search = self.two_opt_first_local_search_opt
        # self.local_search = None

    def init_filtering(self, population: List[Individual]) -> List[Individual]:

        sum_dis = np.array([sum([self.distance(x, y) for y in population]) for x in population])
        print(sum_dis)
        idx_sort = np.argsort(sum_dis)
        print(idx_sort)
        idx_sort = np.flip(idx_sort)
        print(idx_sort)

        return [population[idx] for idx in idx_sort[:self.lamda]]

    def one_iteration(self, population):
        offsprings: List[Individual] = list()

        for j in range(self.mu//2):
            p1 = self.selection(population)
            p2 = self.selection(population)
            c = self.recombination(p1, p2)
            offsprings.append(c)
            c = self.recombination(p2, p1)
            offsprings.append(c)

        # Modify mutation rate base on fitness vs avg fitness
        self.adaptive_mutation_rate(population)

        # Apply mutation to offsprings
        for i in range(len(offsprings)):
            offsprings[i] = self.mutation(offsprings[i])

        # Apply mutation to population
        for i in range(len(population)):
            population[i] = self.mutation(population[i])

        if self.multi_threading:
            p = Pool(processes=2)

            pool_res = p.map(self.apply_local_search, [population[0: (len(population) // 2)],
                                            population[(len(population) // 2): len(population)]])
            population[0: (len(population) // 2)] = pool_res[0]
            population[(len(population) // 2): len(population)] = pool_res[1]
        else:
            self.apply_local_search(population)

        # Elimination
        # population = self.elimination(offsprings)
        # print("pool finish")
        population = self.elimination(list(np.concatenate((population, offsprings))))

        return population

    def apply_local_search(self, population: List[Individual]):
        for i in range(len(population)):
            if self.local_search is not None:
                population[i] = self.local_search(population[i])
        # print("finish")
        return population

    def adaptive_mutation_rate(self, population: List[Individual]):
        population_fitness = [self.tsp.fitness(x) for x in population]
        mean_fitness = stats.mean(population_fitness)
        for i in range(0, len(population)):
            if population_fitness[i] < mean_fitness:
                population[i].alpha -= self.adaptive_mutation_step
            else:
                population[i].alpha += self.adaptive_mutation_step

            population[i].alpha = max(self.init_alpha, population[i].alpha)

    def random_nearest_neigbor_path(self):

        order = []
        remain = [x for x in range(1, self.tsp.N + 2)]

        start = remain[random.randint(0, len(remain) - 1)]
        remain.remove(start)

        order.append(start)

        while len(remain) > 0:
            u = order[len(order) - 1]

            best = self.tsp.max_elem + 100
            best_v = None
            for v in remain:
                if self.tsp.c(u, v) < best:
                    best = self.tsp.c(u, v)
                    best_v = v

            order.append(best_v)
            remain.remove(best_v)

        start_i = None
        for i, x in enumerate(order):
            if x == 1:
                start_i = i
                break

        final_order = order[(start_i + 1): len(order)] + order[0: start_i]

        return Individual(len(final_order), self.init_alpha, np.array(final_order))

    def nearest_neighbor_init(self) -> List[Individual]:
        population = []
        for i in range(self.lamda):
            path = self.random_nearest_neigbor_path()
            # print("path " + str(i) + " is valid " + str(self.tsp.valid_path(path)))
            population.append(path)

        # for i in range(len(population)):
        #     if self.local_search is not None:
        #         population[i] = self.local_search(population[i])

        return population

    def nearest_neighbor_buffer_init(self) -> List[Individual]:
        population = []
        for i in range(self.lamda * self.init_population_multiplier):
            path = self.random_nearest_neigbor_path()
            # print("path " + str(i) + " is valid " + str(self.tsp.valid_path(path)))
            population.append(path)

        final_population = self.shared_elimination(population)
        # final_population = self.init_filtering(population)
        # final_population = population
        # for i in range(len(population)):
        #     if self.local_search is not None:
        #         population[i] = self.local_search(population[i])

        return final_population

    def normal_initialize(self) -> List[Individual]:
        population = [Individual(self.tsp.N, self.init_alpha) for _ in range(self.lamda)]

        # for i in range(len(population)):
        #     if self.local_search is not None:
        #         population[i] = self.local_search(population[i])

        return population

    def hamming_distance(self, ind1: Individual, ind2: Individual):
        return len(np.bitwise_xor(ind1.order, ind2.order).nonzero()[0])

    def kendal_tau_distance(self, ind1: Individual, ind2: Individual):
        return sp_stats.kendalltau(ind1.order, ind2.order)[0]

    def shared_fitness_wrapper(self, fun, X, pop=None, betaInit=0):
        if pop is None:
            return fun(X)

        alpha = 2.0
        sigma = self.tsp.N * 0.3
        # sigma = 1.0 * 0.3

        mod_objv = np.zeros(len(X))
        for i, x in enumerate(X):
            ds = np.array([self.distance(x, y) for y in pop])
            one_plus_beta = betaInit
            for d in ds:
                if d <= sigma:
                    one_plus_beta += 1 - (d / sigma) ** alpha
            fval = fun(x)
            mod_objv[i] = fval * one_plus_beta ** np.sign(fval)

        return mod_objv

    def shuffle_mutation(self, ind: Individual) -> Individual:
        if random.random() < ind.alpha:
            idx1 = random.randint(0, len(ind.order))
            idx2 = random.randint(0, len(ind.order))

            start = min(idx1, idx2)
            end = max(idx1, idx2)
            random.shuffle(ind.order[start:end])

            ind.alpha = self.init_alpha
            # np.random.shuffle(ind.order[idx1, idx2])

        return ind

    def k_tour_selection(self, population) -> Individual:
        ri = random.choices(range(np.size(population, 0)), k=self.k)
        rif = np.array([self.tsp.fitness(population[e]) for e in ri])
        idx = np.argmin(rif)
        return population[ri[idx]]

    def shared_selection(self, population) -> Individual:
        ri = random.choices(range(np.size(population, 0)), k=self.k)
        rif = self.shared_fitness_wrapper(self.tsp.fitness, [population[e] for e in ri], population)
        idx = np.argmin(rif)
        return population[ri[idx]]

    def normal_elimination(self, population: List[Individual]) -> List[Individual]:
        l = list(population)
        l.sort(key=lambda x: self.tsp.fitness(x))
        return l[:self.lamda]

    def shared_elimination(self, population: List[Individual]) -> List[Individual]:
        survivors: List[Individual] = []
        for i in range(self.lamda):
            fvals = self.shared_fitness_wrapper(self.tsp.fitness, population, survivors[0:i - 1], 1)
            idx = np.argmin(fvals)
            survivors.append(population[idx])

        return survivors

    def order_crossover(self, pa1: Individual, pa2: Individual):
        size = len(pa1.order)

        beta = abs(pa2.alpha - pa1.alpha)
        sign = random.randint(0, 1) * 1.0
        alpha = (pa2.alpha + pa1.alpha) / 2 + sign * beta * random.random()

        child = Individual(pa1.N, alpha, np.array([0] * size))

        rnd1 = random.randint(0, size - 1)
        rnd2 = random.randint(0, size)

        start = min(rnd1, rnd2)
        end = max(rnd1, rnd2)

        tmp = pa1.order[start:end].copy()
        child.order[start:end] = tmp

        inter = tmp
        remain = []
        for pa2x in pa2.order:
            if pa2x not in inter:
                remain.append(pa2x)

        for i in range(0, size):
            if child.order[i] == 0:
                child.order[i] = remain.pop(0)

        return child

    def pmx_recombination(self, ind3: Individual, ind4: Individual):

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

    def two_opt_local_search(self, ind: Individual) -> Individual:
        best_arr = None
        original = np.ndarray.copy(ind.order)
        best_fitness = self.tsp.fitness(ind)
        # print("init: " + str(best_fitness))

        tmp = np.ndarray.copy(original)
        for i in range(0, len(ind.order) - 1):
            for j in range(i + 1, len(ind.order)):
                tmp[i], tmp[j] = tmp[j], tmp[i]
                tmp_fit = self.tsp.fitness_arr(tmp)
                if tmp_fit < best_fitness:
                    best_arr = np.ndarray.copy(tmp)
                    best_fitness = tmp_fit
                tmp[i], tmp[j] = tmp[j], tmp[i]

        if best_arr is not None:
            ind.order = best_arr
            # print("after: " + str(best_fitness))
            # print("After: " + str(self.tsp.fitness(ind)))

        return ind

    def two_opt_local_search_opt(self, ind: Individual) -> Individual:
        best_ij = None
        original = np.ndarray.copy(ind.order)
        original_fit = self.tsp.fitness_arr(original)

        best_fitness = self.tsp.fitness(ind)
        # print("init: " + str(best_fitness))

        for i in range(1, len(ind.order) - 2):
            for j in range(i + 1, len(ind.order) - 1):

                tmp_fit = original_fit \
                          - self.tsp.c(ind.order[i - 1], ind.order[i]) \
                          - self.tsp.c(ind.order[i], ind.order[i + 1]) \
                          - self.tsp.c(ind.order[j - 1], ind.order[j]) \
                          - self.tsp.c(ind.order[j], ind.order[j + 1]) \
                          + self.tsp.c(ind.order[i - 1], ind.order[j]) \
                          + self.tsp.c(ind.order[j], ind.order[i + 1]) \
                          + self.tsp.c(ind.order[j - 1], ind.order[i]) \
                          + self.tsp.c(ind.order[i], ind.order[j + 1])

                if tmp_fit < best_fitness:
                    best_ij = (i, j)
                    best_fitness = tmp_fit

        if best_ij is not None:
            ind.order[best_ij[0]], ind.order[best_ij[1]] = ind.order[best_ij[1]], ind.order[best_ij[0]]
            # print("after: " + str(best_fitness))
            # print("After: " + str(self.tsp.fitness(ind)))

        return ind

    def two_opt_local_search_opt_np(self, ind: Individual) -> Individual:
        best_ij = None
        original_fit = self.tsp.fitness(ind)

        best_fitness = original_fit

        for i in range(1, len(ind.order) - 2):
            tmp1_fit = original_fit \
                       - self.tsp.c(ind.order[i - 1], ind.order[i]) \
                       - self.tsp.c(ind.order[i], ind.order[i + 1])

            for j in range(i + 1, len(ind.order) - 1):

                tmp_fit = tmp1_fit \
                          - self.tsp.c(ind.order[j - 1], ind.order[j]) \
                          - self.tsp.c(ind.order[j], ind.order[j + 1]) \
                          + self.tsp.c(ind.order[i - 1], ind.order[j]) \
                          + self.tsp.c(ind.order[j], ind.order[i + 1]) \
                          + self.tsp.c(ind.order[j - 1], ind.order[i]) \
                          + self.tsp.c(ind.order[i], ind.order[j + 1])

                if tmp_fit < best_fitness:
                    best_ij = (i, j)
                    best_fitness = tmp_fit

        if best_ij is not None:
            ind.order[best_ij[0]], ind.order[best_ij[1]] = ind.order[best_ij[1]], ind.order[best_ij[0]]
            # print("after: " + str(best_fitness))
            # print("After: " + str(self.tsp.fitness(ind)))

        return ind

    def two_opt_first_local_search_opt(self, ind: Individual) -> Individual:
        original = np.ndarray.copy(ind.order)
        original_fit = self.tsp.fitness_arr(original)

        best_fitness = self.tsp.fitness(ind)
        # print("init: " + str(best_fitness))

        ii = [i for i in range(1, len(ind.order) - 2)]

        random.shuffle(ii)

        for i in ii:
            for j in range(i + 1, len(ind.order) - 1):
                tmp_fit = original_fit \
                          - self.tsp.c(ind.order[i - 1], ind.order[i]) \
                          - self.tsp.c(ind.order[i], ind.order[i + 1]) \
                          - self.tsp.c(ind.order[j - 1], ind.order[j]) \
                          - self.tsp.c(ind.order[j], ind.order[j + 1]) \
                          + self.tsp.c(ind.order[i - 1], ind.order[j]) \
                          + self.tsp.c(ind.order[j], ind.order[i + 1]) \
                          + self.tsp.c(ind.order[j - 1], ind.order[i]) \
                          + self.tsp.c(ind.order[i], ind.order[j + 1])

                if tmp_fit < best_fitness:
                    ind.order[i], ind.order[j] = ind.order[j], ind.order[i]
                    return ind

        return ind

    def get_results(self, population):
        population_fitness = [self.tsp.fitness(x) for x in population]
        min_val = self.tsp.max_elem * (self.tsp.N + 1)
        min_elem = None
        for i in range(len(population_fitness)):
            if population_fitness[i] < min_val:
                min_val = population_fitness[i]
                min_elem = population[i]
        return stats.mean(population_fitness), self.tsp.fitness(min_elem), self.tsp.get_path(min_elem)


class r0896104:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.iterations = 1000
        self.max_int = 1000000000.0
        self.theta = 0.01
        self.window_size = 50

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)

        distance_matrix = np.loadtxt(file, delimiter=",")
        max_elem = self.max_int / np.shape(distance_matrix)[0]
        distance_matrix[distance_matrix <= 0.0] = max_elem
        distance_matrix[distance_matrix == np.inf] = max_elem
        file.close()

        # Your code here.
        N = np.shape(distance_matrix)[0]
        tsp = TSP(N, distance_matrix, max_elem)
        tsp_ea = TspEa(tsp)

        meanObjectives = []
        bestObjectives = []

        total_time = 0.0

        best_obj_window = deque(maxlen=self.window_size)

        init_start = time.time()
        population = tsp_ea.initialize()
        print(f'Init time: {time.time() - init_start: .2f}s')

        # while( yourConvergenceTestsHere ):
        for i in range(self.iterations):
            iteration_time = time.time()
            # Your code here.

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            population = tsp_ea.one_iteration(population)

            iteration_duration = time.time() - iteration_time
            total_time += iteration_duration

            meanObjective, bestObjective, bestSolution = tsp_ea.get_results(population)

            meanObjectives.append(meanObjective)
            bestObjectives.append(bestObjective)
            best_obj_window.append(bestObjective)

            print(
                f'{iteration_duration: .2f}s:\t Iteration = {i: .0f} \t Mean fitness = {meanObjective: .5f} \t Best fitness = {bestObjective: .5f}')
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            if len(best_obj_window) >= best_obj_window.maxlen and stats.stdev(best_obj_window) < self.theta:
                break

            if timeLeft < 0:
                break

        print(f'Duration: {total_time: .2f}s')
        return meanObjectives, bestObjectives


if __name__ == '__main__':
    best_obj_in_runs = []

    for _ in range(1):
        f = r0896104()
        mean_objs, best_objs = f.optimize("tours/tour250.csv")
        x_axis = list(np.arange(len(mean_objs)))
        plt.plot(x_axis, mean_objs, label="Mean objective")
        plt.plot(x_axis, best_objs, label="Best objective")
        plt.xlabel("Iterations")
        plt.ylabel("Cycle length")
        plt.legend()
        plt.show()

        best_obj = best_objs[len(best_objs) - 1]
        best_obj_in_runs.append(best_obj)

    # tsp = TSP(10, np.random.rand(10, 10), 10000)
    # tspEA = TspEa(tsp)
    #
    # ind1 = Individual(tsp.N + 1, 0, np.array([8, 4, 7, 3, 6, 2, 5, 1, 9, 0]))
    # ind2 = Individual(tsp.N + 1, 0, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    # ind3 = tspEA.order_crossover(ind1, ind2)
