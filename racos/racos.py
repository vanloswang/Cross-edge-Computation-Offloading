"""
This module defines the class Racos.

Author:
    Hailiang Zhao (Consult works done by Yi-Qi Hu from Nanjing University)
"""
from racos.instance import Instance
from cross_edge_offloading.utils.tool_function import ToolFunction


class Racos(object):
    """
    This class defines every properties and methods used in RACOS algorithm.
    """

    def __init__(self, dimension, sample_size=15, iterations_num=30, budget=2000, pos_num=1, rand_prob=0.99,
                 uncertain_bits_num=1, online=False):
        """
        Initialization.

        :param dimension: the instance of class Dimension
        :param sample_size: the number of sampling in one iteration
        :param iterations_num: the number of iterations
        :param budget: the budget of evaluation (only applicable when sequential/online mode is turned on)
        :param pos_num: the number of positive solutions
        :param rand_prob: the probability of random sampling
        :param uncertain_bits_num: the number of uncertain bits (the size of dimensions sampled randomly)
        :param online: whether Sequential RACOS (SRACOS) is used, False in default
        """
        self.__dimension = dimension
        self.__sample_size = sample_size
        self.__pos_num = pos_num
        self.__rand_prob = rand_prob
        self.__uncertain_bits_num = uncertain_bits_num

        self.__population = []                               # the population set of feasible solutions
        self.__pos_population = []                           # the positive population set of feasible solutions
        self.__optimal_solution = []                         # the best-so-far optimal solutions
        self.__next_population = []                          # the population set for next iteration

        self.__online = online
        if online is False:
            self.__iterations_num = iterations_num
            self.__budget = 0
        else:
            self.__budget = budget
            self.__iterations_num = 0

        # set regions and labels, which has the same shape with dimension
        self.__regions = dimension.get_regions()
        # labels define whether self.__dimension[i] can be randomly sampled
        self.__labels = [True] * dimension.get_dim_size()

    def turn_on_online(self):
        self.__online = True

    def turn_off_online(self):
        self.__online = False

    def clear(self):
        """
        Clear populations and best-so-far optimal solution information.

        :return: no return
        """
        self.__population = []
        self.__pos_population = []
        self.__optimal_solution = []
        self.__next_population = []

    def generate_random_ins(self):
        """
        Generate a random instance for those dimensions whose labels are True.

        :return: the generated instance
        """
        instance = Instance(self.__dimension)
        for i in range(self.__dimension.get_dim_size()):
            instance.set_feature(i, ToolFunction.sample_uniform_integer(self.__regions[i][0], self.__regions[i][1]))
        return instance

    def generate_from_pos_ins(self, pos_instance):
        """
        Generate an instance from an exist positive instance.

        :param pos_instance: the exist positive instance
        :return: the generated instance
        """
        instance = Instance(self.__dimension)
        for i in range(self.__dimension.get_dim_size()):
            if not self.__labels[i]:
                instance.set_feature(i, ToolFunction.sample_uniform_integer(self.__regions[i][0], self.__regions[i][1]))
            else:
                instance.set_feature(i, pos_instance.get_feature(i))
        return instance

    def reset(self):
        """
        Reset all labels as True.

        :return: no return
        """
        # self.__regions will not be changed
        self.__labels = [True] * self.__dimension.get_dim_size()

    @staticmethod
    def is_ins_in_list(instance, ins_list, end):
        """
        Check whether the instance is stored in the ins_list.

        :param instance: the need-to-be-checked instance
        :param ins_list: the list of instances
        :param end: the end index of the list
        :return: checking result (True or False)
        """
        for i in range(len(ins_list)):
            if i == end:
                break
            if instance.is_equal(ins_list[i]):
                return True
        return False

    def generate_solutions(self, optimization_func):
        """
        Generate 'population' and 'pos_population' of feasible solutions, then store the best-so-far solution
        who has the minimum optimization function.

        :param optimization_func: the optimization function
        :return: no return
        """
        # clear the changes on labels
        self.reset()

        solutions = []
        # generate instances which are each distinct
        for i in range(self.__sample_size + self.__pos_num):
            instance = self.generate_random_ins()
            while True:
                if not Racos.is_ins_in_list(instance, solutions, i):
                    break
                else:
                    instance = self.generate_random_ins()
            # calculate fitness
            instance.set_fitness(optimization_func(instance))
            solutions.append(instance)

        solutions.sort(key=lambda ins: ins.get_fitness())

        # store the best ones in pos_population
        index = 0
        while index < self.__pos_num:
            self.__pos_population.append(solutions[index])
            index += 1
        while index < (self.__pos_num + self.__sample_size):
            self.__population.append(solutions[index])
            index += 1

        # store the best-so-far optimal solution
        self.__optimal_solution = solutions[0].deep_copy()

    def is_distinguish(self, ins, chosen_dim):
        """
        Judge whether all instances in population set are different from the chosen instance in those
        chosen dimensions.

        :param ins: the chosen instance
        :param chosen_dim: the chosen dimensions
        :return: judge result, True or False
        """
        if len(chosen_dim) == 0:
            return False
        for i in range(self.__sample_size):
            j = 0
            while j < len(chosen_dim):
                if ins.get_feature(chosen_dim[j]) != self.__population[i].get_feature(chosen_dim[j]):
                    break
                j += 1
            if j == len(chosen_dim):
                return False
        return True

    def shrink(self, instance):
        """
        Shrink the dimensions.

        :param instance: the chosen instance
        :return: the shrunken dimensions with size only being self.__uncertain_bits_num
        """
        non_chosen_dimension = [n for n in range(self.__dimension.get_dim_size())]
        chosen_dimension = []
        while not self.is_distinguish(instance, chosen_dimension):
            tmp = non_chosen_dimension[ToolFunction.sample_uniform_integer(0, len(non_chosen_dimension) - 1)]
            chosen_dimension.append(tmp)
            non_chosen_dimension.remove(tmp)
        while len(non_chosen_dimension) > self.__uncertain_bits_num:
            tmp = non_chosen_dimension[ToolFunction.sample_uniform_integer(0, len(non_chosen_dimension) - 1)]
            chosen_dimension.append(tmp)
            non_chosen_dimension.remove(tmp)
        return non_chosen_dimension

    def update_labels(self):
        """
        Update self.__labels according to the number of uncertain bits.

        :return: no return
        """
        dims = [n for n in range(self.__dimension.get_dim_size())]
        for i in range(self.__uncertain_bits_num):
            index = ToolFunction.sample_uniform_integer(0, self.__dimension.get_dim_size() - i - 1)
            self.__labels[dims[index]] = False
            dims.remove(dims[index])

    def update_pos_population(self):
        """
        Put those better solutions in population into appropriate position of pos_population, and then filter those
        bad solutions out (keep solutions in both sets in order).

        :return: no return
        """
        for i in range(self.__sample_size):
            j = 0
            while j < self.__pos_num:
                if self.__population[i].get_fitness() < self.__pos_population[j].get_fitness():
                    break
                else:
                    j += 1
            if j < self.__pos_num:
                # the populations need to be update
                tmp = self.__population[i]
                self.__population[i] = self.__pos_population[self.__pos_num - 1]
                idx = self.__pos_num - 1
                # insert tmp into appropriate position while keeping the set in order
                while idx > j:
                    self.__pos_population[idx] = self.__pos_population[idx - 1]
                    idx -= 1
                self.__pos_population[j] = tmp

    def update_optimal(self):
        """
        Compare self.__optimal_solution with the first solution in pos_population. Update it if possible.

        :return: no return
        """
        if self.__pos_population[0].get_fitness() < self.__optimal_solution.get_fitness():
            self.__optimal_solution = self.__pos_population[0].deep_copy()

    def check_illegal(self):
        """
        Check whether there exists illegal solution in population set, i.e., solutions whose dimensions out of regions.

        :return: True or False
        """
        for i in range(self.__sample_size):
            j = 0
            while j < self.__dimension.get_dim_size():
                if not (self.get_region(j)[0] < self.__population[i].get_feature(j) < self.get_region(j)[1]):
                    break
                else:
                    j += 1
            if j == self.__dimension.get_dim_size():
                return False
        return True

    def racos(self, optimization_func):
        """
        The optimization algorithm, RACOS.

        :return: no return
        """
        self.clear()
        self.generate_solutions(optimization_func)
        # if sequential is not used
        if not self.__online:
            # begin iteration
            for it in range(self.__iterations_num):
                print('RACOS iteration #', it)
                self.__next_population = []
                for i in range(self.__sample_size):
                    while True:
                        self.reset()
                        chosen_pos_idx = ToolFunction.sample_uniform_integer(0, self.__pos_num - 1)
                        eps = ToolFunction.sample_uniform_double(0, 1)
                        if eps <= self.__rand_prob:
                            non_chosen_dim = self.shrink(self.__pos_population[chosen_pos_idx])
                            for j in range(len(non_chosen_dim)):
                                self.__labels[non_chosen_dim[j]] = False
                        ins = self.generate_from_pos_ins(self.__pos_population[chosen_pos_idx])
                        if (not Racos.is_ins_in_list(ins, self.__pos_population, self.__pos_num)) \
                                and (not Racos.is_ins_in_list(ins, self.__next_population, i)):
                            ins.set_fitness(optimization_func(ins))
                            break
                    self.__next_population.append(ins)
                self.__population = []
                for i in range(self.__sample_size):
                    self.__population.append(self.__next_population[i])
                self.update_pos_population()
                self.update_optimal()
                # print the best-so-far fitness for test
                print('Best-so-far fitness:', self.get_optimal_solution().get_fitness())
        else:
            # sequential mode is not interested in this paper
            pass

    def get_dimension(self):
        return self.__dimension

    def set_dimension(self, dimension):
        self.__dimension = dimension

    def get_sample_size(self):
        return self.__sample_size

    def set_sample_size(self, sample_size):
        self.__sample_size = sample_size

    def get_iterations_num(self):
        return self.__iterations_num

    def set_iterations_num(self, iterations_num):
        self.__iterations_num = iterations_num

    def get_budget(self):
        return self.__budget

    def set_budget(self, budget):
        self.__budget = budget

    def get_pos_num(self):
        return self.__pos_num

    def set_pos_num(self, pos_num):
        self.__pos_num = pos_num

    def get_rand_prob(self):
        return self.__rand_prob

    def set_rand_prob(self, rand_prob):
        self.__rand_prob = rand_prob

    def get_uncertain_bits_num(self):
        return self.__uncertain_bits_num

    def set_uncertain_bits_num(self, uncertain_bits_num):
        self.__uncertain_bits_num = uncertain_bits_num

    # below properties do not have setter methods
    def get_population(self):
        return self.__population

    def get_pos_population(self):
        return self.__pos_population

    def get_optimal_solution(self):
        return self.__optimal_solution

    def get_next_population(self):
        return self.__next_population

    def get_regions(self):
        return self.__regions

    def get_region(self, index):
        return self.__regions[index]

    def get_labels(self):
        return self.__labels

    def get_label(self, index):
        return self.__labels[index]

    def get_online(self):
        return self.__online
