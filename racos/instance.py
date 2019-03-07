"""
This module defines the class Instance, which describes feasible solutions.

Author:
    Hailiang Zhao (Consult works done by Yi-Qi Hu from Nanjing University)
"""


class Instance(object):
    """
    This class describes the solution of the concerned discrete optimization problem.
    """

    def __init__(self, dimension):
        """
        Initialize a solution with [0,0,...,0].

        :param dimension: the instance of class Dimension
        """
        self.__dimension = dimension
        # the value of each dimension
        self.__features = [0] * dimension.get_dim_size()
        # the value of optimization function
        self.__fitness = 0

    def get_dimension(self):
        return self.__dimension

    def set_dimension(self, dimension):
        self.__dimension = dimension
        self.__features = [0] * dimension.get_dim_size()
        self.__fitness = 0

    def get_feature(self, index):
        return self.__features[index]

    def set_feature(self, index, feature):
        self.__features[index] = feature

    def get_features(self):
        return self.__features

    def set_features(self, features):
        self.__features = features

    # fitness is calculated by optimization function
    def get_fitness(self):
        return self.__fitness

    # set_fitness will only be called when deep copy an instance
    def set_fitness(self, fitness):
        self.__fitness = fitness

    def is_equal(self, instance):
        """
        Judge whether the instance itself is equal to another instance.

        :param instance: the compared instance
        :return: judge result, True or False
        """
        if len(self.__features) != len(instance.get_features()):
            return False
        for i in range(len(self.__features)):
            if self.__features[i] != instance.get_feature(i):
                return False
        return True

    def deep_copy(self):
        """
        Deep copy a new instance of the instance itself.

        :return: the duplicated instance
        """
        copy = Instance(self.__dimension)
        # copy.set_features(self.__features)
        for i in range(len(self.__features)):
            copy.set_feature(i, self.__features[i])
        copy.set_fitness(self.__fitness)
        return copy
