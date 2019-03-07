"""
This module defines the class Dimension, which describes the features of solution by RACOS.

Author:
    Hailiang Zhao (Consult works done by Yi-Qi Hu from Nanjing University)
"""


class Dimension(object):
    """
    This class defines basic information of solution's dimension, including dimension size, the feasible region,
    and the type of each dimension.
    """

    def __init__(self, dim_size):
        """
        Initialization.
        regions and types are lists stored information of every dimension.

        :param dim_size: the size of dimension of feasible solutions
        """
        self.__dim_size = dim_size
        self.__regions = [[0, 1]] * self.__dim_size
        # False means 'discrete'
        self.__types = [False] * self.__dim_size

    def set_dim_size(self, s):
        self.__dim_size = s

    def get_dim_size(self):
        return self.__dim_size

    # regions and types do not have setters
    def get_regions(self):
        return self.__regions

    def get_region(self, index):
        return self.__regions[index]

    def get_types(self):
        return self.__types

    def get_type(self, index):
        return self.__types[index]
