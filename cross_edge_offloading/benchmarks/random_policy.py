"""
This module defines the class RandomPolicy, which is the first benchmark (RS) proposed in this paper.
It inherits from class OffloadingCommon.

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.offloading_common import OffloadingCommon


class RandomPolicy(OffloadingCommon):
    """
    This class implements the benchmark named RS.
    """
    def __init__(self, parameter):
        super().__init__(parameter)
