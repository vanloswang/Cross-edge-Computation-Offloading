"""
This module defines the class GreedyCompute, which is the third benchmark (GSC2) proposed in this paper.
It inherits from class OffloadingCommon.

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.offloading_common import OffloadingCommon
from cross_edge_offloading.utils.tool_function import ToolFunction
import numpy as np
import random


class GreedyCompute(OffloadingCommon):
    """
    This class implements the benchmark named GSC2.
    """
    def __init__(self, parameter):
        super().__init__(parameter)

    def obtain_edge_selections(self):
        """
        Obtain the feasible solution with greedy policy on computation.

        :return: edge_selections, every row denotes a mobile device who has task request
        """
        parameter = self.get_parameter()
        # first initialize with zero
        edge_selections = []
        for i in range(parameter.get_user_num()):
            edge_selection = np.repeat(0, len(parameter.get_connectable_servers()[i]))
            edge_selections.append(edge_selection)

        # for every edge site, generate a random integer with [0, max_assign], and distribute connections to
        # connectable mobile devices
        for j in range(parameter.get_server_num()):
            assign_num = parameter.get_max_assign()
            connectable_user_num = len(parameter.get_connectable_users()[j])
            if assign_num >= connectable_user_num:
                # every mobile device in it can be chosen
                for i in range(connectable_user_num):
                    user_index = parameter.get_connectable_users()[j][i]
                    edge_index = list.index(parameter.get_connectable_servers()[user_index], j)
                    edge_selections[user_index][edge_index] = 1
            else:
                # randomly choose assign_num users to distribute j's computation capacity
                user_indices = random.sample(parameter.get_connectable_users()[j], assign_num)
                for i in range(len(user_indices)):
                    user_index = user_indices[i]
                    edge_index = list.index(parameter.get_connectable_servers()[user_index], j)
                    edge_selections[user_index][edge_index] = 1

        # set those mobile devices who do not have task request to [0, 0, ..., 0]
        # we can not delete them from the list because every row is the index of the corresponding mobile device
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 0:
                edge_selections[i] = np.zeros(len(edge_selections[i]))
            else:
                division = int(sum(edge_selections[i]))
                if division:
                    times = self.obtain_time_consumption(division, edge_selections[i],
                                                         parameter.get_connectable_gains()[i])
                    energys = ToolFunction.obtain_transmit_energy(division, edge_selections[i], parameter,
                                                                  parameter.get_connectable_gains()[i])
                    # satisfy the constraint
                    if times >= parameter.get_ddl() or energys > self.get_battery_energy_levels()[i]:
                        edge_selections[i] = np.zeros(len(edge_selections[i]))
        return edge_selections
