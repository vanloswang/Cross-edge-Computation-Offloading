"""
This module contains the class OffloadingCommon, which is the base class of all algorithms (benchmarks, cco and decor).
OffloadingCommon defines several points in a computation offloading problem.

[--
In order to avoid Multiple Inheritance, CcoAlgorithm only inherit from Racos. Similar methods and properties are
copied from OffloadingCommon, which are marked by annotations.
--]

Author:
    Hailiang Zhao, Cheng Zhang
"""
from cross_edge_offloading.utils.tool_function import ToolFunction
import numpy as np
import random


class OffloadingCommon(object):
    """
    This class contains several points in a computation offloading problem, including:
    (1) the objective function of the cross-edge computation offloading problem;
    (2) the solution of the problem (edge_selection, harvested_energys).
    """

    def __init__(self, parameter):
        """
        Initialize key parameters in offloading problems of one time slot.

        :param parameter: the instance of class Parameter
        """
        self.__parameter = parameter

        # =============================== state information ===============================
        self.__battery_energy_levels = np.repeat(parameter.get_perturbation_para() / 2, parameter.get_user_num())
        self.__virtual_energy_levels = self.__battery_energy_levels - \
            np.repeat(parameter.get_perturbation_para(), parameter.get_user_num())

        # =============================== independent variables ===============================
        # edge_selections is a list with every element (edge_selection) being a numpy array,
        # which is the feasible solution (independent var) of the problem $\mathcal{P}_2^{es}$
        # 'self.edge_selections' stores the final optimal solution
        self.__edge_selections = []
        self.__harvested_energys = []

    def obtain_time_consumption(self, division, edge_selection, channel_power_gains):
        """
        Calculate the time consumption on transmission and edge execution for one mobile device.

        :param division: the number of chosen edge sites (not zero)
        :param edge_selection: the edge selection decision of one mobile devices
        :param channel_power_gains: the channel power gains of one mobile devices to every connectable servers
        :return: the time consumption on transmission and edge execution
        """
        parameter = self.get_parameter()
        transmit_times = ToolFunction.obtain_transmit_times(division, edge_selection, parameter, channel_power_gains)
        edge_exe_times = ToolFunction.obtain_edge_exe_times(division, parameter)
        edge_times = transmit_times + edge_exe_times
        time_consumption = max(edge_times) + parameter.get_local_exe_time() + parameter.get_coordinate_cost() * division
        return time_consumption

    def obtain_overall_costs(self, edge_selections):
        """
        Calculate the overall costs, which is the sum of cost of each mobile device.

        :param edge_selections: the edge selection decisions for all mobile devices
        :return: overall costs
        """
        parameter = self.get_parameter()
        overall_costs = 0
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 1:
                division = int(sum(edge_selections[i]))
                if division:
                    # cost = self.obtain_time_consumption(
                    #     division, edge_selections[i], parameter.get_connectable_gains[i])
                    transmit_times = ToolFunction.obtain_transmit_times(division, edge_selections[i], parameter,
                                                                        parameter.get_connectable_gains()[i])
                    edge_exe_times = ToolFunction.obtain_edge_exe_times(division, parameter)
                    edge_times = transmit_times + edge_exe_times
                    cost = max(edge_times) + parameter.get_local_exe_time() + parameter.get_coordinate_cost() * division
                else:
                    cost = parameter.get_drop_penalty()
            else:
                cost = 0
            overall_costs += cost
        return overall_costs

    def obtain_edge_selections(self):
        """
        Obtain the feasible solution with random policy.

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
            assign_num = random.randint(0, parameter.get_max_assign())
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
                    if times >= parameter.get_ddl() or energys > self.__battery_energy_levels[i]:
                        edge_selections[i] = np.zeros(len(edge_selections[i]))
        return edge_selections

    def obtain_harvested_energys(self):
        """
        Randomly choose energy between $[0, E_i^H]$ for every mobile device, and then set self.harvested_energys.

        :return: no return
        """
        parameter = self.get_parameter()
        return list(map(random.uniform, [0] * parameter.get_user_num(), parameter.get_harvestable_energys()))

    def update_energy_levels(self):
        """
        Update the cost & virtual energy levels according to the involution expression \eqref{10}.

        :return: no return
        """
        parameter = self.get_parameter()
        for i in range(parameter.get_user_num()):
            division = int(sum(self.__edge_selections[i]))
            if division:
                self.__battery_energy_levels[i] = self.__battery_energy_levels[i] + \
                    self.__harvested_energys[i] - ToolFunction.obtain_transmit_energy(
                    division, self.__edge_selections[i], parameter, parameter.get_connectable_gains()[i]) - \
                    parameter.get_local_exe_energy()
            else:
                # check whether need to minus local_exe_energys
                # if self.__battery_energy_levels[i] < parameter.get_local_exe_energy():
                #     self.__battery_energy_levels[i] = self.__battery_energy_levels[i] + self.__harvested_energys[i]
                # else:
                #     self.__battery_energy_levels[i] = self.__battery_energy_levels[i] + \
                #         self.__harvested_energys[i] - parameter.get_local_exe_energy()
                self.__battery_energy_levels[i] = self.__battery_energy_levels[i] + self.__harvested_energys[i]
            self.__virtual_energy_levels[i] = self.__battery_energy_levels[i] - parameter.get_perturbation_para()

    def get_parameter(self):
        return self.__parameter

    def get_battery_energy_levels(self):
        return self.__battery_energy_levels

    def get_virtual_energy_levels(self):
        return self.__virtual_energy_levels

    def get_harvested_energys(self):
        return self.__harvested_energys

    def set_harvested_energys(self, harvested_energys):
        self.__harvested_energys = harvested_energys

    def get_edge_selections(self):
        return self.__edge_selections

    def set_edge_selections(self, edge_selections):
        self.__edge_selections = edge_selections
