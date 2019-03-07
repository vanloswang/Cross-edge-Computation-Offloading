"""
This module defines the class CcoAlgorithm, which is the main algorithm proposed in this paper. It inherits from class
Racos. In order to avoid Multiple Inheritance, the code from class OffloadingCommon is coped.

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.utils.tool_function import ToolFunction
import numpy as np
import random
from racos.racos import Racos
from racos.instance import Instance
from racos.dimension import Dimension


class CcoAlgorithm(Racos):
    """
    This class implements the algorithm named CCO proposed in the paper.
    """

    def __init__(self, parameter, dimension):
        """
        Super from base class OffloadingCommon and Esracos, then update the way CCO initializes feasible solution.

        :param parameter: the instance of class Parameter
        :param dimension: the instance of class Dimension
        """
        super().__init__(dimension)

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
        ============================= [copy from offloading_common.py] =============================
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
        ============================= [copy from offloading_common.py] =============================
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

    def obtain_sub_problem_es(self, edge_selections):
        """
        Calculate the optimization goal of sub-problem $\mathcal{P}_2^{es}$.

        :param edge_selections: the edge selection decisions for all mobile devices
        :return: the optimization goal of $\mathcal{P}_2^{es}$
        """
        parameter = self.get_parameter()
        optimization_func_value = 0
        for i in range(parameter.get_user_num()):
            division = int(sum(edge_selections[i]))
            if division:
                energy_consumes = parameter.get_local_exe_energy() + \
                                  ToolFunction.obtain_transmit_energy(division, edge_selections[i], parameter,
                                                                      parameter.get_connectable_gains()[i])
                negative_part = self.__virtual_energy_levels[i] * energy_consumes
            else:
                if self.__battery_energy_levels[i] < parameter.get_local_exe_energy():
                    negative_part = 0
                else:
                    negative_part = self.__virtual_energy_levels[i] * parameter.get_local_exe_energy()
            optimization_func_value -= negative_part
        optimization_func_value += parameter.get_v() * self.obtain_overall_costs(edge_selections)
        return optimization_func_value

    def obtain_edge_selections(self):
        """
        Override base method with greedy policy.

        :return: a feasible solution: edge_selections, which has the same shape with connectable_servers
        """
        parameter = self.get_parameter()
        assignable_nums = np.repeat(parameter.get_max_assign(), parameter.get_server_num())
        return self.greedy_sample(assignable_nums)

    def greedy_sample(self, assignable_nums):
        """
        According to the assignable numbers of edge sites, sample solution with greedy policy.

        :param assignable_nums: the assignable numbers of edge sites, which is <= max_assign
        :return: a feasible solution edge_selections, which has the same shape with connectable_servers
        """
        parameter = self.get_parameter()
        # first initialize with zero
        edge_selections = []
        for i in range(parameter.get_user_num()):
            edge_selection = np.repeat(0, len(parameter.get_connectable_servers()[i]))
            edge_selections.append(edge_selection)

        # for every edge site, directly distribute max_assign connections to connectable mobile devices
        for j in range(parameter.get_server_num()):
            assign_num = assignable_nums[j]
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
                for i in range(assign_num):
                    user_index = user_indices[i]
                    edge_index = list.index(parameter.get_connectable_servers()[user_index], j)
                    edge_selections[user_index][edge_index] = 1

        # set those mobile devices who do not have task request to [0, 0, ..., 0]
        # we can not delete them from the list because every row is the index of the corresponding mobile device
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 0:
                edge_selections[i] = np.zeros(len(edge_selections[i]))
        return edge_selections

    def obtain_harvested_energys(self):
        """
        Obtain the optimal harvested energy by solving the 'optimal energy harvesting' sub-problem
        according to \eqref{18}, and then update self.harvested_energys.

        :return: a list of optimal harvested energy of every mobile devices
        """
        parameter = self.get_parameter()
        harvested_energys = []
        for i in range(parameter.get_user_num()):
            if self.__virtual_energy_levels[i] <= 0:
                harvested_energys.append(parameter.get_harvestable_energys()[i])
            else:
                harvested_energys.append(0)
        return harvested_energys

    def update_energy_levels(self):
        """
        ============================= [copy from offloading_common.py] =============================
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

    def instance_to_edge_selections(self, instance):
        """
        Convert the (best-so-far) instance of class Instance to edge_selections (instance -> edge_selections,
        called for __optimal_solution).

        :param instance: the instance of class Instance
        :return: edge selection decisions of all mobile devices
        """
        parameter = self.get_parameter()
        chosen_sites = instance.deep_copy()
        edge_selections = []
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 1:
                edge_selection = np.array(chosen_sites.get_features()[0: len(parameter.get_connectable_servers()[i])])
                edge_selections.append(edge_selection)
                del chosen_sites.get_features()[0: len(parameter.get_connectable_servers()[i])]
            else:
                edge_selections.append(np.repeat(0, len(parameter.get_connectable_servers()[i])))
        return edge_selections

    def generate_random_ins(self):
        """
        Generate random instance for those dimensions whose labels are True. In this version, we check whether the two
        constraints can be satisfied. (Actually, we first generate random edge_selections, and then convert it to an
        instance.)

        :return: a feasible instance
        """
        parameter = self.get_parameter()
        assignable_nums = np.repeat(parameter.get_max_assign(), parameter.get_server_num())
        edge_selections = self.greedy_sample(assignable_nums)
        ins_features = []
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 1:
                division = int(sum(edge_selections[i]))
                if division:
                    times = self.obtain_time_consumption(division, edge_selections[i],
                                                         parameter.get_connectable_gains()[i])
                    energys = ToolFunction.obtain_transmit_energy(division, edge_selections[i], parameter,
                                                                  parameter.get_connectable_gains()[i])
                    # satisfy the constraint
                    if times >= parameter.get_ddl() or energys > self.__battery_energy_levels[i]:
                        edge_selections[i] = np.zeros(len(edge_selections[i]))
                else:
                    pass
                for j in range(len(edge_selections[i])):
                    ins_features.append(edge_selections[i][j])
        dimension = Dimension(parameter.get_dim_size())
        instance = Instance(dimension)
        instance.set_features(ins_features)
        return instance

    def generate_from_pos_ins(self, pos_instance):
        """
        Generate an instance from an exist positive instance.

        :param pos_instance: the exist positive instance
        :return: a feasible instance
        """
        parameter = self.get_parameter()
        assignable_nums = np.repeat(parameter.get_max_assign(), parameter.get_server_num())
        edge_selections = self.greedy_sample(assignable_nums)
        ins_features = []
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 1:
                division = int(sum(edge_selections[i]))
                if division:
                    times = self.obtain_time_consumption(division, edge_selections[i],
                                                         parameter.get_connectable_gains()[i])
                    energys = ToolFunction.obtain_transmit_energy(division, edge_selections[i], parameter,
                                                                  parameter.get_connectable_gains()[i])
                    # satisfy the constraint
                    if times >= parameter.get_ddl() or energys > self.__battery_energy_levels[i]:
                        edge_selections[i] = np.zeros(len(edge_selections[i]))
                else:
                    pass
                for j in range(len(edge_selections[i])):
                    ins_features.append(edge_selections[i][j])
        dimension = Dimension(parameter.get_dim_size())
        instance = Instance(dimension)
        instance.set_features(ins_features)

        # update from the chosen positive instance
        for i in range(len(instance.get_features())):
            if self.get_labels()[i]:
                instance.set_feature(i, pos_instance.get_feature(i))

        # re-check whether the maximum assignable nums is satisfied
        edge_selections = self.instance_to_edge_selections(instance)
        for j in range(parameter.get_server_num()):
            if assignable_nums[j] >= len(parameter.get_connectable_users()[j]):
                continue
            connect_users = []
            for i in range(len(parameter.get_connectable_users()[j])):
                user_idx = parameter.get_connectable_users()[j][i]
                edge_idx = list.index(parameter.get_connectable_servers()[user_idx], j)
                if edge_selections[user_idx][edge_idx] == 1:
                    connect_users.append(user_idx)
            if len(connect_users) <= assignable_nums[j]:
                continue
            permitted_connect_users = random.sample(connect_users, assignable_nums[j])
            non_permitted = [u for u in connect_users if u not in permitted_connect_users]
            for i in range(len(non_permitted)):
                user_idx = non_permitted[i]
                edge_idx = list.index(parameter.get_connectable_servers()[user_idx], j)
                edge_selections[user_idx][edge_idx] = 0

        # re-check whether the ddl and energy consumption are satisfied
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 1:
                division = int(sum(edge_selections[i]))
                if division:
                    # times = self.obtain_time_consumption(division, edge_selections[i],
                    #                                      parameter.get_connectable_gains()[i])
                    transmit_times = ToolFunction.obtain_transmit_times(division, edge_selections[i], parameter,
                                                                        parameter.get_connectable_gains()[i])
                    edge_exe_times = ToolFunction.obtain_edge_exe_times(division, parameter)
                    edge_times = transmit_times + edge_exe_times
                    times = max(edge_times) + parameter.get_local_exe_time() + \
                        parameter.get_coordinate_cost() * division
                    energys = ToolFunction.obtain_transmit_energy(division, edge_selections[i], parameter,
                                                                  parameter.get_connectable_gains()[i])
                    # satisfy the constraint
                    if times >= parameter.get_ddl() or energys > self.__battery_energy_levels[i]:
                        edge_selections[i] = np.zeros(len(edge_selections[i]))
                else:
                    pass

        # get final instance from the new edge_selections
        ins_features = []
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 1:
                for j in range(len(edge_selections[i])):
                    ins_features.append(edge_selections[i][j])
        instance.set_features(ins_features)
        return instance

    def calculate_object_func(self, instance):
        """
        According to the input instance, calculate the value of objective function (i.e., optimization_func).

        :param instance: the instance of class Instance
        :return: the value of objective function, i.e., V * overall_costs + Lyapunov_drift
        """
        # calculate edge_selections from Solution instances (obtained by RACOS)
        edge_selections = self.instance_to_edge_selections(instance)
        return self.obtain_sub_problem_es(edge_selections)

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
