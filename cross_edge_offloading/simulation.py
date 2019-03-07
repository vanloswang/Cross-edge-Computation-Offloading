"""
This module defines the interface to run Simulation.

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.cco.cco_algorithm import CcoAlgorithm
from cross_edge_offloading.benchmarks.random_policy import RandomPolicy
from cross_edge_offloading.benchmarks.greedy_communication_policy import GreedyComm
from cross_edge_offloading.benchmarks.greedy_computation_policy import GreedyCompute
from cross_edge_offloading.parameter import Parameter
from racos.dimension import Dimension
import numpy as np


def opt():
    parameter = Parameter()
    dimension = Dimension(parameter.get_dim_size())
    cco = CcoAlgorithm(parameter, dimension)
    rand_policy = RandomPolicy(parameter)
    greed_comm = GreedyComm(parameter)
    greed_compute = GreedyCompute(parameter)

    cco_costs, rand_costs, greed_comm_costs, greed_compute_costs = [], [], [], []
    cco_battery = np.zeros([parameter.get_time_horizon(), parameter.get_user_num()])
    rand_battery = np.zeros([parameter.get_time_horizon(), parameter.get_user_num()])
    greed_comm_battery = np.zeros([parameter.get_time_horizon(), parameter.get_user_num()])
    greed_compute_battery = np.zeros([parameter.get_time_horizon(), parameter.get_user_num()])

    # write overall costs and cost energy levels to txt files
    cost_file = open('overall_costs.txt', 'w+')
    battery_file = open('battery_levels.txt', 'w+')

    for t in range(parameter.get_time_horizon()):
        print('Time slot #', t)

        # record the cost energy levels
        for i in range(parameter.get_user_num()):
            cco_battery[t][i] = cco.get_battery_energy_levels()[i]
            rand_battery[t][i] = rand_policy.get_battery_energy_levels()[i]
            greed_comm_battery[t][i] = greed_comm.get_battery_energy_levels()[i]
            greed_compute_battery[t][i] = greed_compute.get_battery_energy_levels()[i]

        # solve sub-problem P_2^{eh}, store the optimal harvested energys
        cco.set_harvested_energys(cco.obtain_harvested_energys())
        rand_policy.set_harvested_energys(rand_policy.obtain_harvested_energys())
        greed_comm.set_harvested_energys(greed_comm.obtain_harvested_energys())
        greed_compute.set_harvested_energys(greed_compute.obtain_harvested_energys())

        # call racos to solve sub-problem P_2^{es}, store the optimal edge selections
        cco.racos(cco.calculate_object_func)
        cco.set_edge_selections(cco.instance_to_edge_selections(cco.get_optimal_solution()))
        cco_costs.append(cco.obtain_overall_costs(cco.get_edge_selections()))
        print('CCO overall cost:', cco_costs[t])

        rand_policy.set_edge_selections(rand_policy.obtain_edge_selections())
        rand_costs.append(rand_policy.obtain_overall_costs(rand_policy.get_edge_selections()))
        print('RS overall cost:', rand_costs[t])

        greed_comm.set_edge_selections(greed_comm.obtain_edge_selections())
        greed_comm_costs.append(greed_comm.obtain_overall_costs(greed_comm.get_edge_selections()))
        print('GSC1 overall cost:', greed_comm_costs[t])

        greed_compute.set_edge_selections(greed_compute.obtain_edge_selections())
        greed_compute_costs.append(greed_compute.obtain_overall_costs(greed_compute.get_edge_selections()))
        print('GSC2 overall cost:', greed_compute_costs[t])

        # update state information
        cco.update_energy_levels()
        rand_policy.update_energy_levels()
        greed_comm.update_energy_levels()
        greed_compute.update_energy_levels()

        # store results in file
        cost_file.write(str(cco_costs[t]) + ' ' + str(rand_costs[t]) + ' ' + str(greed_comm_costs[t]) + ' ' +
                        str(greed_compute_costs[t]) + '\n')
        battery_file.write(str(np.mean(cco_battery[t])) + ' ' + str(np.mean(rand_battery[t])) + ' ' +
                           str(np.mean(greed_comm_battery[t])) + ' ' + str(np.mean(greed_compute_battery[t])) + '\n')

        parameter.go_next_slot()
        dimension.set_dim_size(parameter.get_dim_size())

    cost_file.close()
    battery_file.close()


if __name__ == '__main__':
    opt()
