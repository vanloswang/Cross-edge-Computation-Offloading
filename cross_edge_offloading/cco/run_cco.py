"""
This module defines the interface to call Cross-edge Computation Offloading Algorithm (CCO).
[--for test--]

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.cco.cco_algorithm import CcoAlgorithm
from cross_edge_offloading.parameter import Parameter
from racos.dimension import Dimension


all_costs = []


def opt():
    parameter = Parameter()
    dimension = Dimension(parameter.get_dim_size())
    cco = CcoAlgorithm(parameter, dimension)
    for t in range(parameter.get_time_horizon()):
        print('Time slot #', t)
        # solve sub-problem P_2^{eh}, store the optimal harvested energys
        cco.set_harvested_energys(cco.obtain_harvested_energys())
        # call racos to solve sub-problem P_2^{es}, store the optimal edge selections
        cco.racos(cco.calculate_object_func)
        cco.set_edge_selections(cco.instance_to_edge_selections(cco.get_optimal_solution()))
        # according the obtained optimal solutions (stored in self.edge_selections),
        # do calculations and draw diagrams ...
        costs = cco.obtain_overall_costs(cco.get_edge_selections())
        all_costs.append(costs)

        # update state information
        cco.update_energy_levels()

        # update information for the next time slot
        parameter.go_next_slot()
        dimension.set_dim_size(parameter.get_dim_size())


if __name__ == '__main__':
    opt()
    for c in range(len(all_costs)):
        print(all_costs[c])
