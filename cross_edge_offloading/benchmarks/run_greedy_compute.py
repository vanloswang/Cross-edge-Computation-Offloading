"""
This module defines the interface to call GreedySelection Benchmark on Computation.
[--for test--]

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.benchmarks.greedy_computation_policy import GreedyCompute
from cross_edge_offloading.parameter import Parameter


all_costs = []


def opt():
    parameter = Parameter()
    greed_compute = GreedyCompute(parameter)
    for t in range(parameter.get_time_horizon()):
        print('Time slot #', t)
        # solve sub-problem P_2^{eh}, store the optimal harvested energys
        greed_compute.set_harvested_energys(greed_compute.obtain_harvested_energys())
        # use random method to solve sub-problem P_2^{es}, store the optimal edge selections
        greed_compute.set_edge_selections(greed_compute.obtain_edge_selections())
        # according the obtained optimal solutions (stored in self.edge_selections),
        # do calculations and draw diagrams ...
        costs = greed_compute.obtain_overall_costs(greed_compute.get_edge_selections())
        all_costs.append(costs)

        # update state information
        greed_compute.update_energy_levels()

        # update information for the next time slot
        parameter.go_next_slot()


if __name__ == '__main__':
    opt()
    for c in range(len(all_costs)):
        print(all_costs[c])
