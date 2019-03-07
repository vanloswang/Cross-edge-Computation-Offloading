"""
This module defines the interface to call Random Selection Benchmark.

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.benchmarks.random_policy import RandomPolicy
from cross_edge_offloading.parameter import Parameter


all_costs = []


def opt():
    parameter = Parameter()
    rand_policy = RandomPolicy(parameter)
    for t in range(parameter.get_time_horizon()):
        print('Time slot #', t)
        # solve sub-problem P_2^{eh}, store the optimal harvested energys
        rand_policy.set_harvested_energys(rand_policy.obtain_harvested_energys())
        # use random method to solve sub-problem P_2^{es}, store the optimal edge selections
        rand_policy.set_edge_selections(rand_policy.obtain_edge_selections())
        # according the obtained optimal solutions (stored in self.edge_selections),
        # do calculations and draw diagrams ...
        costs = rand_policy.obtain_overall_costs(rand_policy.get_edge_selections())
        all_costs.append(costs)

        # update state information
        rand_policy.update_energy_levels()

        # update information for the next time slot
        parameter.go_next_slot()


if __name__ == '__main__':
    opt()
    for c in range(len(all_costs)):
        print(all_costs[c])
