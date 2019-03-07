"""
This module defines the class GreedyComm, which is the second benchmark (GSC1) proposed in this paper.
It inherits from class OffloadingCommon.

Author:
    Hailiang Zhao
"""
from cross_edge_offloading.offloading_common import OffloadingCommon
from cross_edge_offloading.utils.tool_function import ToolFunction
import numpy as np


class GreedyComm(OffloadingCommon):
    """
    This class implements the benchmark named GSC1.
    """
    def __init__(self, parameter):
        super().__init__(parameter)

    def obtain_edge_selections(self):
        """
        Obtain the feasible solution with greedy policy on communication.

        :return: edge_selections, every row denotes a mobile device who has task request
        """
        parameter = self.get_parameter()
        # deep copy the connectable servers and gains
        shadow_servers, shadow_gains, edge_selections = [], [], []
        for i in range(parameter.get_user_num()):
            tmp_s, tmp_g = [], []
            for j in range(len(parameter.get_connectable_servers()[i])):
                tmp_s.append(parameter.get_connectable_servers()[i][j])
                tmp_g.append(parameter.get_connectable_gains()[i][j])
            shadow_servers.append(tmp_s)
            shadow_gains.append(tmp_g)
            edge_selections.append(np.zeros(len(parameter.get_connectable_servers()[i])))

        best_gains, best_servers = [], []
        for i in range(parameter.get_user_num()):
            best_gain = max(shadow_gains[i])
            best_gains.append(best_gain)

            best_server_idx = shadow_gains[i].index(best_gain)
            best_servers.append(shadow_servers[i][best_server_idx])

        checked = [False] * parameter.get_user_num()
        while False in checked:
            # satisfy the maximum assignment constraint
            for j in range(parameter.get_server_num()):
                connected_users = [idx for idx, server in enumerate(best_servers) if server == j]
                if len(connected_users):
                    # at least one mobile device choose this server
                    connected_gains = [best_gains[i] for i in connected_users]

                    # only the user with the best channel power gains can be chosen
                    lucky_user = connected_users[connected_gains.index(max(connected_gains))]
                    checked[lucky_user] = True

                    # update best connectable information (remove j)
                    for i in range(parameter.get_user_num()):
                        if not checked[i]:
                            if shadow_servers[i].count(j) != 0:
                                server_idx = shadow_servers[i].index(j)
                                shadow_servers[i].pop(server_idx)
                                shadow_gains[i].pop(server_idx)
                else:
                    # this server is not chosen by any mobile device
                    continue
                # re-calculate the best server and gains for each mobile device
                for i in range(parameter.get_user_num()):
                    if len(shadow_gains[i]) != 0:
                        best_gains[i] = max(shadow_gains[i])
                        best_server_index = shadow_gains[i].index(best_gains[i])
                        best_servers[i] = shadow_servers[i][best_server_index]
                    else:
                        checked[i] = True

        # obtain edge_selections from best_servers
        for i in range(parameter.get_user_num()):
            if parameter.get_task_requests()[i] == 1:
                if best_servers[i]:
                    edge_idx = parameter.get_connectable_servers()[i].index(best_servers[i])
                    edge_selections[i][edge_idx] = 1

                # check whether the constraints can be satisfied
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
