"""
This module contains the class Parameter, which includes all parameters used for the computation
offloading problem at the edge.

Author:
    Hailiang Zhao, Cheng Zhang
"""
import pandas as pd
import numpy as np
from cross_edge_offloading.utils.tool_function import ToolFunction
import random


class Parameter(object):
    """
    This class contains all parameters used in computation offloading problem at the edge.
    """

    def __init__(self, time_slot_length=2e-3, time_horizon=50, ddl=2e-3, drop_penalty=2e-3, coordinate_cost=2e-5,
                 task_request_prob=0.7, unit_cpu_num=737.5, local_input_size=50, local_cpu_freq=1.5e9,
                 switched_cap=1e-28, max_transient_discharge=0.04, edge_input_size=3000, max_assign=3,
                 edge_cpu_freq=4.5e9, bandwidth=1e6, noise=1e-13, transmit_power=1, path_loss_const=1e-4,
                 min_distance=150, max_distance=400, max_channel_power_gain=1.02e-13, max_harvest_energy=4.8e-4,
                 v=1e-6):
        """
        Initialization.
        Without loss of generality, we set properties of every mobile device and every edge site the same, respectively.

        :param time_slot_length: the length of time slot in second
        :param time_horizon: the length of time horizon, i.e., the number of time slots
        :param ddl: the deadline of computation task (unit: second)
        :param drop_penalty: the penalty for dropping the computation task (unit: second)
        :param coordinate_cost: the cost for coordination/collaboration of edge sites (unit: second) [--tunable--]
        :param task_request_prob: the task request probability of all users under Bernoulli process
        :param unit_cpu_num: number of CPU-cycles required to process one bit computation task (num/bit)

        :param local_input_size: the input data size of the computation task for local execution (in bits)
        :param local_cpu_freq: the CPU-cycle frequency of all mobile devices
        :param switched_cap: the effective switched capacitance of all mobile devices
        :param max_transient_discharge: the maximum allowable transient discharge of mobile devices [--danger--]

        :param edge_input_size: the input data size of the computation task for edge/remote execution (in bits)
        :param max_assign: maximum number of assignments (acceptable mobile devices) of all edge sites [--tunable--]
        :param edge_cpu_freq: the CPU-cycle frequency of all edge servers

        :param bandwidth: the bandwidth of all edge sites (unit: Hz)
        :param noise: the background noise power of edge sites (unit: W)
        :param transmit_power: the transmitting power of mobile devices (unit: W)
        :param path_loss_const: the pass-loss constant for transmission
        :param min_distance: the minimum distance from mobile device to edge site under wireless signal coverage
        :param max_distance: the maximum distance from mobile device to edge site under wireless signal coverage
        :param max_channel_power_gain: the 'empirical' maximum channel power gain under exponential distribution
        with 4e8 trials
        :param max_harvest_energy: the maximum harvestable energy at each time slot (unit: J)
        :param v: the tuning parameter of Lyapunov Optimization (unit: J^2/sec) [--tunable--]
        """
        # =============================== basic parameters ===============================
        # parameters for scenario construction
        self.__time_slot_length = time_slot_length
        self.__time_horizon = time_horizon
        self.__ddl = ddl
        self.__drop_penalty = drop_penalty
        self.__coordinate_cost = coordinate_cost
        self.__task_request_prob = task_request_prob
        self.__unit_cpu_num = unit_cpu_num

        # parameters for local execution
        self.__local_input_size = local_input_size
        self.__local_cpu_freq = local_cpu_freq
        self.__switched_cap = switched_cap
        self.__max_transient_discharge = max_transient_discharge

        # parameter for edge execution
        self.__edge_input_size = edge_input_size
        self.__max_assign = max_assign
        self.__edge_cpu_freq = edge_cpu_freq

        # parameter for communication, energy harvesting and V
        self.__bandwidth = bandwidth
        self.__noise = noise
        self.__transmit_power = transmit_power
        self.__path_loss_const = path_loss_const
        self.__min_distance = min_distance
        self.__max_distance = max_distance
        self.__max_channel_power_gain = max_channel_power_gain
        self.__max_harvest_energy = max_harvest_energy
        self.__v = v

        # =============================== position information ===============================
        # read scenario data from dataset: initial users and servers' position
        self.__user_info, self.__edge_info = self.import_scenario()
        # get number of edge servers and users
        self.__user_num = len(self.__user_info)
        self.__server_num = len(self.__edge_info)
        # user position information
        self.__connectable_servers, self.__connectable_users, self.__connectable_distances, self.__connectable_gains, \
            self.__global_distances = self.obtain_wireless_signal_coverage()
        # mobility management
        self.__latitude_drv, self.__longitude_drv = self.obtain_derivation()

        # =============================== random events ===============================
        self.__harvestable_energys = ToolFunction.generate_harvestable_energys(self.__max_harvest_energy,
                                                                               self.__user_num)
        self.__task_requests = self.generate_task_request()

        # =============================== execution & cost information ===============================
        # calculate local execution time
        self.__local_exe_time = self.__local_input_size * self.__unit_cpu_num / self.__local_cpu_freq
        # calculate local execution energy consumption
        self.__local_exe_energy = self.__switched_cap * self.__local_input_size * self.__unit_cpu_num * \
            (self.__local_cpu_freq ** 2)
        # calculate the lower bound of perturbation parameter
        self.__perturbation_para = self.obtain_perturbation_para()
        # calculate the dimension size
        self.__dim_size = self.calculate_dim_size()

    def get_time_slot_length(self):
        return self.__time_slot_length

    def set_time_slot_length(self, time_slot_length):
        self.__time_slot_length = time_slot_length

    def get_time_horizon(self):
        return self.__time_horizon

    def set_time_horizon(self, time_horizon):
        self.__time_horizon = time_horizon

    def get_ddl(self):
        return self.__ddl

    def set_ddl(self, ddl):
        self.__ddl = ddl

    def get_drop_penalty(self):
        return self.__drop_penalty

    def set_drop_penalty(self, drop_penalty):
        self.__drop_penalty = drop_penalty

    def get_coordinate_cost(self):
        return self.__coordinate_cost

    def set_coordinate_cost(self, coordinate_cost):
        self.__coordinate_cost = coordinate_cost

    def get_task_request_prob(self):
        return self.__task_request_prob

    def set_task_request_prob(self, task_request_prob):
        self.__task_request_prob = task_request_prob

    def get_unit_cpu_num(self):
        return self.__unit_cpu_num

    def set_unit_cpu_num(self, unit_cpu_num):
        self.__unit_cpu_num = unit_cpu_num

    def get_local_input_size(self):
        return self.__local_input_size

    def set_local_input_size(self, local_input_size):
        self.__local_input_size = local_input_size

    def get_local_cpu_freq(self):
        return self.__local_cpu_freq

    def set_local_cpu_freq(self, local_cpu_freq):
        self.__local_cpu_freq = local_cpu_freq

    def get_switched_cap(self):
        return self.__switched_cap

    def set_switched_cap(self, switched_cap):
        self.__switched_cap = switched_cap

    def get_max_transient_discharge(self):
        return self.__max_transient_discharge

    def set_max_transient_discharge(self, max_transient_discharge):
        self.__max_transient_discharge = max_transient_discharge

    def get_edge_input_size(self):
        return self.__edge_input_size

    def set_edge_input_size(self, edge_input_size):
        self.__edge_input_size = edge_input_size

    def get_max_assign(self):
        return self.__max_assign

    def set_max_assign(self, max_assign):
        self.__max_assign = max_assign

    def get_edge_cpu_freq(self):
        return self.__edge_cpu_freq

    def set_edge_cpu_freq(self, edge_cpu_freq):
        self.__edge_cpu_freq = edge_cpu_freq

    def get_bandwidth(self):
        return self.__bandwidth

    def set_bandwidth(self, bandwidth):
        self.__bandwidth = bandwidth

    def get_noise(self):
        return self.__noise

    def set_noise(self, noise):
        self.__noise = noise

    def get_transmit_power(self):
        return self.__transmit_power

    def set_transmit_power(self, transmit_power):
        self.__transmit_power = transmit_power

    def get_path_loss_const(self):
        return self.__path_loss_const

    def set_path_loss_const(self, path_loss_const):
        self.__path_loss_const = path_loss_const

    def get_min_distance(self):
        return self.__min_distance

    def set_min_distance(self, min_distance):
        self.__min_distance = min_distance

    def get_max_distance(self):
        return self.__max_distance

    def set_max_distance(self, max_distance):
        self.__max_distance = max_distance

    def get_max_channel_power_gain(self):
        return self.__max_channel_power_gain

    def set_max_channel_power_gain(self, max_channel_power_gain):
        self.__max_channel_power_gain = max_channel_power_gain

    def get_max_harvest_energy(self):
        return self.__max_harvest_energy

    def set_max_harvest_energy(self, max_harvest_energy):
        self.__max_harvest_energy = max_harvest_energy

    def get_v(self):
        return self.__v

    def set_v(self, v):
        self.__v = v

    def import_scenario(self, user_data_dir='../dataset/users/users-melbcbd-generated.csv',
                        edge_data_dir='../dataset/edge-servers/site-optus-melbCBD.csv'):
        """
        Construct scenario (information of users/mobile devices and base stations) from imported dataset.
        'Melbourne CBD' in default.

        :param user_data_dir: csv file directory of user data
        :param edge_data_dir: csv file directory of edge site data
        :return: user information (latitude and longitude) and base station information (latitude, longitude
        and wireless signal coverage), presented in numpy array
        """
        user_info = pd.read_csv(user_data_dir).values
        # randomly choose a certain amount of users, 80 in default
        user_info = np.array([user_info[i] for i in random.sample(range(len(user_info)), 80)])
        edge_info = pd.read_csv(edge_data_dir, usecols=[1, 2]).values
        # edge_info = np.array([edge_info[j] for j in random.sample(range(len(edge_info)), 80)])

        # generate signal coverage (randomly chosen from minimum acceptable distance to maximum acceptable distance)
        signal_cov = np.array(np.random.randint(self.__min_distance, self.__max_distance, size=(len(edge_info), 1)))
        edge_info = np.c_[edge_info, signal_cov]

        return user_info, edge_info

    def obtain_perturbation_para(self):
        """
        Calculate the lower bound of perturbation parameter used in Lyapunov Optimization.

        :return: the lower bound of perturbation parameter
        """
        max_achievable_rate = ToolFunction.obtain_achieve_rate(self.__bandwidth, self.__max_channel_power_gain,
                                                               self.__transmit_power, self.__noise)
        part1 = self.__v * max_achievable_rate / (self.__transmit_power * self.__edge_input_size)
        part2 = self.__server_num * self.__drop_penalty - self.__edge_input_size / max_achievable_rate - \
            self.__edge_input_size * self.__unit_cpu_num / self.__edge_cpu_freq
        max_energy_all = self.__local_exe_energy + self.__server_num * \
            self.__transmit_power * (self.__ddl - self.__local_exe_time)
        return part1 * part2 + min(max_energy_all, self.__max_transient_discharge)

    def obtain_wireless_signal_coverage(self):
        """
        According to the position of mobile devices and edge sites, judge whether mobile devices are covered by edge
        sites' wireless signal.

        :return: lists stored indices of connectable edge sites, power gains, and mobile devices, respectively, and
        distances from mobile devices to connectable edge sites
        """
        connectable_servers, connectable_distances, connectable_gains = [], [], []
        global_distances = []  # a 'N * M' matrix stored distances between each user and each edge site
        for i in range(len(self.__user_info)):
            tmp_s, tmp_d, tmp_g = [], [], []
            tmp_gains = []
            for j in range(len(self.__edge_info)):
                distance = ToolFunction.obtain_geo_distance(self.__user_info[i], self.__edge_info[j])
                if distance <= self.__edge_info[j][2]:
                    # under signal coverage
                    tmp_s.append(j)
                    tmp_d.append(distance)
                    tmp_gains.append(ToolFunction.obtain_channel_power_gain(self.__path_loss_const, distance))
                tmp_g.append(distance)
            connectable_servers.append(tmp_s)
            connectable_distances.append(tmp_d)
            connectable_gains.append(tmp_gains)
            global_distances.append(tmp_g)

        connectable_users = []
        for j in range(len(self.__edge_info)):
            tmp_u = []
            for i in range(len(self.__user_info)):
                if global_distances[i][j] <= self.__edge_info[j][2]:
                    tmp_u.append(i)
            connectable_users.append(tmp_u)

        return connectable_servers, connectable_users, connectable_distances, connectable_gains, global_distances

    # properties below do not have setter methods
    def get_user_info(self):
        return self.__user_info

    def get_edge_info(self):
        return self.__edge_info

    def get_user_num(self):
        return self.__user_num

    def get_server_num(self):
        return self.__server_num

    def get_connectable_servers(self):
        return self.__connectable_servers

    def get_connectable_users(self):
        return self.__connectable_users

    def get_connectable_distances(self):
        return self.__connectable_distances

    def get_connectable_gains(self):
        return self.__connectable_gains

    def get_global_distances(self):
        return self.__global_distances

    def get_latitude_drv(self):
        return self.__latitude_drv

    def get_longitude_drv(self):
        return self.__longitude_drv

    # getter and setter for random events
    def get_harvestable_energys(self):
        return self.__harvestable_energys

    def set_harvestable_energys(self, harvestable_energys):
        self.__harvestable_energys = harvestable_energys

    def get_task_requests(self):
        return self.__task_requests

    def set_task_requests(self, task_requests):
        self.__task_requests = task_requests

    # properties below do not have setter methods
    def get_local_exe_time(self):
        return self.__local_exe_time

    def get_local_exe_energy(self):
        return self.__local_exe_energy

    def get_perturbation_para(self):
        return self.__perturbation_para

    def get_dim_size(self):
        return self.__dim_size

    # Below two methods are defined for user mobility.
    # In future, user mobility will be implemented by 'Random Walk Model'.
    def obtain_derivation(self):
        """
        Generate maximum derivation on user positions.
        [--This function will be deprecated in future. User mobility will be implemented by 'Random Walk Model'.--]

        :return: maximum derivation on latitude and longitude
        """
        positions = self.__user_info.T
        chosen_users = random.sample([i for i in range(self.__user_num)], int(self.__user_num / 2))
        non_chosen_users = [i for i in range(self.__user_num) if i not in chosen_users]

        latitudes_1 = [positions[0][i] for i in chosen_users]
        latitudes_2 = [positions[0][i] for i in non_chosen_users]
        latitude_drv = (np.mean(latitudes_1) - np.mean(latitudes_2)) / 20

        longitude_1 = [positions[1][i] for i in chosen_users]
        longitude_2 = [positions[1][i] for i in non_chosen_users]
        longitude_drv = (np.mean(longitude_1) - np.mean(longitude_2)) / 20

        return latitude_drv, longitude_drv

    def go_next_slot(self):
        """
        Generate users' position with random policy, and then update user_info.
        Finally, update connection information and dimension size.
        !! [--bugs--] Cannot confine users within this specific area. Therefore, bugs may arise. !!
        [--This function will be deprecated in future. User mobility will be implemented by 'Random Walk Model'.--]

        :return: no return
        """
        positions = self.__user_info.T
        positions[0] += np.random.uniform(-self.__latitude_drv, self.__latitude_drv, size=self.__user_num)
        positions[1] += np.random.uniform(-self.__longitude_drv, self.__longitude_drv, size=self.__user_num)
        self.__user_info = positions.T

        # update information who changes with user_info
        self.__connectable_servers, self.__connectable_users, self.__connectable_distances, self.__connectable_gains, \
            self.__global_distances = self.obtain_wireless_signal_coverage()
        self.__dim_size = self.calculate_dim_size()

    def generate_task_request(self):
        """
        Generate task request from Bernoulli Distribution.

        :return: a numpy array denoted task request, presented in [0, 1, ..., 1]
        """
        return ToolFunction.sample_from_bernoulli(self.__user_num, self.__task_request_prob)

    def calculate_dim_size(self):
        """
        Calculate dimension size from shape of 'edge_selections' for SAC (racos) algorithm.

        :return: no return
        """
        dim_size = 0
        for i in range(self.__user_num):
            if self.__task_requests[i] == 1:
                dim_size += len(self.__connectable_servers[i])
        return dim_size
