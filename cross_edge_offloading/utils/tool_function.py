"""
This module contains static tool functions about Communication and Computing details in edge computing systems.

Author:
    Hailiang Zhao
"""
import math
import numpy as np
from geopy.distance import geodesic
import random


class ToolFunction(object):
    """
    This class defines the tool functions about Communication and Computing.
    """

    def __init__(self):
        pass

    @staticmethod
    def obtain_achieve_rate(bandwidth, channel_power_gain, transmit_power, noise):
        """
        Calculate the achievable rate under Shannon-Hartley Theorem, with both inter-cell & intra-cell
        interference not considered.

        :param bandwidth: transmission bandwidth
        :param channel_power_gain: channel power gain from sender to receiver
        :param transmit_power: transmitting power from sender to receiver
        :param noise: background noise at the receiver
        :return: the achievable rate of the transmission
        """
        return bandwidth * math.log2(1 + channel_power_gain * transmit_power / noise)

    @staticmethod
    def obtain_channel_power_gain(path_loss_const, distance):
        """
        Calculate the channel power gain from source to destination. Currently we assume that it is exponentially
        distributed with mean $g_0 d^{-4}$, where $g_0$ is the path-loss constant, $d$ is the distance from sender
        to receiver.

        :param path_loss_const: path-loss constant
        :param distance: distance from sender to receiver
        :return: the channel power gain under exponential distribution
        """
        mu = path_loss_const * (distance ** -4)
        return np.random.exponential(scale=mu)

    @staticmethod
    def obtain_geo_distance(user_pos, server_pos):
        """
        Calculate the geography distance between a particular mobile device and a particular edge server.

        :param user_pos: user's position (latitude, longitude), described in numpy array
        :param server_pos: edge server's position (latitude, longitude), described in numpy array
        :return: geography distance between user and server
        """
        return geodesic(tuple(user_pos), tuple(server_pos)).m

    @staticmethod
    def generate_harvestable_energys(max_harvest_energy, trials):
        """
        Generate harvestable energy $E_i^H$ for all mobile devices.

        :param max_harvest_energy: the pre-defined maximum harvestable energy
        :param trials: the number of trials
        :return: a list contains generated harvestable energy
        """
        return list(map(random.uniform, [0] * trials, [max_harvest_energy] * trials))

    @staticmethod
    def obtain_transmit_times(division, edge_selection, parameter, channel_power_gains):
        """
        Calculate the transmission time of one mobile device to its every chosen 'connectable' edge sites,
        described in numpy array. Only be called when edge_selections is not [0, 0, ..., 0].

        :param division: the number of chosen edge sites
        :param edge_selection: the edge selection decision, such as [0,1,0,...,1] (numpy array)
        :param parameter: the instance of Parameter
        :param channel_power_gains: the channel power gains from a mobile device to every connectable edge sites
        :return: the transmit times from a user to chosen edge sites (numpy array)
        """
        offload_data_size = parameter.get_edge_input_size() / division
        # remove non-chosen gains
        gains = [channel_power_gains[j] for j in range(len(channel_power_gains)) if edge_selection[j] == 1]

        achieve_rates = list(map(ToolFunction.obtain_achieve_rate, [parameter.get_bandwidth()] * division,
                                 gains, [parameter.get_transmit_power()] * division,
                                 [parameter.get_noise()] * division))
        transmit_times = np.repeat(offload_data_size, division) / achieve_rates
        return transmit_times

    @staticmethod
    def obtain_transmit_energy(division, edge_selection, parameter, channel_power_gains):
        """
        Calculate the transmission energy consumption of one mobile device.

        :param division: the number of chosen edge sites
        :param edge_selection: the edge selection decision, such as [0,1,0,...,1] (numpy array)
        :param parameter: the instance of Parameter
        :param channel_power_gains: the channel power gains from a mobile device to every connectable edge sites
        :return: the transmit energy consumption
        """
        return sum(ToolFunction.obtain_transmit_times(division, edge_selection, parameter, channel_power_gains) *
                   parameter.get_transmit_power())

    @staticmethod
    def obtain_edge_exe_times(division, parameter):
        """
        Calculate the edge execution time of a mobile device on every chosen edge sites. Only be called when
        edge_selections is not [0, 0, ..., 0].

        :param division: the number of chosen edge sites
        :param parameter: the instance of Parameter
        :return: the edge execution times o f chosen edge sites (numpy array)
        """
        # CPU-cycle required in every chosen edge sites
        cpu_cycle_required = parameter.get_unit_cpu_num() * parameter.get_edge_input_size() / division
        # notice that we set CPU-cycle frequency of every edge sites the same,
        # thus the edge execution times are the same, otherwise we should use map function
        edge_exe_times = np.repeat(cpu_cycle_required / parameter.get_edge_cpu_freq(), division)
        return edge_exe_times

    @staticmethod
    def sample_from_bernoulli(trials, prob_threshold):
        """
        Sample from Bernoulli Distribution with probability $\\rho$.

        :param trials: number of trials
        :param prob_threshold: the threshold probability
        :return: sampling results in numpy array ([0, 1, 1, ..., 0, 1])
        """
        samples = np.repeat(0, trials)
        for i in range(trials):
            samples[i] = 1 if random.random() <= prob_threshold else 0
        return samples

    @staticmethod
    def sample_uniform_integer(lower, upper):
        """
        Sample from uniform distribution in discrete region [lower, upper].

        :param lower: the lower bound of region
        :param upper: the upper bound of region
        :return: the sampled result
        """
        return random.randint(lower, upper)

    @staticmethod
    def sample_uniform_double(lower, upper):
        """
        Sample from uniform distribution in continues region [lower, upper].

        :param lower: the lower bound of region
        :param upper: the upper bound of region
        :return: the sampled result
        """
        return random.uniform(lower, upper)
