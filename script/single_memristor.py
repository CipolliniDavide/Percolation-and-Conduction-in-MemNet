from script_Opt.Class_SciPySparse.mem_parameters import mem_param, sim_param, volt_param, net_param, train_param, \
    env_param
from script_Opt.Class_SciPySparse.utils import utils
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.anim_vis import plot_H_evolution
from script_Opt.Class_SciPySparse.make_gif import make_gif
from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend
from copy import deepcopy
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
#import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import signal
# from IPython.display import display, clear_output
import networkx as nx
import sys
import copy
from os.path import isfile, join, abspath
from os import pardir
import time
import argparse

class Memristor():
    def __init__(self, mem_param):
        self.mem_param = mem_param
        self.g = 0
        self.G = mem_param.g_min

    def update(self, dV, delta_t):
        kp = self.mem_param.kp0 * np.exp(self.mem_param.eta_p * np.abs(dV))
        kd = self.mem_param.kd0 * np.exp(- self.mem_param.eta_d * np.abs(dV))

        self.g = kp / (kp + kd) * (1 - np.exp(-(kp + kd)*delta_t)) + self.g * np.exp(-(kp + kd)*delta_t)
        # print(self.g)
        # if self.g > 1:
        #     a=0
        self.G = self.mem_param.g_max * self.g + self.mem_param.g_min * (1-self.g)
        self.kd = kd
        self.kp = kp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='OutputGridAdiab', type=str)
    parser.add_argument('-diag', '--diagonals', default=0, type=int)
    # parser.add_argument('-crt_sim_data', '--create_sim_data', default=0, type=int)
    args = parser.parse_args()

    sim_param = edict({'T': 100,  # 4e-3, # [s]
                       # 'steps': 100,
                       'sampling_rate': 500  # 1000 # [Hz]  # =steps / T  # [Hz]
                       # dt = T / steps  # [s] or    dt = 1/sampling_rate bc  steps= sampling_rate *T
                       })

    mem_param = edict({'kp0': 1e-3,  # model kp_0
                       'kd0': 1,  # model kd_0
                       'eta_p': 10,  # model eta_p
                       'eta_d': 1,  # model eta_d
                       'g_min': 1,  # model g_min
                       'g_max': 100,  # model g_max
                       'g0': 1  # model g_0
                       })

    mem = Memristor(mem_param=mem_param)

    Vbias = 1

    # inputsignal = ControlSignal(sources=[0], sim_param=sim_param, volt_param=volt_param)
    # inputsignal.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)
    # inputsignal.V_list[0] = [Vbias] * len(inputsignal.t_list)
    # inputsignal.V_list[0][:2] = .1

    # j=2
    # a = np.hstack((np.linspace(.0001, Vbias, len(inputsignal.t_list) // j),
    #                Vbias * np.ones(len(inputsignal.t_list) - len(inputsignal.t_list) // j)))
    # a = np.hstack((a, list(a)[::-1]))
    # sim_param.T = sim_param.T * 2
    # inputsignal.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)
    # inputsignal.V_list = np.zeros((1, len(inputsignal.t_list)))
    # inputsignal.V_list[0] = list(a[:-1])


    # Define control signal
    sim_param.T = 1
    inputsignal = ControlSignal(sources=[0], sim_param=sim_param, volt_param=volt_param)
    inputsignal.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)
    # sample rate
    freq = 1  # the frequency of the signal
    inputsignal.V_list[0] = Vbias * np.sin(2 * np.pi * freq * (inputsignal.t_list))  # / fs))


    G = np.zeros(len(inputsignal.t_list))
    g = np.zeros(len(inputsignal.t_list))
    kd = np.zeros(len(inputsignal.t_list))
    kp = np.zeros(len(inputsignal.t_list))

    delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]
    for t in range(0, len(inputsignal.t_list)):
        mem.update(inputsignal.V_list[0][t], delta_t=delta_t)
        G[t] = mem.G
        g[t] = mem.g
        kp[t] = mem.kp
        kd[t] = mem.kd

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(inputsignal.t_list, G, color="red", marker="o")
    ax.set_ylabel("G", color="red", fontsize=14)
    ax.set_xlabel("t", color="black", fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(inputsignal.t_list, inputsignal.V_list[0], color="blue", marker="o")
    ax2.set_ylabel("V", color="blue", fontsize=14)
    plt.show()

    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.scatter(inputsignal.V_list[0], g, color='red')
    # ax.set_ylabel("g", color="red", fontsize=14)
    # ax.set_xlabel("V", color="blue", fontsize=14)
    # ax2 = ax.twinx()
    # ax2.plot(inputsignal.V_list[0][1:], np.diff(g) / np.diff(inputsignal.V_list[0]), color='blue')
    # ax2.set_ylabel("dg/dV", color="blue", fontsize=14)

    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(inputsignal.V_list[0], inputsignal.V_list[0]*G, color='red')
    ax.set_ylabel("I", color="red", fontsize=14)
    ax.set_xlabel("V", color="blue", fontsize=14)
    # ax2 = ax.twinx()
    # ax2.plot(inputsignal.V_list[0], , color='blue')
    # ax2.set_ylabel("dg/dV", color="blue", fontsize=14)

    # plt.plot(g)
    # plt.show()

    a=0





