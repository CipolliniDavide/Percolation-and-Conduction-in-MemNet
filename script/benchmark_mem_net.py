import os.path
import random
import datetime
# from easydict import EasyDict as edict
import numpy as np
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
from script_Opt.Class_SciPySparse.utils import utils
from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend

def optimized_alg_SciPySparse(N, src, gnd):
    from script_Opt.Class_SciPySparse.mem_parameters import mem_param, sim_param, volt_param, net_param, train_param, env_param
    from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet
    from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal

    time_list = []
    edge_list = []
    for i in N:
        print('Opt: {:d}'.format(i))
        start_time = time.time()
        net_param.rows = i
        net_param.cols = i
        # Instantiate Classes
        net = MemNet(mem_param=mem_param, net_param=net_param, gnd=gnd, src=src)
        inputsignal = ControlSignal(sources=src)
        edge_list.append(net.number_of_edges)
        net.run(t_list=inputsignal.t_list, groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list)
        time_list.append((time.time() - start_time))
    np.save(file=save_path+'time_optAlg_SciPySparse', arr=time_list)
    return time_list, edge_list

def unoptimized_alg_NetworkX(N, src, gnd):
    from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
    from Grid_Graph_Modeling_Memristive_Nanonetworks.network_model import Milano_implementation

    time_list = []
    edge_list = []
    for i in N:
        print('UnOpt: {:d}'.format(i))
        start_time = time.time()
        inputsignal = ControlSignal(sources=src)

        # Instantiate Classes
        net = Milano_implementation(lin_size=i, src=src, gnd=gnd, t_list=inputsignal.t_list, V_list=inputsignal.V_list)
        time_list.append((time.time() - start_time))
    np.save(file=save_path+'time_unoptAlg_NetworkX', arr=time_list)
    return time_list, edge_list



if __name__ == "__main__":
    root = abspath(join(".", pardir))
    src = [2, 5]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [22, 10]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    # src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    # gnd = [430]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

    # Number of rows and columns
    N = np.arange(10, 110, 10)

    labels = [(gnd[0], 'Gnd'), (src[0], 'Src')]
    save_path = join(root, 'BenchmarkOutput/{:d}/'.format(N[-1]))
    # save_path = root + '/BenchmarkOutput/'
    utils.ensure_dir(save_path)
    print(save_path)


    t_scipy, n_ed_scipy = optimized_alg_SciPySparse(N, src, gnd)
    # t_unop, n_ed_netX = unoptimized_alg_NetworkX(N, src, gnd)

    import pandas as pd
    # t_jl = pd.read_csv('/Users/dav/PycharmProjects/MemNet/julia_memNet/time_list_jl.csv')
    t_scipy = np.load(save_path+'time_optAlg_SciPySparse.npy')/60
    t_unop = np.load(save_path+'time_unoptAlg_NetworkX.npy')/60
    # # fig, [ax, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    number_of_edges = 2*np.power(N, 2)-2*N
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 10))
    ax.plot(number_of_edges, t_scipy, '-o', label='Our implementation')
    ax.plot(number_of_edges, t_unop, '-o', label='From literature')
    # ax.plot(N ** 2, t_scipy_mix, '-o', label='SciPy_mixed')
    # ax.plot(N ** 2, np.array(t_jl['time']), '-o', label='Julia')
    set_ticks_label(ax=ax, ax_label='Time [min]', data=np.concatenate((t_scipy, t_unop)),
                    ax_type='y', num=5, valfmt="{x:.1f}")
    set_ticks_label(ax=ax, ax_label='Number of memristors', data=number_of_edges,
                    ax_type='x', num=5, valfmt="{x:.0f}")
    set_legend(ax=ax, title='', ncol=1, loc=1)
    plt.tight_layout()
    plt.savefig(save_path+'benchmark.svg', format='svg', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 10))
    ax.semilogy(number_of_edges, t_scipy, '-o', label='Our implementation')
    ax.semilogy(number_of_edges, t_unop, '-o', label='From literature')
    # ax.plot(N ** 2, t_scipy_mix, '-o', label='SciPy_mixed')
    # ax.plot(N ** 2, np.array(t_jl['time']), '-o', label='Julia')
    set_ticks_label(ax=ax, ax_label='Time [min]', data=[0, 2], add_ticks=[.1],
                    ax_type='y', valfmt="{x:.1f}", scale='log')
    set_ticks_label(ax=ax, ax_label='Number of memristors', data=number_of_edges,
                    ax_type='x', num=5, valfmt="{x:.1e}")
    set_legend(ax=ax, title='', ncol=1, loc=1)
    plt.tight_layout()
    plt.savefig(save_path + 'log_benchmark.svg', format='svg', dpi=1200)
    plt.show()

    a=0

    # ax2.plot(n_ed_scipy, t_scipy, '-o', label='SciPySparse')
    # ax2.plot(n_ed_netX, t_NetX, '-o', label='NetworkX')
    # ax2.set_xlabel('# edges')
    # ax2.set_ylabel('Time [s]')

    # from script_Opt.Class import visual_utils
    # x = [N**2] #, np.array([n_ed_netX, n_ed_scipy])]
    # x_label = ['Number of nodes', 'Number of edges']
    # y = np.array([t_scipy_mix, t_scipy])
    # for i, axes in enumerate([ax]):
    #     visual_utils.set_ticks_label(ax=axes, ax_label=x_label[i], ax_type='x', data=x[i].flatten())
    #     visual_utils.set_ticks_label(ax=axes, ax_type='y', ax_label='Time [s]',data=y.flatten())
    #
    # plt.savefig(save_path+'benchmark_SciPyOnly.png')
    # #plt.show()
    #plt.close()

