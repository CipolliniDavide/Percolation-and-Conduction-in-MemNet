from script_Opt.Class_SciPySparse.mem_parameters import sim_param, volt_param, net_param, mem_param
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.utils import utils, create_dataset, getListOfFiles
from script_Opt.Class_SciPySparse.anim_vis import plot_H_evolution
from script_Opt.Class_SciPySparse.make_gif import make_gif
from script_Opt.Class_SciPySparse.visualize import visualize
from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend
# from script_Opt.infotheory.density import batch_beta_relative_entropy

import math
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm, contrib
#import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import signal
# from IPython.display import display, clear_output
import networkx as nx
import sys
import copy
from glob import glob
from os.path import isfile, join, abspath
from os import pardir
import time
from multiprocessing import Pool, cpu_count
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph, save_npz, load_npz
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm, logm, eigvalsh
from scipy.linalg import eigvals, eigh
from easydict import EasyDict as edict
import itertools

def take_2nd_derivative(x, y):
    dy = np.diff(y, 1)
    dx = np.diff(x, 1)
    yfirst = dy / dx
    xfirst = 0.5 * (x[:-1] + x[1:])
    dyfirst = np.diff(yfirst, 1)
    dxfirst = np.diff(xfirst, 1)
    ysecond = dyfirst / dxfirst
    xsecond = 0.5 * (xfirst[:-1] + xfirst[1:])
    return ysecond, xsecond

def take_1rst_derivative(x, y):
    dy = np.diff(y, 1)
    dx = np.diff(x, 1)
    yfirst = dy / dx
    xfirst = 0.5 * (x[:-1] + x[1:])
    return yfirst

def run(net, groundnode_list, sourcenode_list, delta_t, dictionary, save_path=None):
    # delta_t = t_list[1] - t_list[0]

    V_rec = []
    Gnw_rec = []
    net_entropy = []
    # eq_G = []
    # eq_V = []
    Vbias = 0
    deltaV = .2

    _ = net.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=np.reshape([.1], (1, 1)), t=0)

    while Vbias <= args.Vbias:
        flag = 0
        Vbias += deltaV
        Gnw_prev = net.calculate_network_conductance(nodeA=sourcenode_list[0], nodeB=groundnode_list[0], V_read=.1)
        print(Vbias)
        while flag == 0:
            V_list = np.reshape([Vbias], (1, 1))
            net.update_edge_weights(delta_t=delta_t)
            _ = net.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=0)
            Gnw_rec.append(- net.source_current[0] / Vbias)
            net_entropy.append(net.net_entropy_from_conductances(net.Condmat))
            V_rec.append(Vbias)
            if Gnw_prev == Gnw_rec[-1]:
                # eq_V.append(Vbias)
                # eq_G.append(net_conductance[-1])
                flag = 1
            else:
                 Gnw_prev = Gnw_rec[-1]

    np.save(file=save_path + 'V_rec.npy', arr=V_rec)
    np.save(file=save_path + 'Gnw_rec.npy', arr=Gnw_rec)
    np.save(file=save_path + 'net_entropy.npy', arr=net_entropy)


def save_evolution(rows, cols, dictionary):
    mem_p_loc = copy.deepcopy(mem_param)
    net_param_loc = copy.deepcopy(net_param)
    net_param_loc.rows = rows
    net_param_loc.cols = cols
    mem_p_loc.g_max = mem_p_loc.g_min * dictionary.ratio
    net_param_loc.frac_of_static_elements = 1 - dictionary.frac_mem_el
    net_param_loc.seed = None

    save_path = dictionary.save_path
    utils.ensure_dir(save_path)

    delta_t = 1 / sim_param.sampling_rate
    # net = MemNet(mem_param=mem_p_loc, net_param=net_param_loc, gnd=gnd, src=src, diag=diag)
    # run(net=net, groundnode_list=gnd, sourcenode_list=src, delta_t=delta_t, save_path=save_path, dictionary=dictionary)

    dict_rob = edict({'batch': dictionary.batch, 'frac_mem_el': dictionary.frac_mem_el,
                      'ratio': dictionary.ratio, 'delta_t': delta_t})  # + mem_p_loc + net_param_loc + sim_param
    dict_3 = {k: v for d in (dict_rob, mem_p_loc, net_param_loc, sim_param) for k, v in d.items()}
    utils.pickle_save(filename='{:s}info.pickle'.format(save_path), obj=dict_3)


    #################
    V_rec = np.load(file=save_path + 'V_rec.npy')
    Gnw_rec = np.load(file=save_path + 'Gnw_rec.npy')

    t = np.arange(start=0, stop=len(V_rec) * delta_t, step=delta_t)
    fig, ax = plt.subplots(figsize=(10, 8))
    # make a plot
    ax.plot(t, Gnw_rec, color="red", marker=".", markersize=3.8, linewidth=2.3,
            label=r'$\mathbf{G_{max}/G_{min}}=$' + '{:.0e}'.format(dictionary.ratio)
            )
    set_ticks_label(ax=ax, ax_type='y', num=10, data=Gnw_rec,
                    ax_label='G [a.u]',
                    valfmt="{x:.2e}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'red'})
    set_ticks_label(ax=ax, ax_type='x', num=10, data=t,
                    ax_label='time [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'})

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(t, V_rec, color="blue", marker=".", markersize=3.8)
    # ax2.set_ylabel(r"$\mathbf{V [a.u]}$", color="blue", fontsize=14)
    set_ticks_label(ax=ax2, ax_type='y', num=10, data=V_rec,
                    ax_label='V [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'blue'})
    ax.set_title('p={:.2f}'.format(dictionary.frac_mem_el), fontweight='bold', fontsize='x-large')
    set_legend(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + 'time_evolution.png')

    #############
    fig, ax = plt.subplots(figsize=(10, 8))
    # make a plot
    ax.plot(V_rec, Gnw_rec, color="red", marker=".", markersize=3.8, linewidth=2.3,
            label=r'$\mathbf{G_{max}/G_{min}}=$' + '{:.0e}'.format(dictionary.ratio)
            )
    set_ticks_label(ax=ax, ax_type='y', num=10, data=Gnw_rec,
                    ax_label='G [a.u]',
                    valfmt="{x:.2e}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'red'})
    set_ticks_label(ax=ax, ax_type='x', num=10, data=V_rec,
                    ax_label='V [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'blue'})
    ax.set_title('p={:.2f}'.format(dictionary.frac_mem_el), fontweight='bold', fontsize='x-large')
    set_legend(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + 'Gnw_vs_Voltage_evolution.png')

    #########
    V_rec = np.array(V_rec)
    ind = np.where(V_rec[:-1] != V_rec[1:])[0]
    v_eq = V_rec[ind]
    Gnw_eq = np.array(Gnw_rec)[ind]
    fig, ax = plt.subplots(figsize=(10, 8))
    # make a plot
    ax.plot(v_eq, Gnw_eq, color="red", marker=".", markersize=3.8, linewidth=2.3,
            label=r'$\mathbf{G_{max}/G_{min}}=$' + '{:.0e}'.format(dictionary.ratio)
            )
    set_ticks_label(ax=ax, ax_type='y', num=10, data=Gnw_eq,
                    ax_label='G [a.u]',
                    valfmt="{x:.2e}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'red'})
    set_ticks_label(ax=ax, ax_type='x', num=10, data=v_eq,
                    ax_label='V [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'blue'})
    ax.set_title('p={:.2f}'.format(dictionary.frac_mem_el), fontweight='bold', fontsize='x-large')
    set_legend(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + 'Gnw_vs_Voltage.png')

    #################

    ######### ENTROPY
    ylabel = r'$\mathbf{\sigma}$'
    valfmt = "{x:.2f}"
    net_entropy = np.load(file=save_path + 'net_entropy.npy') / np.log(2*(rows**2) - 2*rows)
    data_y = net_entropy

    V_rec = np.array(V_rec)
    ind = np.where(V_rec[:-1] != V_rec[1:])[0]
    v_eq = V_rec[ind]
    data_y_eq = np.array(net_entropy)[ind]

    fig, ax = plt.subplots(figsize=(10, 8))
    # make a plot
    ax.plot(v_eq, data_y_eq, color="red", marker=".", markersize=3.8, linewidth=2.3,
            label=r'$\mathbf{G_{max}/G_{min}}=$' + '{:.0e}'.format(dictionary.ratio)
            )
    set_ticks_label(ax=ax, ax_type='y', num=10, data=data_y_eq,
                    ax_label=ylabel,
                    valfmt=valfmt,
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'red'})
    set_ticks_label(ax=ax, ax_type='x', num=10, data=v_eq,
                    ax_label='V [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'blue'})
    ax.set_title('p={:.2f}'.format(dictionary.frac_mem_el), fontweight='bold', fontsize='x-large')
    set_legend(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + 'Ent_vs_Voltage.png')

    ######
    t = np.arange(start=0, stop=len(V_rec) * delta_t, step=delta_t)

    fig, ax = plt.subplots(figsize=(10, 8))
    # make a plot
    ax.plot(t, data_y, color="red", marker=".", markersize=3.8, linewidth=2.3,
            label=r'$\mathbf{G_{max}/G_{min}}=$' + '{:.0e}'.format(dictionary.ratio)
            )
    set_ticks_label(ax=ax, ax_type='y', num=10, data=data_y,
                    ax_label=ylabel,
                    valfmt=valfmt,
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'red'})
    set_ticks_label(ax=ax, ax_type='x', num=10, data=t,
                    ax_label='time [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'})

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(t, V_rec, color="blue", marker=".", markersize=3.8)
    # ax2.set_ylabel(r"$\mathbf{V [a.u]}$", color="blue", fontsize=14)
    set_ticks_label(ax=ax2, ax_type='y', num=10, data=V_rec,
                    ax_label='V [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'blue'})
    ax.set_title('p={:.2f}'.format(dictionary.frac_mem_el), fontweight='bold', fontsize='x-large')
    set_legend(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + 'Ent_time_evolution.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    # make a plot
    ax.plot(V_rec, data_y, color="red", marker=".", markersize=3.8, linewidth=2.3,
            label=r'$\mathbf{G_{max}/G_{min}}=$' + '{:.0e}'.format(dictionary.ratio)
            )
    set_ticks_label(ax=ax, ax_type='y', num=10, data=data_y,
                    ax_label=ylabel,
                    valfmt=valfmt,
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'red'})
    set_ticks_label(ax=ax, ax_type='x', num=10, data=V_rec,
                    ax_label='V [a.u]',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'blue'})
    ax.set_title('p={:.2f}'.format(dictionary.frac_mem_el), fontweight='bold', fontsize='x-large')
    set_legend(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path + 'Ent_vs_Voltage_evolution.png')

    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='OutputGridAdiabaticII', type=str)
    parser.add_argument('-lin_size', '--linear_size', default=21, type=int)
    parser.add_argument('-Vb', '--Vbias', default=20, type=float)
    parser.add_argument('-b_start', '--batch_start', default=0, type=int)
    parser.add_argument('-b_end', '--batch_end', default=1, type=int)
    parser.add_argument('-diag', '--random_diagonals', default=0, type=int)
    # parser.add_argument('-ent_f', '--entropy_func', default='SpecExpEnt', type=str, help='VNGE_FINGER')
    args = parser.parse_args()

    diag = args.random_diagonals
    root = abspath(join(".", pardir))
    # root = abspath(".")
    print(root)
    # sys.exit(1)
    time_list = []
    edge_list = []
    start_time = time.time()

    rows = args.linear_size
    cols = args.linear_size
    src = [(rows - 1)//2]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [rows**2 - (rows - 1) // 2 - 1]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

    save_path_sim = join(root,
                         '{:}/L{:d}/NetSim_StartEnd/{:}/'.format(args.save_path, args.linear_size,
                                                                 net_param.weight_init))
    utils.ensure_dir(save_path_sim)

    # Save Evolution Start and End
    print('Starting simulations...\n\n')
    frac_of_mem_list = np.round([1], decimals=2) #np.round(np.arange(.1, 1.1, .1, dtype=np.float16), decimals=2)
    ratio_list = [2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6] #, 1e7, 1e8]
    # ratio_list = [1e2]
    # frac_list = np.round(np.arange(0, .5, .1, dtype=np.float16), decimals=2)

    # ratio_list = [1e3, 1e4, 1e5, 1e6]
    # frac_of_mem_list = np.round(np.arange(.5, 1, .1, dtype=np.float16), decimals=2)

    l = [edict({'ratio': r, 'frac_mem_el': f, 'batch': b,
                'save_path': '{:s}p{:.2f}/ratio{:.1e}/batch{:04d}/'.format(save_path_sim, f, r, b)
                })
         for b in range(args.batch_start, args.batch_end)
         for f in frac_of_mem_list
         for r in ratio_list]

    # for item in tqdm(l):
    #     save_evolution(rows=rows, cols=cols, dictionary=item, save_fold=save_path_sim)

    ratio_lab_bold = ['2', '10', '100', r'$\mathbf{1x10^3}$', r'$\mathbf{1x10^4}$', r'$\mathbf{1x10^5}$',
                      r'$\mathbf{1x10^6}$']
    ratio_lab = ['2', '10', '100', r'$1x10^3$', r'$1x10^4$', r'$1x10^5$', r'$1x10^6$']

    r_l = np.zeros(len(ratio_list))
    dGdV_list = []
    dEdV_list = []
    for key, ylabel, valfmt, name_, norm in zip(['Gnw_rec.npy', 'net_entropy.npy'],
                                                # ['Conductance\n'+r'$\mathbf{\frac{(G_{nw}-G_{nw}(V=0))G_{min}}{G_{max}}}$', #+ ' [a.u]',
                                                 ['Conductance\n' +r'$\mathbf{G_{nw}}$'+' [a.u.]',
                                                     'Entropy\n'+r'$\mathbf{\sigma}$'],
                                                ["{x:.1e}", "{x:.1f}"],
                                                ['Gnw', 'Ent'],
                                                [False, False]):
        save_path_figures = join(root,
                                 '{:s}/L{:d}_newNorm/Figures/p{:.2f}/'.format(args.save_path, args.linear_size,
                                                                            frac_of_mem_list[0]))
        print('Save to:\n\t{:s}'.format(save_path_figures))
        utils.ensure_dir(save_path_figures)

        max_list = list()
        min_list = list()

        fig, ax = plt.subplots(figsize=(8, 6))
        ind_for_loop = 0
        for item in tqdm(l):
            V_rec = np.load(file=item.save_path + 'V_rec.npy')
            Gnw_rec = np.load(file=item.save_path + key)
            ind = np.where(V_rec[:-1] != V_rec[1:])[0]
            v_eq = V_rec[ind]
            Gnw_eq = np.array(Gnw_rec)[ind]
            dGdv = np.zeros((len(ratio_list), len(v_eq)))
            dEdv = np.zeros((len(ratio_list), len(v_eq)))

            if norm:
                Gnw_eq_norm = (Gnw_eq - Gnw_eq.min()) / (item.ratio)
                y_data = Gnw_eq_norm
            else:
                y_data = Gnw_eq / np.log(2*(rows**2) - 2*rows)
            max_list.append(y_data.max())
            min_list.append(y_data.min())

            r_l[ind_for_loop] = item.ratio
            if 'Gnw' in name_:
                ax.semilogy(v_eq, y_data, marker=".", markersize=3.8, linewidth=2.3,
                        # label='{:.0e}'.format(item.ratio)
                        label = '{:s}'.format(ratio_lab[ratio_list.index(item.ratio)])

                )
                dGdV_list.append( take_1rst_derivative(x=v_eq, y=y_data))

            else:
                ax.plot(v_eq, y_data, marker=".", markersize=3.8, linewidth=2.3,
                        # label='{:.0e}'.format(item.ratio)
                        label='{:s}'.format(ratio_lab[ratio_list.index(item.ratio)])
                        )
                # dEdV_list.append( take_2nd_derivative(x=v_eq, y=y_data)[0] )
                dEdV_list.append(take_1rst_derivative(x=v_eq, y=y_data))

            ind_for_loop = ind_for_loop +1
        if 'Gnw' in name_:
            set_ticks_label(ax=ax, ax_type='y',
                            num=2,
                            # data=np.reshape([max_list, min_list], -1),
                            data=[0, 1 + math.floor(math.log(np.max(max_list), 10))],
                            add_ticks=[1e-4, 1e-3, 1e-2, 1e-1],
                            ax_label=ylabel,
                            valfmt=valfmt,
                            fontdict_ticks_label={'size': 'large'},
                            fontdict_label={'color': 'black'}, scale='log')
        else:
            set_ticks_label(ax=ax, ax_type='y',
                            num=5,
                            data=np.reshape([max_list, min_list], -1),
                            ax_label=ylabel,
                            valfmt=valfmt,
                            fontdict_ticks_label={'size': 'large'},
                            fontdict_label={'color': 'black'})

        set_ticks_label(ax=ax, ax_type='x', num=5, data=v_eq,
                        ax_label='V [a.u.]\nVoltage input',
                        valfmt="{x:.1f}",
                        fontdict_ticks_label={'size': 'large'},
                        fontdict_label={'color': 'black'})
        # ax.set_title('p={:.2f}'.format(item.frac_mem_el), fontweight='bold', fontsize='x-large', family='Courier')
        set_legend(ax=ax, title='Interaction strength\n'+ r'$\mathbf{G_{max}/G_{min}}$')
        plt.tight_layout()
        plt.savefig(save_path_figures + '{:s}_vs_Voltage.svg'.format(name_), format='svg', dpi=1200)


##################################################################################################
    save_path_figures = join(root,
                             '{:s}/L{:d}_newNorm/Figures/p{:.2f}/'.format(args.save_path, args.linear_size, frac_of_mem_list[0]))
    utils.ensure_dir(save_path_figures)

    ratio_lab_bold=['2', '10', '100', r'$\mathbf{1x10^3}$', r'$\mathbf{1x10^4}$', r'$\mathbf{1x10^5}$', r'$\mathbf{1x10^6}$']
    ratio_lab = ['2', '10', '100', r'$1x10^3$', r'$1x10^4$', r'$1x10^5$', r'$1x10^6$']

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, dGdv in enumerate(dGdV_list):
        x = 0.5 * (v_eq[:-1] + v_eq[1:])
        index_to_plot = dGdv > 1e-11
        # x = v_eq[1:][index_to_plot]
        x = x[index_to_plot]
        y = dGdv[index_to_plot]
        ax.semilogy(x, y, marker=".", markersize=3.8, linewidth=2.3,
                label='{:s}'.format(ratio_lab[i]))
        # ax.semilogy(v_eq[1:], dGdv, marker=".", markersize=3.8, linewidth=2.3, label='{:.0e}'.format(r_l[i]))
    set_ticks_label(ax=ax, ax_type='y',
                    num=2,
                    # data=np.reshape([max_list, min_list], -1),
                    data=[0, 1 + math.floor(math.log(np.max(dGdV_list), 10))],
                    add_ticks=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
                    ax_label= r'$\mathbf{dG_{nw}/dV}$'+' [a.u.]',
                    valfmt=valfmt,
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'}, scale='log')
    set_ticks_label(ax=ax, ax_type='x', num=5, data=x,
                    ax_label='V [a.u.]\nVoltage input',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'})
    # ax.set_title('p={:.2f}'.format(item.frac_mem_el), fontweight='bold', fontsize='x-large', family='Courier')
    plt.tight_layout()
    set_legend(ax=ax, title='Interaction strength\n' + r'$\mathbf{G_{max}/G_{min}}$', loc=4)
    plt.savefig(save_path_figures + '{:s}_vs_Voltage.svg'.format('dGdV'), format='svg', dpi=1200)
    plt.show()

    #######################################################################################
    ## Plot voltage for maximum derivative of G
    v_for_maxdGdV=np.zeros(len(r_l))
    for i, dGdv in enumerate(dGdV_list):
        x = 0.5 * (v_eq[:-1] + v_eq[1:])
        index_to_plot = dGdv > 1e-11
        max_ind = np.max(dGdv[index_to_plot])
        v_for_maxdGdV[i] = x[dGdv==np.max(dGdv[index_to_plot])][0]
        # x = v_eq[1:][index_to_plot]
        # x = x[index_to_plot]
        # y = np.max(dGdv[index_to_plot])

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for x, y in zip(range(len(r_l)), v_for_maxdGdV):
        ax1.scatter(x, y, marker="o")
    set_ticks_label(ax=ax1, ax_type='y', num=5, data=v_for_maxdGdV,
                    ax_label='V [a.u.]\nVoltage input',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'})
    set_ticks_label(ax=ax1, ax_type='x', num=5, data=v_for_maxdGdV,
                    ticks=np.arange(len(r_l)),
                    # tick_lab=['2', '10', '100', '1x10^3', '1x10^4', '1x10^5', '1x10^6'],
                    tick_lab= ratio_lab_bold,
                    ax_label='Interaction strength\n', # r'$\mathbf{G_{max}/G_{min}}$',
                    # valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'})
    plt.tight_layout()
    plt.savefig(save_path_figures + '{:s}_vs_Voltage.svg'.format('max_dGdV'), format='svg', dpi=1200)
    plt.show()
    # ax1.scatter(range(len(r_l)), v_for_maxdGdV, marker="o")
    # plt.show()
    # set_ticks_label(ax=ax, ax_type='x', num=5, data=x,
    #                 ax_label='',
    #                 valfmt="{x:.1f}",
    #                 fontdict_ticks_label={'size': 'large'},
    #                 fontdict_label={'color': 'black'},
    #                 add_ticks=)

    ############################################################
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, dGdv in enumerate(dEdV_list):
        x = 0.5 * (v_eq[:-1] + v_eq[1:])
        # index_to_plot = dGdv > 1e-11
        # ax.semilogy(v_eq[1:][index_to_plot], dGdv[index_to_plot], marker=".", markersize=3.8, linewidth=2.3,
        #         label='{:.0e}'.format(r_l[i])
        #         )
        # ax.plot(v_eq[2:], dGdv, marker=".", markersize=3.8, linewidth=2.3, label='{:.0e}'.format(r_l[i]))
        ax.plot(x, dGdv, marker=".", markersize=3.8, linewidth=2.3, label='{:s}'.format(ratio_lab[i]))
    set_ticks_label(ax=ax, ax_type='y', num=5, data=dEdV_list,
                    ax_label=r'$\mathbf{dE/dV}$' + ' [a.u.]',
                    valfmt="{x:.2f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'})
    set_ticks_label(ax=ax, ax_type='x', num=5, data=x,
                    ax_label='V [a.u.]\nVoltage input',
                    valfmt="{x:.1f}",
                    fontdict_ticks_label={'size': 'large'},
                    fontdict_label={'color': 'black'})
    # ax.set_title('p={:.2f}'.format(item.frac_mem_el), fontweight='bold', fontsize='x-large', family='Courier')
    set_legend(ax=ax, title='Interaction strength\n' + r'$\mathbf{G_{max}/G_{min}}$')
    plt.tight_layout()
    plt.savefig(save_path_figures + '{:s}_vs_Voltage.svg'.format('dEdV'), format='svg', dpi=1200)
    plt.show()
    a=0


