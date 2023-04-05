from script_Opt.Class_SciPySparse.mem_parameters import mem_param, sim_param, volt_param, train_param, \
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
from create_plots_entropy import net_entropy_from_conductances

def plot_evol_adiab(rows, cols, dictionary, save_fold, plot_second_row='ent'):
    mem_p_loc = copy.deepcopy(mem_param)
    net_param_loc = copy.deepcopy(net_param)
    net_param_loc.rows = rows
    net_param_loc.cols = cols
    mem_p_loc.g_max = mem_p_loc.g_min * dictionary.ratio
    net_param_loc.frac_of_static_elements = dictionary.frac_of_static_elements
    net_param_loc.seed = None

    save_path = '{:s}p{:.2f}/ratio{:.1e}/batch{:04d}/'.format(save_fold,
                                                                 1-net_param_loc.frac_of_static_elements,
                                                                 dictionary.ratio,
                                                                 dictionary.batch)
    utils.ensure_dir(save_path)
    delta_t = 1 / sim_param.sampling_rate
    # sim_param.T = .5 #sim_param.T*2
    sourcenode_list = src
    groundnode_list = gnd
    edge_voltage_list = []
    print(sim_param.sampling_rate)
    inputsignal = ControlSignal(sources=src, sim_param=sim_param, volt_param=volt_param)
    inputsignal.t_list = np.arange(0, sim_param.T + delta_t, delta_t)
    inputsignal.V_list[0] = [dictionary.Vbias] * len(inputsignal.t_list)
    inputsignal.V_list[0][:2] = .1

    # edge_voltage_list = []
    # V_rec = []
    # Gnw_rec = []
    # net_entropy = []
    # eq_G = []
    # eq_V = []
    # Vbias = 0
    # deltaV = .2
    #
    # net = MemNet(mem_param=mem_p_loc, net_param=net_param_loc, gnd=gnd, src=src, diag=args.diagonals)
    # coordinates = [(node, (feat['coord'])) for node, feat in net.G.nodes(data=True)]
    # _ = net.MVNA(groundnode_list=gnd, sourcenode_list=src, V_list=np.reshape([.1], (1, 1)), t=0)
    # edge_voltage_list.append(deepcopy(net.dVmat.data))
    #
    # while Vbias <= args.Vbias:
    #     flag = 0
    #     Vbias += deltaV
    #     Gnw_prev = net.calculate_network_conductance(nodeA=sourcenode_list[0], nodeB=groundnode_list[0], V_read=.1)
    #     print(Vbias)
    #     while flag == 0:
    #         V_list = np.reshape([Vbias], (1, 1))
    #         net.update_edge_weights(delta_t=delta_t)
    #         _ = net.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=0)
    #         Gnw_rec.append(- net.source_current[0] / Vbias)
    #         net_entropy.append(net.net_entropy_from_conductances(net.Condmat))
    #         V_rec.append(Vbias)
    #         if Gnw_prev == Gnw_rec[-1]:
    #             eq_V.append(Vbias)
    #             eq_G.append(Gnw_rec[-1])
    #             edge_voltage_list.append(deepcopy(net.dVmat.data))
    #             flag = 1
    #         else:
    #             Gnw_prev = Gnw_rec[-1]

    # np.save(arr=edge_voltage_list, file=save_path + 'all_edges.npy')
    # np.save(arr=eq_V, file=save_path + 'eq_V.npy')
    # np.save(arr=eq_G, file=save_path + 'eq_G.npy')

    ######################################################################################
    edge_voltage_list = np.load(save_path + 'all_edges.npy')[:-1]
    eq_V= np.load(save_path + 'eq_V.npy')[:-1]
    applied_V = np.arange(0, 20.2, eq_V[0])  # [0]+list(np.unique(V_rec))
    applied_V[0] = .1
    eq_G = np.load(save_path + 'eq_G.npy')[:-1]

    edge_voltage_list = np.abs(edge_voltage_list)
    ######################################## Hist edge DeltaV  ################################################
    utils.ensure_dir(save_path + 'Hist_edgeV/')

    for i in range(len(edge_voltage_list)):
        weights = edge_voltage_list[i]
        # w_norm = (weights - weights.min()) / (weights.max() - weights.min())
        # plt.hist(w_norm, bins=20, alpha=.7, label=str(i) +', ' + str(entropy_net[i]))
        fig, ax = plt.subplots()
        ax.hist(weights, bins=20, alpha=.7, label=str(i))  # + ', ' + str(entropy_net[i]))
        # pdf, _, bins = utils.empirical_pdf_and_cdf(w_norm, bins=20)
        # plt.hist(bins[1:], pdf, alpha=.6, label='t={:f}'.format(inputsignal.t_list[i]))
        plt.legend()
        plt.title(r'$V_{inp}$' + '={:.2f}'.format(applied_V[i]) + ', ' + r'$<ΔV>_E$' + '={:.3}'.format(np.mean(weights)))
        plt.savefig(save_path + 'Hist_edgeV/' + 'histogram_weights_{:d}.png'.format(i), format='png')  # , dpi=1200)
        plt.close()

    ##################################################################################################################
    mean_edge_volt = np.mean(edge_voltage_list, axis=1)

    plt.plot(eq_V, (eq_G-eq_G[0])*np.min(eq_G)/(np.max(eq_G)-np.min(eq_G)))
    plt.ylabel('G_norm')
    plt.xlabel('Voltage input')
    plt.savefig(save_path + 'G.png')
    plt.show()
    plt.close()


    fig, ax = plt.subplots()
    # # make a plot
    ax.scatter(applied_V, mean_edge_volt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{<ΔV>_E}$', data=mean_edge_volt, num=5, valfmt="{x:.3f}")
    set_ticks_label(ax=ax, ax_type='x', ax_label='V [a.u.]\nVoltage input', data=applied_V, num=5, valfmt="{x:.1f}")
    plt.tight_layout()
    plt.savefig(save_path + 'Hist_edgeV/mean_edge_volt_vs_applV.png')
    plt.show()
    plt.close()

    max_volt = np.max(edge_voltage_list, axis=1)
    fig, ax = plt.subplots()
    # # make a plot
    ax.scatter(applied_V, max_volt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{max\{ΔV\}_E}$', data=np.max(edge_voltage_list, axis=1), num=5)
    set_ticks_label(ax=ax, ax_type='x', ax_label='V [a.u.]\nVoltage input', data=applied_V, num=5, valfmt="{x:.1f}")
    plt.tight_layout()
    plt.savefig(save_path + 'Hist_edgeV/max_edge_volt_vs_applV.png')
    plt.show()
    plt.close()

    ######################### K_p and K_d ########################################################################
    save_kp_hist = save_path + 'kp/'
    save_kd_hist = save_path + 'kd/'
    save_firstTerm_hist = save_path + 'firstTerm/'
    utils.ensure_dir(save_kp_hist)
    utils.ensure_dir(save_kd_hist)
    utils.ensure_dir(save_firstTerm_hist)

    volt = mean_edge_volt
    xlabel = r'$\mathbf{<ΔV>_E}$'
    kp = mem_p_loc.kp0 * np.exp(mem_p_loc.eta_p * volt)
    kd = mem_p_loc.kd0 * np.exp(-mem_p_loc.eta_d * volt)
    first_term = kp / (kd + kp)

    valfmt = "{x:.3f}"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volt, first_term, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label=xlabel, data=volt, valfmt=valfmt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{\frac{k_p}{k_p + k_d}}$', data=first_term, valfmt='{x:.2f}')
    plt.tight_layout()
    # plt.savefig(save_firstTerm_hist + 'first_term.png')
    plt.savefig(save_firstTerm_hist + 'first_term.svg', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volt, kp, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label=xlabel, data=volt, valfmt=valfmt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{k_p}$', data=kp, valfmt='{x:.3f}')
    plt.tight_layout()
    # plt.savefig(save_kp_hist + 'kp.png')
    plt.savefig(save_kp_hist + 'kp.svg', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volt, kd, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label=xlabel, data=volt, valfmt=valfmt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{k_d}$', data=kd, valfmt='{x:.2f}')
    plt.tight_layout()
    # plt.savefig(save_kd_hist + 'kd.png')
    plt.savefig(save_kd_hist + 'kd.svg', dpi=1200)
    plt.show()

    ############################################ Mean k_p and Mean k_d  #################################################################

    volt = np.abs(edge_voltage_list)

    kp_mean = np.mean(mem_p_loc.kp0 * np.exp(mem_p_loc.eta_p * volt), axis=1)
    kd_mean = np.mean(mem_p_loc.kd0 * np.exp(-mem_p_loc.eta_d * volt), axis=1)
    # first_term_mean = kp / (kd + kp)
    first_term_mean = mem_p_loc.kp0 * np.exp(mem_p_loc.eta_p * volt)/(mem_p_loc.kd0 * np.exp(-mem_p_loc.eta_d * volt) + mem_p_loc.kp0 * np.exp(mem_p_loc.eta_p * volt))
    first_term_mean = np.mean(first_term_mean, axis=1)
    xlabel = 'Voltage input [a.u.]'
    valfmt = "{x:.0f}"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(applied_V, first_term_mean, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label=xlabel, data=applied_V, valfmt=valfmt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{<\frac{k_p}{k_p + k_d}>}$', data=first_term_mean, valfmt='{x:.2f}')
    plt.tight_layout()
    plt.savefig(save_firstTerm_hist + 'mean_first_term.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(applied_V, kp_mean, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label=xlabel, data=applied_V, valfmt=valfmt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{<k_p>}$', data=kp_mean, valfmt='{x:.1e}')
    plt.tight_layout()
    plt.savefig(save_kp_hist + 'mean_kp.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(applied_V, kd_mean, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label=xlabel, data=applied_V, valfmt=valfmt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{<k_d>}$', data=kd_mean, valfmt='{x:.1f}')
    plt.tight_layout()
    plt.savefig(save_kd_hist + 'mean_kd.png')
    plt.show()

    a=0
    ####################################################################################################################
    ####################################### Save Histograms kp, kd #######################################################################

    list_firstTerm_mean = []

    for i, edge_volt in enumerate(edge_voltage_list):
        volt = np.abs(edge_volt)
        kp = mem_p_loc.kp0 * np.exp(mem_p_loc.eta_p * volt)
        kd = mem_p_loc.kd0 * np.exp(-mem_p_loc.eta_d * volt)
        first_term = kp / (kd + kp)
        list_firstTerm_mean.append(first_term.mean())

        fig, ax = plt.subplots()
        ax.hist(kp, bins=20, alpha=.7, label=str(i))  # + ', ' + str(entropy_net[i]))
        # pdf, _, bins = utils.empirical_pdf_and_cdf(w_norm, bins=20)
        # plt.hist(bins[1:], pdf, alpha=.6, label='t={:f}'.format(inputsignal.t_list[i]))
        plt.legend()
        plt.title(r'$V_{inp}$' + '={:.2f}'.format(applied_V[i]) +', ' + r'$<k_p>$'+'={:.3}'.format(np.mean(kp)))
        plt.xlabel(r'$k_p$')
        plt.ylabel('#')
        plt.tight_layout()
        plt.savefig(save_kp_hist + 'histogram_weights_{:d}.png'.format(i), format='png')  # , dpi=1200)
        plt.close()

        fig, ax = plt.subplots()
        ax.hist(kd, bins=20, alpha=.7, label=str(i))  # + ', ' + str(entropy_net[i]))
        plt.legend()
        plt.ylabel('#')
        plt.xlabel(r'$k_d$')
        plt.title(r'$V_{inp}$' + '={:.2f}'.format(applied_V[i]) + ', ' + r'$<k_d>$'+'={:.3}'.format(np.mean(kd)))
        plt.tight_layout()
        plt.savefig(save_kd_hist + 'histogram_weights_{:d}.png'.format(i), format='png')  # , dpi=1200)
        plt.close()

        fig, ax = plt.subplots()
        ax.hist(first_term, bins=20, alpha=.7, label=str(i))  # + ', ' + str(entropy_net[i]))
        plt.legend()
        plt.xlabel(r'$\mathbf{\frac{k_p}{k_p + k_d}}$')
        plt.ylabel('#')
        plt.title(r'$V_{inp}$' + '={:.2f}'.format(applied_V[i]) +', '+ r'$<\frac{k_p}{k_p + k_d}>$'+'={:.3}'.format(np.mean(first_term)))
        plt.tight_layout()
        plt.savefig(save_firstTerm_hist + 'histogram_weights_{:d}.png'.format(i), format='png')  # , dpi=1200)
        plt.close()


    a=0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', #default='OutputGridAdiabatic_GoodOC', type=str)
                        default='OutputGridAdiabaticPlot_kpkd', type=str)
    parser.add_argument('-Vb', '--Vbias',  # default='OutputGridAdiabatic_GoodOC', type=str)
                        default=20, type=str)
    parser.add_argument('-diag', '--diagonals', default=0, type=int)
    # parser.add_argument('-crt_sim_data', '--create_sim_data', default=0, type=int)
    args = parser.parse_args()

    # if args.diagonals == 0:
    #     args.save_path = args.save_path + 'Grid'

    net_param = edict({'rows': 5,
                       'cols': 5,
                       'frac_of_static_elements': .3,
                       'weight_init': None,  # 'rand',
                       # 'weight_init': 'good_OC',
                       'seed': 2})

    sim_param = edict({'T': 1.5,  # 4e-3, # [s]
                       'sampling_rate': 500  # [Hz]  # =steps / T  # [Hz]
                       })

    root = abspath(join(".", pardir))
    # root = abspath(".")
    print(root)

    # time_list = []
    # edge_list = []
    # start_time = time.time()
    # rows = 11
    # cols = 11
    # src = [5]
    # gnd = [115]

    rows = 21
    cols = 21
    src = [(rows - 1)//2]
    gnd = [rows**2 - (rows - 1) // 2 - 1]
    # src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    # gnd = [430]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

    # rows = 20
    # cols = 20
    # src = [11]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    # gnd = [391]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

    labels = [(gnd[0], 'Gnd'), (src[0], 'Src')]

    batch = 1
    # frac_stat_el_list = np.round(np.arange(0, 1, .1, dtype=np.float16), decimals=1)
    # p = [.3, .4, .6, .7]
    p = [1]
    frac_stat_el_list = 1 - np.array(p)# [.2, .5, .8]
    # ratio_list = [2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]  # 1e9, 1e10]
    ratio_list = [1e6, 1e2, 1e1]
    # V_bias_list = np.arange(0.1, 16, 1)
    V_bias_list = [args.Vbias]
    l = [edict({'ratio': r, 'frac_of_static_elements': f, 'batch': b, 'Vbias': Vbias})
         for Vbias in V_bias_list
         for b in range(batch)
         for f in frac_stat_el_list
         for r in ratio_list]


    # l = [edict({'ratio': 10000, 'frac_of_static_elements': 0, 'batch': 0, 'Vbias': 3})]
    ent_rec = list()
    eff_c_rec = list()

    # key_second_row = 'G'
    key_second_row = 'ent'
    k=0
    for item in tqdm(l):
        if (item.frac_of_static_elements == 0) and (item.batch >= 1):
            pass
        else:
            save_path_sim = join(root,
                                 '{:}/SavePlotEvolN{:03d}/{:}/Vbias{:.1f}/'.format(args.save_path, rows, net_param.weight_init, item.Vbias))
            print(save_path_sim)
            utils.ensure_dir(save_path_sim)
            # eff_c, ent, t_list, V_list = plot_evol(rows=rows, cols=cols, dictionary=item, save_fold=save_path_sim, plot_second_row=key_second_row)
            plot_evol_adiab(rows=rows, cols=cols, dictionary=item, save_fold=save_path_sim, plot_second_row=key_second_row)

    #         ent_rec.append(ent)
    #         eff_c_rec.append(eff_c)
    # a=0
    # if key_second_row == 'ent':
    #     y_label_2ndrow = '$\mathbf{\sigma}$'
    #     data_2row = ent_rec
    # if key_second_row == 'G':
    #     y_label_2ndrow = '$\mathbf{G_{nw}^{norm}}$'
    #     data_2row = [np.array(a) for a in eff_c_rec]
    #     data_2row = [(a - a.min())/(a.max() - a.min()) for a in data_2row]
    #
    # list_dic = []
    # for item in l:
    #     if (item.frac_of_static_elements == 0) and (item.batch >= 1):
    #         pass
    #     else:
    #         list_dic.append(item)
    #
    # fig, ax = plt.subplots()
    # # make a plot
    # for i, it in enumerate(list_dic):
    #     ax.plot(t_list, data_2row[i],
    #              label='b{:d}, p{:.1f}, r{:.0e}'.format(it.batch, 1 - it.frac_of_static_elements, it.ratio))
    #                  # label='p{:.1f}, '.format(1-it.frac_of_static_elements) + r'$\mathbf{G_{max}/G_{min}}$'+'={:.0e}'.format(it.ratio))
    #                  # label=r'$\mathbf{G_{max}/G_{min}}$' + '={:.0e}'.format(it.ratio))
    #
    # ax.set_ylabel(y_label_2ndrow, color="black", fontsize=14)
    # ax.set_xlabel(r'$\mathbf{Time}$', color="black", fontsize=14)
    # # twin object for two different y-axis on the sample plot
    # ax2 = ax.twinx()
    # # make a plot with different y-axis using second axis object
    # ax2.plot(t_list, V_list[0], color="blue", marker=".", alpha=.6)
    # ax2.set_ylabel(r"$\mathbf{V}$", color="blue", fontsize=14)
    # ax.legend(loc=4)
    # plt.savefig(save_path_sim + key_second_row + '.png')
    # plt.close()
    #
    # fig, ax = plt.subplots()
    # # make a plot
    # for i, it in enumerate(list_dic):
    #     ax.plot(V_list[0], data_2row[i],
    #             # label='b{:d}, p{:.1f}, r{:.0e}'.format(it.batch, 1 - it.frac_of_static_elements, it.ratio))
    #             # label='p{:.1f}, '.format(1-it.frac_of_static_elements) + r'$\mathbf{G_{max}/G_{min}}$'+'={:.0e}'.format(it.ratio))
    #             # label=r'$\mathbf{G_{max}/G_{min}}$' + '={:.0e}'.format(it.ratio))
    #             label='b{:d}, p{:.1f}, r{:.0e}'.format(it.batch, 1 - it.frac_of_static_elements, it.ratio))
    #
    # ax.set_ylabel(y_label_2ndrow, color="black", fontsize=14)
    # ax.set_xlabel(r'$\mathbf{V}$', color="black", fontsize=14)
    # plt.legend()
    # plt.savefig(save_path_sim + key_second_row + '_vs_Volt.png')
    # # for i, it in enumerate(l):
    # #     plt.plot(t_list, data_2row[i],
    # #              label='b{:d}, p{:.1f}, r{:.0e}'.format(it.batch, 1 - it.frac_of_static_elements, it.ratio))
    # # plt.xlabel(r'$\mathbf{Time}$')
    # # plt.ylabel(y_label_2ndrow)
    # # plt.legend()
    # # plt.savefig(save_path_sim + key_second_row + '.png')
    #


# def plot_evol(rows, cols, dictionary, save_fold, plot_second_row='ent'):
#     mem_p_loc = copy.deepcopy(mem_param)
#     net_param_loc = copy.deepcopy(net_param)
#     net_param_loc.rows = rows
#     net_param_loc.cols = cols
#     mem_p_loc.g_max = mem_p_loc.g_min * dictionary.ratio
#     net_param_loc.frac_of_static_elements = dictionary.frac_of_static_elements
#     net_param_loc.seed = None
#
#     save_path = '{:s}p{:.2f}/ratio{:.1e}/batch{:04d}/'.format(save_fold,
#                                                                  1-net_param_loc.frac_of_static_elements,
#                                                                  dictionary.ratio,
#                                                                  dictionary.batch)
#     utils.ensure_dir(save_path)
#     # sim_param.T = .5 #sim_param.T*2
#     print(sim_param.sampling_rate)
#     inputsignal = ControlSignal(sources=src, sim_param=sim_param, volt_param=volt_param)
#     inputsignal.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)
#     inputsignal.V_list[0] = [dictionary.Vbias] * len(inputsignal.t_list)
#     inputsignal.V_list[0][:2] = .1
#
#     # a = np.hstack((np.linspace(.5, dictionary.Vbias, len(inputsignal.t_list) // 4), dictionary.Vbias*np.ones(len(inputsignal.t_list) - len(inputsignal.t_list) // 4)))
#     for j in [4]:
#         # a = np.hstack((np.linspace(.1, dictionary.Vbias, len(inputsignal.t_list) * j),
#         #                dictionary.Vbias * np.ones(len(inputsignal.t_list) - len(inputsignal.t_list) // j)))
#         a = np.linspace(.1, dictionary.Vbias, len(inputsignal.t_list))
#         inputsignal.V_list[0] = list(a)
#
#         net = MemNet(mem_param=mem_p_loc, net_param=net_param_loc, gnd=gnd, src=src, diag=args.diagonals)
#         coordinates = [(node, (feat['coord'])) for node, feat in net.G.nodes(data=True)]
#
#         node_voltage_list = []
#         edge_voltage_list = []
#         H_list = [[] for t in range(len(inputsignal.t_list))]
#         eff_cond = list()
#         eff_cond_2 = list()
#         # eff_cond_3 = list()
#         entropy_net = list()
#
#         h = net.MVNA(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=0)
#         H_list[0] = deepcopy(h)
#         node_voltage_list.append(deepcopy(net.node_Voltage))
#         edge_voltage_list.append(deepcopy(net.dVmat.data))
#
#         # import time
#         # start = time.process_time()
#         # eff_cond.append(1 / net.effective_resistance(nodeA=net.src[0], nodeB=net.gnd[0]))
#         # time_1 =  time.process_time() - start
#         #
#         # start =  time.process_time()
#         # eff_cond_2.append(-net.source_current[0] / inputsignal.V_list[0][0])
#         # time_2 =  time.process_time() - start
#
#         start = time.process_time()
#         eff_cond_2.append(net.calculate_network_conductance(nodeA=net.src[0], nodeB=net.gnd[0], V_read=3))
#         entropy_net.append(net_entropy_from_conductances(adj_matrix=h))
#         # time_3 =  time.process_time() - start
#
#         print('Start')
#         delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]
#         for t in range(1, len(inputsignal.t_list)):
#             print(t, '/', len(inputsignal.t_list))
#             net.update_edge_weights(delta_t=delta_t)
#             # conductance_mat_list.append(deepcopy(net.Condmat))
#             h = net.MVNA(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=t)
#             # H = nx.DiGraph(h.todense())
#             # nx.set_node_attributes(H, dict(coordinates), 'coord')
#             # nx.set_node_attributes(H, net.node_Voltage, 'V')
#
#             # eff_cond.append(1 / net.effective_resistance(nodeA=net.src[0], nodeB=net.gnd[0]))
#             # eff_cond_2.append(-net.source_current[0]/inputsignal.V_list[0][t])
#             eff_cond_2.append(net.calculate_network_conductance(nodeA=net.src[0], nodeB=net.gnd[0], V_read=3))
#             entropy_net.append(net_entropy_from_conductances(adj_matrix=h))
#             H_list[t] = deepcopy(h)
#             node_voltage_list.append(deepcopy(net.node_Voltage))
#             edge_voltage_list.append(deepcopy(net.dVmat.data))
#
#         # store_cond = eff_cond
#         # plt.scatter(inputsignal.t_list, store_cond)
#         # plt.plot(inputsignal.t_list, eff_cond_2, color='red')
#         # plt.plot(inputsignal.t_list, eff_cond_3, color='green')
#     # plt.plot(inputsignal.t_list, entropy_net, color='red')
#     derivative = np.diff(eff_cond_2)/delta_t
#     # plt.scatter(range(len(entropy_net)), entropy_net, color='red')
#
#     print('derivative always >0', (derivative > 0).sum() == len(derivative))
#     utils.ensure_dir(save_path+'Hist_edgeV/')
#     for i in range(len(edge_voltage_list)):
#         weights = edge_voltage_list[i]
#         # w_norm = (weights - weights.min()) / (weights.max() - weights.min())
#         # plt.hist(w_norm, bins=20, alpha=.7, label=str(i) +', ' + str(entropy_net[i]))
#         plt.hist(weights, bins=20, alpha=.7, label=str(i) )#+ ', ' + str(entropy_net[i]))
#         # pdf, _, bins = utils.empirical_pdf_and_cdf(w_norm, bins=20)
#         # plt.hist(bins[1:], pdf, alpha=.6, label='t={:f}'.format(inputsignal.t_list[i]))
#         plt.legend()
#         plt.savefig(save_path+'Hist_edgeV/'+'histogram_weights_{:d}.png'.format(i), format='png')#, dpi=1200)
#         plt.close()
#
#     # for i in [4, 49, len(H_list)-1]:
#     #     weights = H_list[i].data
#     #     w_norm = (weights - weights.min()) / (weights.max() - weights.min())
#     #     plt.hist(w_norm, bins=20, alpha=.7, label=str(i) +', ' + str(entropy_net[i]))
#     #     # pdf, _, bins = utils.empirical_pdf_and_cdf(w_norm, bins=20)
#     #     # plt.hist(bins[1:], pdf, alpha=.6, label='t={:f}'.format(inputsignal.t_list[i]))
#     # plt.legend()
#     # plt.savefig(save_path+'histogram_weights.svg', format='svg', dpi=1200)
#     # plt.close()
#     #
#     # fig = plt.figure('', figsize=(8,8))
#     # ax = fig.subplots(nrows=1, ncols=1)
#     # for k in [5, 100]:
#     #     ax.scatter(np.arange(net.number_of_nodes), node_voltage_list[k], label=inputsignal.t_list[k])
#     # set_legend(ax=ax, title='Time', ncol=1, loc=1)
#     # set_ticks_label(ax=ax, ax_label='Node id', data=range(net.number_of_nodes), ax_type='x', num=5, valfmt="{x:.0f}",
#     #                 # ticks=np.unique(np.round(x_loop, decimals=2)))
#     #                 )
#     # # set_ticks_label(ax=ax, ax_label='V', data=node_voltage_list, ax_type='x', num=5, valfmt="{x:d}",
#     # #                 ticks=np.unique(np.round(x_loop, decimals=2)))
#     #                 # )
#     # plt.tight_layout()
#     # plt.savefig(save_path + '/voltage_evolution.png')
#     # plt.close()
#     #
#     # if plot_second_row == 'ent':
#     #     y_label_2ndrow = '$\mathbf{\sigma}$'
#     #     data_2row = entropy_net
#     # if plot_second_row == 'G':
#     #     y_label_2ndrow = '$\mathbf{G_{nw}}$'
#     #     data_2row = eff_cond_2
#     #
#     # fig, ax = plt.subplots()
#     # # make a plot
#     # ax.plot(inputsignal.t_list, data_2row,
#     #             # label='b{:d}, p{:.1f}, r{:.0e}'.format(it.batch, 1 - it.frac_of_static_elements, it.ratio))
#     #             # label='p{:.1f}, '.format(1-it.frac_of_static_elements) + r'$\mathbf{G_{max}/G_{min}}$'+'={:.0e}'.format(it.ratio))
#     #             label=r'$\mathbf{G_{max}/G_{min}}$' + '={:.0e}'.format(dictionary.ratio))
#     # ax.set_ylabel(y_label_2ndrow, color="black", fontsize=14)
#     # ax.set_xlabel(r'$\mathbf{Time}$', color="black", fontsize=14)
#     # # twin object for two different y-axis on the sample plot
#     # ax2 = ax.twinx()
#     # # make a plot with different y-axis using second axis object
#     # ax2.plot(inputsignal.t_list, inputsignal.V_list[0], color="blue", marker=".", alpha=.6)
#     # ax2.set_ylabel(r"$\mathbf{V}$", color="blue", fontsize=14)
#     # plt.title(r'$\mathbf{G_{max}/G_{min}}$' + '={:.0e}'.format(dictionary.ratio))
#     # plt.savefig(save_path +'/' + key_second_row + '.png')
#     # plt.close()
#     #
#     # save_evol = save_path + plot_second_row + '/'
#     # utils.ensure_dir(save_evol)
#     # plot_H_evolution(H_Horizon=H_list,
#     #                  t_list_Horizon=inputsignal.t_list,
#     #                  numeric_label=False,
#     #                  eff_cond_list=data_2row,
#     #                  desired_eff_conductance=None,
#     #                  V_Horizon=inputsignal.V_list,
#     #                  node_labels=labels,
#     #                  y_label_2ndrow=y_label_2ndrow,
#     #                  src=src,  # G=net.G,
#     #                  coordinates=coordinates,
#     #                  number_of_plots=len(H_list) // 4,
#     #                  node_voltage_list=node_voltage_list,
#     #                  title='p={:.2f}, '.format(1-net_param_loc.frac_of_static_elements) +\
#     #                        r'$\mathbf{G_{max}/G_{min}}$'+'={:.0e}'.format(dictionary.ratio),
#     #                  save_path=save_evol,
#     #                  format='svg')
#     #
#     # # # time_list.append((time.time() - start_time))
#     # # make_gif(frame_folder=save_evol + '/H_Evolution', gif_name='memNet.gif', images_format='png',
#     # #          save_path=save_evol + '/')
#     #
#     # dict_rob = edict({'batch': dictionary.batch, 'ratio': dictionary.ratio,
#     #                   'Vbias': dictionary.Vbias})  # + mem_p_loc + net_param_loc + sim_param
#     # dict_3 = {k: v for d in (dict_rob, mem_p_loc, net_param_loc, sim_param) for k, v in d.items()}
#     # utils.pickle_save(filename='{:s}info.pickle'.format(save_path), obj=dict_3)
#     # # np.save(arr=inputsignal.t_list, file='{:s}t_list.np'.format(save_path))
#     return eff_cond_2, entropy_net, inputsignal.t_list, inputsignal.V_list
