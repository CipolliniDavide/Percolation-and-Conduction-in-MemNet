from script_Opt.Class_SciPySparse.mem_parameters import sim_param, volt_param, net_param, mem_param
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet, Measure
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.utils import utils, create_dataset, getListOfFiles
from script_Opt.Class_SciPySparse.anim_vis import plot_H_evolution
from script_Opt.Class_SciPySparse.make_gif import make_gif
from script_Opt.Class_SciPySparse.visualize import visualize
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
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph, save_npz, load_npz
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm, logm, eigvalsh
from scipy.linalg import eigvals, eigh
from easydict import EasyDict as edict
import itertools

from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend


def plot_voltage_sweep_percThr(df, save_path,
                               ratio_to_plot,
                               frac_mem_to_plot,
                               voltage,
                               normalize=False,
                               ylabel= 'I [mA]',
                               title=None,
                               save_name=None,
                               nrows=1,
                               ncols=2,
                               name_fig='', loc=2, figsize=(8, 6), add_yticks=[]):
    import matplotlib as mpl
    cmap = plt.cm.get_cmap("jet")
    norm = mpl.colors.Normalize(vmin=(frac_mem_to_plot).min(), vmax=(frac_mem_to_plot).max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    time_steps = len(voltage)
    ymax = 0
    ymin = 0
    fig = plt.figure('I-V', figsize=figsize)
    axes = fig.subplots(nrows=nrows, ncols=ncols)
    if (nrows==1) & (ncols==1):
        axes = [axes]
    for i, (r, ax) in enumerate(zip(ratio_to_plot, axes)):
        for f in sorted(frac_mem_elements_to_plot):
            folds = df.loc[(df['ratio'] == r) & (df['frac_of_mem_elements'] == f)]['path']
            curr_arr = np.zeros((len(folds), time_steps))
            for j, p in enumerate(np.array(folds)):
                if r < 10:
                    curr_arr[j] = np.load(p+'current.npy')*1000
                else:
                    curr_arr[j] = np.load(p + 'current.npy')
            y = curr_arr.mean(axis=0)
            e = curr_arr.std(ddof=0, axis=0)

            # mask = (voltage > 0)
            # mask[100:] = False
            # y = y[mask]
            # v = voltage[mask]
            # e = e[mask]

            if y.max() > ymax:
                ymax = y.max()
            if y.min() < ymin:
                ymin = y.min()
            ax.plot(voltage, y, label='{:.2f}'.format(f), color=cmap(norm(f)))#, marker='^',)
            # ax.fill_between(v, y - e, y + e, alpha=0.2, color=cmap(norm(f)))
            # ax.errorbar(voltage, y, yerr=e, fmt='.', ecolor=cmap(norm(f)), color=cmap(norm(f)))
        if i % ncols == 0:
            if r < 10:
                set_ticks_label(ax=ax, ax_label=ylabel, data=np.array([ymax, ymin]), ax_type='y', num=5,
                                valfmt="{x:.1f}")
            else:
                set_ticks_label(ax=ax, ax_label='I [A]', data=np.array([ymax, ymin]), ax_type='y', num=5, valfmt="{x:.1e}")
    if cols > 1:
        for ax in axes[-ncols:]:
            set_ticks_label(ax=ax, ax_label='V [V]', data=voltage, ax_type='x', num=3,
                            valfmt="{x:.1f}",
                            # ticks=),
                            )

    set_legend(ax=ax, title='p', ncol=1, loc=loc)
    # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
    # # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # cbar = plt.colorbar(sm, ticks=ratio_l, format=mpl.ticker.ScalarFormatter(),
    #                     shrink=1.0, fraction=0.1, pad=0, cax=cbar_ax)
    # set_ticks_label(ax=cbar.ax, ax_label=r'$\mathbf{C_{on}/C_{off}}$', ax_type='y', data=ratio_l, num=5, scale='log',
    #                 valfmt="{x:.0e}",
    #                 ticks=np.concatenate((np.array([2]), np.logspace(start=1, stop=10, num=4, endpoint=True))))

    # plt.title('{:s}'.format(args.entropy_func), size='xx-large', weight='bold')
    # name = args.entropy_func+'vs p'
    plt.tight_layout()
    plt.savefig(join(save_path + '{:s}_IVsweep_thresh.png'.format(name_fig)))
    plt.show()
    plt.close('all')


def voltagesweep(inputsignal, rows, cols, dictionary, save_fold):

    mem_p_loc = copy.deepcopy(mem_param)
    net_param_loc = copy.deepcopy(net_param)
    net_param_loc.rows = rows
    net_param_loc.cols = cols
    mem_p_loc.g_max = mem_p_loc.g_min * dictionary.ratio
    net_param_loc.frac_of_static_elements = dictionary.frac_of_static_elements
    net_param_loc.seed = None

    save_path = '{:s}frac{:.2f}/ratio{:.1e}/batch{:04d}/'.format(save_fold,
                                                                 net_param_loc.frac_of_static_elements,
                                                                 dictionary.ratio,
                                                                 dictionary.batch)

    utils.ensure_dir(save_path)

    net = MemNet(mem_param=mem_p_loc, net_param=net_param_loc, gnd=gnd, src=src, diag=args.diagonals)

    current = list()

    _ = net.MVNA(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=0)
    current.append(inputsignal.V_list[0][0] / net.effective_resistance(nodeA=net.src[0], nodeB=net.gnd[0]))

    delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]
    for t in range(1, len(inputsignal.t_list)):
        net.update_edge_weights(delta_t=delta_t)
        _ = net.MVNA(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=t)
        current.append(inputsignal.V_list[0][t] / net.effective_resistance(nodeA=net.src[0], nodeB=net.gnd[0]))

    np.save(file=save_path+'current.npy', arr=current)

    dict_rob = edict({'batch': dictionary.batch, 'ratio': dictionary.ratio})  # + mem_p_loc + net_param_loc + sim_param
    dict_3 = {k: v for d in (dict_rob, mem_p_loc, net_param_loc, sim_param) for k, v in d.items()}
    utils.pickle_save(filename='{:s}info.pickle'.format(save_path), obj=dict_3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='OutputGrid', type=str)
    parser.add_argument('-crt_data', '--create_sim_data', default=0, type=int)
    parser.add_argument('-diag', '--diagonals', default=0, type=int)
    parser.add_argument('-b_start', '--batch_start', default=0, type=int)
    parser.add_argument('-b_end', '--batch_end', default=100, type=int)
    parser.add_argument('-Vb', '--Vbias', default=10, type=float)
    parser.add_argument('-nc', '--ncols', default=21, type=int)
    parser.add_argument('-nr', '--nrows', default=21, type=int)
    args = parser.parse_args()

    root = abspath(join(".", pardir))
    # root = abspath(".")
    print(root)

    # rows = cols = int(21)
    # src = [11]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    # gnd = [391]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    #src = [2]
    # gnd = [22]
    rows = args.nrows
    cols = args.ncols
    src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [430]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

    # save_path_sim = join(root,
    #                      '{:}/PercolationTh/IVCurve/Net{:02d}x{:02d}/{:}/Vbias{:.1f}/'.format(args.save_path,
    #                                                                                           args.nrows,
    #                                                                                           args.ncols,
    #                                                                                           net_param.weight_init,
    #                                                                                           args.Vbias))

    save_path_sim = join(root,
                         '{:}/PercolationTh/IVCurve/{:}/Vbias{:.1f}/'.format(args.save_path,
                                                                             net_param.weight_init,
                                                                             args.Vbias))

    utils.ensure_dir(save_path_sim)

    # Save Evolution Start and End
    if args.create_sim_data == 1:
        print('Starting simulations...\n\n')
        # ratio_list = [2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
        ratio_list = [1e10]

        # frac_mem_elements = np.round([1, .9, .8, .7, .75, .6, .65, .5, .55, .45], decimals=2)
        frac_mem_elements = np.round([.40, .45, .50, .55, .6], decimals=2)
        # frac_mem_elements = np.round([.30, .35], decimals=2)

        l = [edict({'ratio': r, 'frac_of_static_elements': f, 'batch': b, 'Vbias': args.Vbias,
                    }) for b in range(args.batch_start, args.batch_end)
             for f in (1 - frac_mem_elements) for r in ratio_list]

        # Define control signal
        sim_param.T = .5
        inputsignal = ControlSignal(sources=src, sim_param=sim_param, volt_param=volt_param)
        inputsignal.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)
        # sample rate
        freq = 2  # the frequency of the signal
        inputsignal.V_list[0] = args.Vbias * np.sin(2 * np.pi * freq * (inputsignal.t_list))  # / fs))
        np.save(file=save_path_sim + 'voltage.npy', arr=inputsignal.V_list[0])

        for item in tqdm(l):
            voltagesweep(inputsignal=inputsignal, rows=rows, cols=cols, dictionary=item, save_fold=save_path_sim)


    #

    ratio_plot = [2, 1e10]
    ratio_plot = [1e10]
    # ratio_plot = [2]
    frac_mem_elements_to_plot = np.round([.4, .5, .6], decimals=2)
    list_of_fold = glob(save_path_sim + "frac*/ratio*/batch*/", recursive=True)

    list_of_dict = list()
    for fold in (list_of_fold):
        if len(glob(fold + '/*.pickle')) == 1:
            i_dic = utils.pickle_load(glob(fold + '/*.pickle')[0])
            i_dic['path'] = fold
            list_of_dict.append(i_dic)

    df = pd.DataFrame(list_of_dict)
    df['frac_of_static_elements'] = np.round(df['frac_of_static_elements'], decimals=2)
    df['frac_of_mem_elements'] = df.apply(lambda row: np.round(1 - row.frac_of_static_elements, decimals=2), axis=1)
    df = df[df['ratio'].isin(ratio_plot)]
    df = df[df['frac_of_mem_elements'].isin(frac_mem_elements_to_plot)]

    voltage = np.load(save_path_sim + 'voltage.npy')

    save_path_fig = '{:s}/Figures/'.format(save_path_sim)
    print('Save figures to:\n\t{:s}'.format(save_path_fig))
    utils.ensure_dir(save_path_fig)
    name = 'ratio{:.0e}'.format(ratio_plot[0])
    for f in frac_mem_elements_to_plot:
        name = name +'_{:.2f}'.format(f)
    plot_voltage_sweep_percThr(df=df, save_path=save_path_fig,
                               normalize=False,
                               voltage=voltage,
                               ylabel='I [mA]',
                               title=None,
                               save_name=None,
                               ratio_to_plot=ratio_plot,
                               frac_mem_to_plot=frac_mem_elements_to_plot,
                               nrows=1,
                               ncols=1,
                               name_fig=name,
                               loc=2,
                               figsize=(8, 8),
                               add_yticks=[])
    a=0

