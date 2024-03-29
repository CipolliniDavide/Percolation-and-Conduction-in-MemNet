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
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph, save_npz, load_npz
from scipy.sparse.linalg import eigsh
from scipy.linalg import expm, logm, eigvalsh
from scipy.linalg import eigvals, eigh
from easydict import EasyDict as edict
import itertools

from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend
import math


def plot_ent_vs_fracMem(df, save_path, legend_loop, x_loop, key_y, key_x, key_legend,
                        ylabel=None, normalize=False,
                        xlabel='p', error='std', ymin_max=None,
                        name_fig='', loc=2, figsize=(8, 6), add_yticks=[],
                        legendlabel=r'$\mathbf{G_{max}/G_{min}}$',
                        log_scale=None,
                        format='svg', dpi=1200):

    import matplotlib as mpl
    cmap = plt.cm.get_cmap("jet")
    norm = mpl.colors.SymLogNorm(2, vmin=legend_loop.min(), vmax=legend_loop.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # frac_stat_el = np.round(np.sort(np.unique(np.array(df['frac_of_static_elements'])))[::-1], decimals=2)
    # frac_of_mem_el = np.round(np.sort(np.unique(np.array(df[key_x]))), decimals=2)


    # y_data = df_grr.reset_index()
    # err_y = df_grvar.reset_index()
    y_data = df

    ymax = 0
    ymin = 1e8
    # ymin_max = None
    fig = plt.figure('', figsize=figsize)
    ax = fig.subplots(nrows=1, ncols=1)
    for i, r in enumerate(legend_loop):
    # for i, r in enumerate([.7]):
    #     print(r)
        # if i % 2 ==0:
        # r=.5
        y = [y_data.loc[(y_data[key_legend] == r) & (y_data[key_x] == fr)][key_y] for fr in x_loop]
        # normalize=True
        if normalize:
            max_val = np.array([np.max(a.values) for a in y]).max()
            min_val = np.array([np.min(a.values) for a in y]).min()
            # y_norm = [(y[i] - np.min(y))/(np.max(y)-np.min(y)) for i in range(len(y))]
            y_norm = [(y[i].values - min_val)/(max_val-min_val) for i in range(len(y))]
            yplot = y_norm
        else:
            yplot = y
        y_mean = np.array([yplot[i].mean() for i in range(len(y))])
        if error == 'sem':
            e = np.array([yplot[i].std(ddof=1)/np.sqrt(yplot[i].size) for i in range(len(y))])
        elif error == 'std':
            e = np.array([yplot[i].std(ddof=1) for i in range(len(y))])

        if y_mean.max() > ymax:
            ymax = y_mean.max()
        if y_mean.min() < ymin:
            ymin = y_mean.min()

        if 'Memristor density' in legendlabel:
            if log_scale == True:
                ax.semilogy(x_loop, y_mean, label='{:.1f}'.format(r), color=cmap(norm(r)), marker='^', )
            else:
                ax.plot(x_loop, y_mean, label='{:.1f}'.format(r), color=cmap(norm(r)), marker='^',)
        else:
            ratio_lab = ['2', '10', '100', r'$1x10^3$', r'$1x10^4$', r'$1x10^5$', r'$1x10^6$']
            ratio_list = [2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
            if log_scale==True:
                ax.semilogy(x_loop, y_mean, label='{:s}'.format(ratio_lab[ratio_list.index(r)]), color=cmap(norm(r)), marker='^', )
            else:
                ax.plot(x_loop, y_mean, label='{:.0e}'.format(r), color=cmap(norm(r)), marker='^', )
        ax.fill_between(x_loop, y_mean - e, y_mean + e, alpha=0.2, color=cmap(norm(r)))
        # linestyle = {"linestyle": "--", "linewidth": 4, "markeredgewidth": 5, "elinewidth": 5, "capsize": 10}
        # ax.errorbar(frac_of_mem_el, y_mean, yerr=e, label='{:.0e}'.format(r), color=cmap(norm(r)))

    a=0
    if (ymin_max is None) or (ymin_max[1] > ymax):
        ymin_max = [ymin, ymax]
    if log_scale == True:
        fontdict_ticks_label = {'weight': 'bold', 'size': 'x-large'}
        fontdict_label = {'weight': 'bold', 'size': 'xx-large', 'color': 'black'}
        # add_ticks = add_yticks
        # data = [math.floor(math.log(np.min(ymin_max), 10)), 1 + math.floor(math.log(np.max(ymin_max), 10))],
        # ticks = np.concatenate((np.logspace(start=np.min(data), stop=np.max(data), num=4, endpoint=True), add_ticks))
        # ax.set_yticks(ticks)
        # ax.set_yticklabels(ticks, fontdict=fontdict_ticks_label)
        ax.set_ylabel(ylabel, fontdict=fontdict_label)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontweight('bold') for label in labels]

        # set_ticks_label(ax=ax, ax_type='y',
        #                 num=4,
        #                 # data=np.reshape([max_list, min_list], -1),
        #                 data=[ math.floor(math.log(np.min(ymin_max), 10)), 1 + math.floor(math.log(np.max(ymin_max), 10))],
        #                 ticks=ticks,
        #                 ax_label=ylabel,
        #                 # valfmt=valfmt,
        #                 fontdict_ticks_label={'size': 'large'},
        #                 fontdict_label={'color': 'black'},
        #                 scale='log')
    else:
        set_ticks_label(ax=ax, ax_label=ylabel,
                        data=np.array(ymin_max),
                        valfmt="{x:.5f}",
                        add_ticks=add_yticks,
                        ax_type='y', num=3,
                        )
    if ymin_max is not None:
        plt.ylim((ymin_max[0], ymin_max[1]))
    set_ticks_label(ax=ax, ax_label=xlabel, data=x_loop, ax_type='x', num=5, valfmt="{x:.1f}",
                    # ticks=np.unique(np.round(x_loop, decimals=2)))
                    )
    set_legend(ax=ax, title=legendlabel, ncol=2, loc=loc)
    plt.tight_layout()
    if log_scale==True:
        plt.savefig(join(save_path + '{:s}_{:s}_log.{:s}'.format(name_fig, key_y, format)), format=format, dpi=dpi)
    else:
        plt.savefig(join(save_path + '{:s}_{:s}.{:s}'.format(name_fig, key_y, format)), format=format, dpi=dpi)
    plt.show()
    plt.close('all')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='GridAdiabatic_poorOC', type=str)
    parser.add_argument('-lin_size', '--linear_size', default=21, type=int)
    parser.add_argument('-w_init', '--weight_init', default='None', type=str)
    parser.add_argument('-comp_ds', '--compute_dataset', default=0, type=int)
    # parser.add_argument('-Vb', '--Vbias', default=10, type=float)
    args = parser.parse_args()

    net_param.weight_init = args.weight_init

    root = abspath(join(".", pardir))
    # root = abspath(".")
    print(root)
    time_list = []
    edge_list = []
    start_time = time.time()

    rows = args.linear_size
    cols = args.linear_size
    src = [(rows - 1) // 2]
    gnd = [rows ** 2 - (rows - 1) // 2 - 1]

    save_path_sim = join(root,
                         '{:}/L{:d}/NetSim_StartEnd/{:}/'.format(args.save_path,
                                                                args.linear_size,
                                                                net_param.weight_init))


    save_path_ds = join(root, '{:s}/L{:d}/Conductance/DS/'.format(args.save_path, args.linear_size))
    print('Save to:\n\t{:s}'.format(save_path_ds))
    utils.ensure_dir(save_path_ds)


    if args.compute_dataset == 1:

        m = Measure()
        list_of_fold = glob(save_path_sim + "Vbias*/frac*/ratio*/batch*/", recursive=True)
        # list_of_fold = glob('/Users/dav/PycharmProjects/MemNet/OutputGrid/NetSim_StartEnd/None/Vbias11.0/frac*/ratio*/batch*/',recursive=True)
        for i, fold in enumerate(tqdm(list_of_fold)):
            i_dic = utils.pickle_load(glob(fold+'/*.pickle')[0])
            vbias = float(fold.split('Vbias')[1].rsplit('/')[0])

            # print(i_dic['frac_of_static_elements'])
            # i_dic['frac_of_static_elements'] = np.round(i_dic['frac_of_static_elements'], decimals=2)
            # print(i_dic['frac_of_static_elements'])
            matx_l = [load_npz(m) for m in sorted(glob(fold + '/*.npz'))]
            adj_only_mem = ((matx_l[1] - matx_l[0]) > 0)
            shortest_path_weighted = nx.bidirectional_dijkstra(nx.Graph(matx_l[1]).to_undirected(), src[0], gnd[0], weight='weight')[0]
            try:
                shortest_path = nx.bidirectional_dijkstra(nx.Graph(adj_only_mem).to_undirected(), src[0], gnd[0])[0]
            except:
                shortest_path = np.inf
            tupl = (i_dic, {'G': np.load(fold+'net_conductance.npy')[-1], #1/m.effective_resistance(sparse_matx=matx_l[1], nodeA=src[0], nodeB=gnd[0]),
                            'G_min': np.load(fold+'net_conductance.npy')[0],
                            'Vbias': vbias,
                            'shortest_path': shortest_path,
                            'shortest_path_weighted': shortest_path_weighted
                            })
            utils.pickle_save(obj=utils.merge_dict(tuple=tupl), filename=save_path_ds+'{:09d}.pickle'.format(i))

    df = create_dataset(load_dir=save_path_ds,  # .rsplit('/', 1)[0],
                        save_dir=None,
                        save_name=None,
                        sheet_name='Sheet1',
                        remove_nan_inf=False,
                        remove_labels=[])

    df['frac_of_static_elements'] = np.round(df['frac_of_static_elements'], decimals=2)
    df['frac_of_mem_elements'] = df.apply(lambda row: np.round(1-row.frac_of_static_elements, decimals=2), axis=1)


    df = df.loc[df['batch'] < 100]
    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures_NewNorm_AbovePerc/'
    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures_NewNorm_belowPerc/'
    save_path_figures = save_path_ds.split('DS')[0] + 'Figures_NewNorm_AroundPerc/'
    utils.ensure_dir(save_path_figures)


    # df = df[df['ratio'].isin([2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])] #1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    df = df[df['ratio'].isin([1e4])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['frac_of_mem_elements'].isin([.5, .6])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['frac_of_mem_elements'].isin([.1, .2, .3, .4])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    df = df[df['frac_of_mem_elements'].isin([.5, .6, .7, .8, .9, 1])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]

    save_path_figures_PLeg = save_path_figures + 'Ratio_fixed/'
    ##################### To obtain S7 #####################################
    for name_fold, norm, log_scale, ymin_max, ylab, key in zip(['G/','G_norm/'],
                                                    [False, True],
                                                    [True, False],
                                                    [None, None],
                                                    [r'$\mathbf{G_{nw}}$'+'[a.u.]', r'$\mathbf{G_{nw}^{norm}}$'],
                                                    ['G','G'],
                                                    ):

        save_path_figures_G_norm = save_path_figures_PLeg + name_fold
        utils.ensure_dir(save_path_figures_G_norm)
        for rat in np.round(np.sort(np.unique(np.array(df['ratio'])))):
            print(rat)
            plot_ent_vs_fracMem(df=df[df['ratio'].isin([rat])],  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])],
                                log_scale=log_scale,
                                key_y=key,
                                key_x='Vbias',
                                key_legend='frac_of_mem_elements',
                                ylabel=ylab,
                                ymin_max=ymin_max,
                                save_path=save_path_figures_G_norm,
                                normalize=norm,
                                xlabel='V [a.u.]\nVoltage input',
                                legendlabel='Memristor density\np',
                                legend_loop=np.round(np.sort(np.unique(np.array(df['frac_of_mem_elements']))), decimals=2),
                                x_loop=np.sort(np.unique(np.array(df['Vbias']))),
                                name_fig='{:.0e}'.format(rat),
                                error='std')

        a=0
