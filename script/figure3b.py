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
                        rTh_log=0,
                        name_fig='', loc=2, figsize=(8, 6), add_yticks=[],
                        legendlabel=r'$\mathbf{G_{max}/G_{min}}$',
                        log_scale=None,
                        format='svg', dpi=1200, z=1):

    import matplotlib as mpl
    cmap = plt.cm.get_cmap("jet")
    norm = mpl.colors.SymLogNorm(2, vmin=legend_loop.min(), vmax=legend_loop.max())
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    y_data = df

    figsize=(12, 14)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(ncols=2, nrows=8)
    ax1 = fig.add_subplot(gs[1:3, 0])
    ax2 = fig.add_subplot(gs[3:5, 0])
    ax3 = fig.add_subplot(gs[5:7, 0])
    ax4 = fig.add_subplot(gs[0:2, 1])
    ax5 = fig.add_subplot(gs[2:4, 1])
    ax6 = fig.add_subplot(gs[4:6, 1])
    ax7 = fig.add_subplot(gs[6:8, 1])
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    # fig = plt.figure('', figsize=figsize)
    # ax = fig.subplots(nrows=1, ncols=1)
    ratio_lab = ['2', '10', '100', r'$1\times10^3$', r'$1\times10^4$', r'$1\times10^5$', r'$1\times10^6$']
    ratio_list = [2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2']  # '#7f7f7f',  '#bcbd22', '#17becf']

    for i, (r, ax) in enumerate(zip(legend_loop, axes)):
        y = [y_data.loc[(y_data[key_legend] == r) & (y_data[key_x] == fr)][key_y] for fr in x_loop]
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
            e = z*np.array([yplot[i].std(ddof=1)/np.sqrt(yplot[i].size) for i in range(len(y))])
        elif error == 'std':
            e = z * np.array([yplot[i].std(ddof=1) for i in range(len(y))])

        if 'Memristor density' in legendlabel:
            if log_scale == True:
                ax.semilogy(x_loop, y_mean, label='{:.1f}'.format(r),
                            color=colors[i],
                            # color=cmap(norm(r)),
                            marker='^')
            else:
                ax.plot(x_loop, y_mean, label='{:.1f}'.format(r),
                        color=colors[i])
                        # color=cmap(norm(r)), marker='^',)
        else:
            if (log_scale == True) and (r >= rTh_log):
                ax.semilogy(x_loop, y_mean, label='{:s}'.format(ratio_lab[ratio_list.index(r)]),
                            color=colors[i],
                            # color=cmap(norm(r)),
                            marker='^', )
            else:
                ax.plot(x_loop, y_mean, label='{:s}'.format(ratio_lab[ratio_list.index(r)]),
                        color=colors[i],
                        # color=cmap(norm(r)),
                        marker='^', )
        ax.fill_between(x_loop, y_mean - e, y_mean + e, alpha=0.2,
                        color=colors[i])
                        # color=cmap(norm(r)))

        fontdict_label = {'weight': 'bold', 'size': '20', 'color': 'black'}
        if (log_scale == True) and (r>=rTh_log):
            fontdict_ticks_label = {'weight': 'bold', 'size': 16}
            # add_ticks = add_yticks
            # data = [math.floor(math.log(np.min(ymin_max), 10)), 1 + math.floor(math.log(np.max(ymin_max), 10))],
            # ticks = np.concatenate((np.logspace(start=np.min(data), stop=np.max(data), num=4, endpoint=True), add_ticks))
            # ax.set_yticks(ticks)
            # ax.set_yticklabels(ticks, fontdict=fontdict_ticks_label)
            labels = ax.get_yticklabels() + ax.get_yticklabels('minor')
            [label.set_fontweight('bold') for label in labels]
            [label.set_size(fontdict_ticks_label['size']) for label in labels]
            ax.set_ylabel(ylab, fontdict=fontdict_label)

        else:
            set_ticks_label(ax=ax, ax_label=ylabel,
                            fontdict_label=fontdict_label,
                            data=y_mean,
                            valfmt="{x:.2e}",
                            add_ticks=add_yticks,
                            ax_type='y', num=3,
                            )
        if (r==100) or (r==1e6):
            set_ticks_label(ax=ax, ax_label=xlabel, data=x_loop, ax_type='x', num=5, valfmt="{x:.1f}",
                            # ticks=np.unique(np.round(x_loop, decimals=2)))
                            )
        else:
            set_ticks_label(ax=ax, ax_label='p', data=x_loop, ax_type='x', num=5, valfmt="{x:.1f}",
                            fontdict_label=fontdict_label,
                            # ticks=np.unique(np.round(x_loop, decimals=2)))
                            )
        set_legend(ax=ax, title=legendlabel, ncol=2, loc=loc, fontsize=20, title_fontsize=20)
    plt.tight_layout()
    # plt.show()

    if log_scale==True:
        plt.savefig(join(save_path + '{:s}_{:s}_log_{:s}z{:.2f}.{:s}'.format(name_fig, key_y, error, z, format)), format=format, dpi=dpi)
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
    # rows = cols = int(21)
    # src = [11]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    # gnd = [391]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    #src = [2]
    # gnd = [22]
    # src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    # gnd = [430]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

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

    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures/Vbias' + save_path_ds.split('Vbias')[1]
    df = df.loc[df['batch'] < 100]
    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures_NewNorm_AbovePerc/'
    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures_NewNorm_belowPerc/'
    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures_NewNorm_AroundPerc/'
    save_path_figures = save_path_ds.split('DS')[0] + 'Figures_NewNorm/'
    utils.ensure_dir(save_path_figures)

    
    v=2.5
    if v==8:
        log=True
    else: log=False
    df = df[df['Vbias'].isin([v])] #1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['ratio'].isin([2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])] #1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['ratio'].isin([1e4])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['ratio'].isin([1e5])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['frac_of_mem_elements'].isin([.5, .6])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['frac_of_mem_elements'].isin([.1, .2, .3, .4])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    # df = df[df['frac_of_mem_elements'].isin([.5, .6, .7, .8, .9, 1])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]


    # ########################### Fixed Voltage #########################
    save_path_figures_vbias = save_path_figures + 'Vbias_fixed/'

    # for v in np.round(np.sort(np.unique(np.array(df['Vbias']))), decimals=2):
    #
    #     # Plot (pure) conductance
    #     for name_fold, norm, log_scale, ymin_max, ylab in zip(['G/', 'G_norm/'],
    #                                                      [False, True], # norm
    #                                                     [log, False], #logscale
    #                                                [None, None],
    #                                                ['Conductance\n'+r'$\mathbf{G_{nw}}$'+' [a.u.]', 'Conductance\n'+r'$\mathbf{G_{nw}^{norm}}$']):
    #                                                           # [r'$\mathbf{G_{nw}}$'+' [a.u.]',
    #                                                           #  'Conductance\n' + r'$\mathbf{G_{nw}^{norm}}$']):

    name_fold = 'G/'
    norm = False
    log_scale = True
    ymin_max = [None, None]
    ylab = 'Conductance\n' + r'$\mathbf{G_{nw}}$' + ' [a.u.]'

    save_path_figures_G_norm = save_path_figures_vbias + name_fold
    utils.ensure_dir(save_path_figures_G_norm)
    for v in np.round(np.sort(np.unique(np.array(df['Vbias']))), decimals=2):
        plot_ent_vs_fracMem(df=df[df['Vbias'].isin([v])],
                            key_y='G',
                            normalize=norm,
                            log_scale=log_scale,
                            key_x='frac_of_mem_elements',
                            key_legend='ratio',
                            ylabel=ylab,
                            save_path=save_path_figures_G_norm,
                            xlabel='p\nMemristor density',
                            # legendlabel='Interaction strength\n'+r'$\mathbf{G_{max}/G_{min}}$',
                            legendlabel=r'$\mathbf{G_{max}/G_{min}}$',
                            ymin_max=ymin_max,
                            # frac_of_mem_el=np.round(np.sort(np.unique(np.array(df['frac_of_mem_elements']))), decimals=2),
                            # ratio=np.sort(df['ratio'].unique()),
                            legend_loop=np.round(np.sort(np.unique(np.array(df['ratio']))), decimals=2),
                            x_loop=np.sort(np.unique(np.array(df['frac_of_mem_elements']))),
                            name_fig='Vbias{:.2f}'.format(v),
                            error='sem',
                            z=1.96)
                            # z=2.576)

    a=0