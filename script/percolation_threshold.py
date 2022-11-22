import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm, contrib
#import tensorflow as tf
from matplotlib import pyplot as plt
from scipy import signal
# from IPython.display import display, clear_output
import networkx as nx
from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend
from script_Opt.Class_SciPySparse.utils import utils

import random
import sys
import copy
from glob import glob
from os.path import isfile, join, abspath
from os import pardir
from scipy import sparse
import time

from script_Opt.Class_SciPySparse.visualize import visualize


def plot(frac_of_mem_elements,
         prob_path,
         pc,
         figsize=(8, 8),
         save_path='./',
         ylabel='Probability of connected cluster',
         figname='percolationThreshold.png'
         ):
    fig = plt.figure('percolation thresh', figsize=figsize)
    ax = fig.subplots(nrows=1, ncols=1)
    # for i, (frac_mem, prob) in enumerate(zip(frac_of_mem_elements, prob_path)):
    ax.plot(frac_of_mem_elements, prob_path , marker='^',)
    ax.vlines(x=pc, ymin=0, ymax=1, colors='orange', ls='--', lw=3, alpha=.6)
    set_ticks_label(ax=ax, ax_label=ylabel, data=np.array([1, 0]), ax_type='y', num=5, valfmt="{x:.1f}")
    set_ticks_label(ax=ax, ax_label='p', ax_type='x', num=5,
                    valfmt="{x:.2f}", add_ticks=[pc], data=[],
                    ticks=[.1, pc, 1],
                    )
    plt.tight_layout()
    # plt.savefig(join('{:s}{:s}'.format(save_path, figname)))
    # plt.show()
    # plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='Percolation', type=str)
    parser.add_argument('-crt_data', '--create_data', default=1, type=int)
    # parser.add_argument('-diag', '--diagonals', default=0, type=int)
    # parser.add_argument('-b_start', '--batch_start', default=0, type=int)
    # parser.add_argument('-b_end', '--batch_end', default=20, type=int)
    parser.add_argument('-b', '--batch', default=20, type=int)
    parser.add_argument('-xs', '--xstep', default=.01, type=float)
    parser.add_argument('-linsize', '--L', default=21, type=int)
    # parser.add_argument('-nc', '--ncols', default=21, type=int)
    # parser.add_argument('-nr', '--nrows', default=21, type=int)
    args = parser.parse_args()

    root = abspath(join(".", pardir))
    # root = abspath(".")
    print(root)
    # save_path = join(root, '{:}/Percolation/Net{:02d}x{:02d}/'.format(args.save_path,
    #                                                                                args.nrows,
    #                                                                                args.ncols))
    save_path = join(root, '{:}/'.format(args.save_path))
    utils.ensure_dir(save_path)
    L_list = [1000]#[21, 41, 51, 61, 71, 81, 100]
    # L_list = [args.L]
    for L in L_list:
        save_path_figures_net = save_path + 'NetFig/'
        utils.ensure_dir(save_path_figures_net)
        save_name = 'prob_of_conn_cluster_batch{:04}'.format(args.batch)

        frac_of_mem_elements = np.round(np.arange(start=.5, stop=.7 + args.xstep, step=args.xstep), decimals=2)
        frac_of_mem_elements = np.hstack((np.round(np.arange(start=.1, stop=1, step=.1), decimals=2), frac_of_mem_elements))
        frac_of_mem_elements = np.sort(np.unique(frac_of_mem_elements))
        # frac_of_mem_elements = [.2, .6]
        src = [(L - 1) // 2]
        gnd = [L ** 2 - (L - 1) // 2 - 1]

        if args.create_data:
            Ggrid = nx.grid_graph(dim=[L, L])
            Ggrid = nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='coord')
            coordinates = [(node, (feat['coord'])) for node, feat in Ggrid.nodes(data=True)]
            # print('# static edges', number_of_stat_edges)
            Adj = nx.to_scipy_sparse_array(Ggrid, format='csr')
            # print(Adj.data.sum())
            Adj = sparse.triu(Adj, k=0)
            number_of_edges = Adj.data.sum()
            # adj_indexes = np.argwhere(Adj > 0)
            # triangular_adj_indexes = np.argwhere(sparse.triu(Adj, k=0) > 0)

            prob_path = np.zeros((len(frac_of_mem_elements)))

            for frac_mem in tqdm(frac_of_mem_elements):
                count = 0
                for b in range(args.batch):
                    number_of_stat_edges = int((1-frac_mem) * number_of_edges)
                    # print('\nFrac mem el: ', frac_mem)
                    # print('Frac stat el: ', number_of_stat_edges/number_of_edges)
                    # print('\r# static edges', number_of_stat_edges)
                    # print('# dyn edges', number_of_edges-number_of_stat_edges)

                    stat_el_index = np.array(random.sample(range(number_of_edges), k=number_of_stat_edges))
                    temp_adj = copy.deepcopy(Adj)
                    # print((temp_adj.data == 0).sum())
                    if len(stat_el_index) == 0:
                        # print(frac_mem)
                        pass
                    else:
                        temp_adj.data[stat_el_index] = 0
                    # print((temp_adj.data == 0).sum())
                    G = nx.from_numpy_array(A=(temp_adj + temp_adj.T))

                    try:
                        l, _ = nx.bidirectional_dijkstra(G, src[0], gnd[0])
                        count += 1
                    except:
                        pass
                prob_path[frac_of_mem_elements == frac_mem] = count/args.batch
                # if frac_mem in np.round(frac_of_mem_elements, decimals=1):
                #     # nx.draw_networkx(G,
                #     #                  labels={src[0]: 'Src', gnd[0]: 'Gnd'},
                #     #                  pos=dict(coordinates), with_labels=True, width=4, node_size=30)
                #     # plt.title('p={:.2f}'.format(frac_mem), size='xx-large', weight='bold')
                #     # plt.show()
                #     # plt.close()
                #     nx.set_node_attributes(G, dict(coordinates), 'coord')
                #     visualize.plot_network(G,
                #                            figsize=(8, 8),
                #                            up_text=.3, hor_text=0,
                #                            node_size=30, numeric_label=False,
                #                            labels=[(src[0], 'Src'), (gnd[0], 'Gnd')],
                #                            save_fold=save_path, title='p={:.2f}'.format(frac_mem),
                #                            show=False)
                #     plt.tight_layout()
                #     plt.savefig(save_path_figures_net+'p{:.2f}.png'.format(frac_mem))
                #     plt.close('all')

            np.save(file=save_path + 'frac_mem_elements_L{:d}.npy'.format(L), arr=frac_of_mem_elements)
            np.save(file='{:s}{:s}_L{:d}.npy'.format(save_path, save_name, L), arr=prob_path)

    pc_arr = np.zeros(len(L_list))
    for i, L in enumerate(L_list):

        frac_mem_elements = np.load(save_path + 'frac_mem_elements_L{:d}.npy'.format(L))
        prob_path = np.load('{:s}{:s}_L{:d}.npy'.format(save_path, save_name, L))

        pc_arr[i] = frac_of_mem_elements[prob_path >= .5][0]

        # plot(frac_of_mem_elements,
        #      prob_path,
        #      pc=pc_arr[i],
        #      figsize=(8, 6),
        #      save_path=save_path,
        #      ylabel='Prob. of mem. spanning cluster',
        #      figname='{:s}.png'.format(save_name)
        #      )
        # plt.show()
        # plt.close()
    plt.plot(pc_arr)



