"Just a very first draft: not working at all"

import os
import math
import argparse
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.visualize import visualize
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from Class_SciPySparseV2.anim_vis import plot_H_evolution
# from Class_SciPySparseV2.make_gif import make_gif
from easydict import EasyDict as edict
# import torch
# import torch.nn as nn

from os.path import isfile, join, abspath
from os import pardir
from script_Opt.Class_SciPySparse.utils import utils
from copy import deepcopy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='TopologyFigure', type=str)
    parser.add_argument('-lin_size', '--linear_size', default=21, type=int)
    # parser.add_argument('-crt_sim_data', '--create_sim_data', default=0, type=int)
    parser.add_argument('-diag', '--random_diagonals', default=1, type=int)
    # parser.add_argument('-ent_f', '--entropy_func', default='SpecExpEnt', type=str, help='VNGE_FINGER')
    args = parser.parse_args()

    ########
    net_param = edict({'rows': args.linear_size,
                       'cols': args.linear_size,
                       'frac_of_static_elements': 0,
                       'weight_init': None,  # 'rand',
                       'seed': 2})


    mem_param = edict({'kp0': 2.555173332603108574e-06,  # model kp_0
                       'kd0': 6.488388862524891465e+01,  # model kd_0
                       'eta_p': 3.492155165334443012e+01,  # model eta_p
                       'eta_d': 5.590601016803570467e+00,  # model eta_d
                       'g_min': 1,  # model g_min
                       'g_max': 1e2,  # model g_max
                       'g0': 1  # model g_0
                       })

    sim_param = edict({'T': 2.5e-1, #4e-3, # [s]
                        #'steps': 100,
                       'sampling_rate': 10000 # [Hz]  # =steps / T  # [Hz]
                        #dt = T / steps  # [s] or    dt = 1/sampling_rate bc  steps= sampling_rate *T
                        })

    # Save path
    root = abspath(join(".", pardir))
    print(root)
    save_path = root + '/' + args.save_path + '/'
    utils.ensure_dir(save_path)


    # Set Input Nodes, Hidden Nodes and Ground nodes
    for diag in [0,1]:
        for (loc_name, (source_node, ground_node)) in zip(['same_row', 'diff_rowcol'],
                                                      [(10, 430), (18, 425)]):
            src = sorted([source_node])
            gnd = [ground_node]
            input_source_labels = [(src[i], 'Inp{:d}'.format(src[i])) for i in range(len(src))]
            node_labels = [(gnd[0], 'Gnd')] + input_source_labels

            # Instantiate memristor network class
            net = MemNet(mem_param=mem_param, net_param=net_param, gnd=gnd, src=src, diag=diag)
            coordinates = [(node, (feat['coord'])) for node, feat in net.G.nodes(data=True)]
            visualize.plot_network(G=net.G, numeric_label=False, labels=node_labels, figsize=(14,14))
            plt.savefig(save_path+'{:s}_diag{:d}.svg'.format(loc_name, diag))
            plt.show()
            plt.close()
    a=0