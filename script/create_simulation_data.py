from script_Opt.Class_SciPySparse.mem_parameters import sim_param, volt_param, net_param, mem_param
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.utils import utils, create_dataset, getListOfFiles

import numpy as np
import argparse
from tqdm import tqdm, contrib
import copy
from os.path import isfile, join, abspath
from os import pardir
import time
from easydict import EasyDict as edict


def save_evolution(input_signal, rows, cols, dictionary, save_fold):
    mem_p_loc = copy.deepcopy(mem_param)
    net_param_loc = copy.deepcopy(net_param)
    net_param_loc.rows = rows
    net_param_loc.cols = cols
    mem_p_loc.g_max = mem_p_loc.g_min * dictionary.ratio
    net_param_loc.frac_of_static_elements = dictionary.frac
    net_param_loc.seed = None

    save_path = '{:s}frac{:.2f}/ratio{:.1e}/batch{:04d}/'.format(save_fold,
                                                                 net_param_loc.frac_of_static_elements,
                                                                 dictionary.ratio,
                                                                 dictionary.batch)
    utils.ensure_dir(save_path)

    net = MemNet(mem_param=mem_p_loc, net_param=net_param_loc, gnd=gnd, src=src, diag=diag)
    net.run(groundnode_list=gnd, sourcenode_list=src, V_list=input_signal.V_list, t_list=input_signal.t_list,
            save_path=save_path)

    dict_rob = edict({'batch': dictionary.batch, 'ratio': dictionary.ratio})  # + mem_p_loc + net_param_loc + sim_param
    dict_3 = {k: v for d in (dict_rob, mem_p_loc, net_param_loc, sim_param) for k, v in d.items()}
    utils.pickle_save(filename='{:s}info.pickle'.format(save_path), obj=dict_3)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', default='OutputGrid', type=str)
    parser.add_argument('-lin_size', '--linear_size', default=21, type=int)
    parser.add_argument('-w_init', '--weight_init', default='None', type=str)
    parser.add_argument('-b_start', '--batch_start', default=0, type=int)
    parser.add_argument('-b_end', '--batch_end', default=20, type=int)
    parser.add_argument('-Vb', '--Vbias', default=10, type=float)
    parser.add_argument('-diag', '--random_diagonals', default=0, type=int)
    # parser.add_argument('-ent_f', '--entropy_func', default='SpecExpEnt', type=str, help='VNGE_FINGER')
    args = parser.parse_args()

    net_param.weight_init = args.weight_init
    args.Vbias = args.Vbias / 10
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

    # Input signal class
    sim_param.T = .5
    inputsignal = ControlSignal(sources=src, sim_param=sim_param, volt_param=volt_param)
    inputsignal.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)
    inputsignal.V_list[0] = [args.Vbias] * len(inputsignal.t_list)
    inputsignal.V_list[0][0] = .1

    # Adiabatic increase of voltage
    j = 3
    a = np.hstack((np.linspace(.1, args.Vbias, len(inputsignal.t_list) // j),
                   args.Vbias * np.ones(len(inputsignal.t_list) - len(inputsignal.t_list) // j)))
    inputsignal.V_list[0] = list(a)

    # Save Path
    # save_path_sim = join(root, '{:}/Entropy/NetSim_StartEnd/{:}/Vbias{:.1f}/'.format(args.save_path, net_param.weight_init, Vbias))
    save_path_sim = join(root,
                         '{:}/L{:d}/NetSim_StartEnd/{:}/Vbias{:.2f}/'.format(args.save_path, args.linear_size,
                                                                   net_param.weight_init, args.Vbias))
    utils.ensure_dir(save_path_sim)

    # Save Evolution Start and End
    print('Starting simulations...\n\n')
    # frac_of_mem_list = np.round([1], decimals=2) #np.round(np.arange(.1, 1.1, .1, dtype=np.float16), decimals=2)
    frac_of_mem_list = np.round(np.arange(.1, 1, .1, dtype=np.float16), decimals=2)
    # frac_of_mem_list = np.round([.3, .7, .9], decimals=2)
    # ratio_list = [2, 1e1, 1e2, 1e3]#, 1e4, 1e5]#, 1e6] #, 1e7, 1e8]
    ratio_list = [2, 1e4, 1e6]
    # frac_list = np.round(np.arange(0, .5, .1, dtype=np.float16), decimals=2)

    # ratio_list = [1e3, 1e4, 1e5, 1e6]
    # frac_of_mem_list = np.round(np.arange(.5, 1, .1, dtype=np.float16), decimals=2)

    l = [edict({'ratio': r, 'frac': f, 'batch': b}) for b in range(args.batch_start, args.batch_end)
         for f in (1-frac_of_mem_list) for r in ratio_list]
    for r in ratio_list: l.append(edict({'ratio': r, 'frac': 0, 'batch': 0}))

    for item in tqdm(l):
        save_evolution(input_signal=inputsignal, rows=rows, cols=cols, dictionary=item, save_fold=save_path_sim)
