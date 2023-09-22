from script_Opt.Class_SciPySparse.mem_parameters import sim_param, volt_param, net_param, mem_param
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.utils import utils, create_dataset, getListOfFiles
from script_Opt.Class_SciPySparse.visualize import visualize
from script_Opt.Class_SciPySparse.anim_vis import plot_H_evolution

import numpy as np
import argparse
from tqdm import tqdm, contrib
from copy import deepcopy
from os.path import isfile, join, abspath
from os import pardir
import time
from easydict import EasyDict as edict
from matplotlib import pyplot as plt


# Network parameters
net_param = edict({'rows': 5,
                   'cols': 5,
                   'frac_of_static_elements': .25, # Fraction of Ohmic conductors in the network
                   'weight_init': None,  # 'rand', None for grid; 'rand' for random diagonals
                   'seed': 2})

# Memristive edges parameters
mem_param = edict({'kp0': 2.555173332603108574e-06,  # model kp_0
                   'kd0': 6.488388862524891465e+01,  # model kd_0
                   'eta_p': 3.492155165334443012e+01,  # model eta_p
                   'eta_d': 5.590601016803570467e+00,  # model eta_d
                   'g_min': 1.014708121672117710e-03,  # model g_min
                   'g_max': 2.723493729125820492e-03,  # model g_max
                   'g0': 1.014708121672117710e-03  # model g_0
                   })

# Simulation parameters
sim_param = edict({'T': 100e-3, # [s]
                    'sampling_rate': 500 # [Hz]
                    })

# This params are needed to initialize the ControlSignal class but signals in the class can be changed on the fly, see in main
volt_param = edict({'VMAX': 20,
                     'VMIN': 1,
                     'VSTART': .3, #'rand',
                     'amplitude_gain_factor': .01
                     })


def net_entropy_from_conductances(adj_matrix):
    '''
    Measure entropy from conductances in the network.
    self.Condmat must be triangular!
    :return:
    '''
    CondMat_norm = adj_matrix.data / adj_matrix.data.sum()
    return - np.multiply(CondMat_norm.data, np.log(CondMat_norm.data)).sum() #/np.log(adj_matrix.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_fold', default='OutputTest', type=str)
    parser.add_argument('-lin_size', '--linear_size', default=21, type=int)
    parser.add_argument('-w_init', '--weight_init', default='None', type=str)
    parser.add_argument('-Vb', '--Vbias', default=2, type=float)
    parser.add_argument('-diag', '--random_diagonals', default=0, type=int)
    args = parser.parse_args()

    net_param.weight_init = args.weight_init
    diag = args.random_diagonals
    root = abspath(join(".", pardir))

    # Update size of network on the fly
    net_param.rows = args.linear_size
    net_param.cols = args.linear_size

    print(f'Fraction of memristive edges in the network is equal to {1-net_param.frac_of_static_elements}')

    # Set source electrodes and ground electrodes
    src = [10, 21*5] # define a list of source nodes in the range [0, net_param.rows*net_param.cols-1]
    gnd = [net_param.rows**2 - (net_param.rows - 1) // 2 - 1] # define a list of ground nodes in the range [0, net_param.rows*net_param.cols-1]

    # Input signal class
    # Set time length updating the sim_para dictionary
    sim_param.T = .5
    # Instantiate the input signal class: it fills already with a pulse signal all your input electrodes
    inputsignal = ControlSignal(sources=src, sim_param=sim_param, volt_param=volt_param)
    # Plot the input signals
    inputsignal.plot_V()
    plt.title('Voltage signal')
    plt.show()

    # You can change on the fly both the time length and the input signal in each electrode: i.e. we change the first
    inputsignal.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)
    inputsignal.V_list[0] = inputsignal.square_ramp(StartAmplitude=.5, number_of_cycles=5)
    # Plot the new input signals
    inputsignal.plot_V()
    plt.title('Updated voltage signal')
    plt.show()


    # Instantiate the memristive network class
    net = MemNet(mem_param=mem_param, net_param=net_param, gnd=gnd, src=src, diag=diag)
    # Visulaize network topology
    visualize.plot_network(net.G, show=True, numeric_label=True,
                           labels=[(gnd[0], 'Gnd'), (src[0], f'Src{src[0]}'), (src[1], f'Src{src[1]}')]
                           )
    plt.show()

    # Create list to save network evolutions
    H_list = [[] for t in range(len(inputsignal.t_list))]
    # List storing effective conductance betw. two nodes
    eff_cond = list()
    # List storing the entropy computed over the network conductance distribution across time
    entropy_net = list()
    # Voltage distribution over nodes in the network
    node_voltage_list = list()

    # Start simulation:
    print('Start simulation')
    # First simulation step is outside the loop
    h = net.MVNA(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=0)
    H_list[0] = deepcopy(h)
    node_voltage_list.append(deepcopy(net.node_Voltage))

    eff_cond.append(net.calculate_network_conductance(nodeA=net.src[0], nodeB=net.gnd[0], V_read=3))
    entropy_net.append(net_entropy_from_conductances(adj_matrix=h))

    delta_t = inputsignal.t_list[1] - inputsignal.t_list[0]
    for t in range(1, len(inputsignal.t_list)):
        print(t, '/', len(inputsignal.t_list))
        net.update_edge_weights(delta_t=delta_t)
        h = net.MVNA(groundnode_list=gnd, sourcenode_list=src, V_list=inputsignal.V_list, t=t)
        eff_cond.append(net.calculate_network_conductance(nodeA=net.src[0], nodeB=net.gnd[0], V_read=3))
        entropy_net.append(net_entropy_from_conductances(adj_matrix=h))
        H_list[t] = deepcopy(h)
        node_voltage_list.append(deepcopy(net.node_Voltage))

    # The simulation can be called by the following muted function
    # H_list = net.run(groundnode_list=gnd,
    #                  sourcenode_list=src,
    #                  V_list=inputsignal.V_list,
    #                  t_list=inputsignal.t_list,
    #                  save_path=None)

    # Save path
    save_path = '../{:s}/fracOC{:.2f}_ratio{:.1e}/'.format(args.save_fold,
                                                           net_param.frac_of_static_elements,
                                                           mem_param.g_max / mem_param.g_min
                                                           )
    utils.ensure_dir(save_path)

    # Create labels for plot of netw. evolution
    node_labels = [(gnd[0], 'Gnd'), (src[0], f'Src{src[0]}'), (src[1], f'Src{src[1]}')]
    # Extract coordinates from node attributes for 2D plot
    coordinates = [(node, (feat['coord'])) for node, feat in net.G.nodes(data=True)]

    plot_H_evolution(H_Horizon=H_list,
                     t_list_Horizon=inputsignal.t_list,
                     numeric_label=False,
                     eff_cond_list=eff_cond,
                     desired_eff_conductance=None,
                     V_Horizon=inputsignal.V_list,
                     node_labels=node_labels,
                     y_label_2ndrow=r'$\mathbf{G_{nw}}$',
                     src=src,
                     coordinates=coordinates,
                     number_of_plots=len(H_list) // 10, # Choose the number of snapshots
                     node_voltage_list=node_voltage_list,
                     title='p={:.2f}, '.format(1 - net_param.frac_of_static_elements) + \
                           r'$\mathbf{G_{max}/G_{min}}$' + '={:.0e}'.format(mem_param.g_max/mem_param.g_min),
                     save_path=save_path,
                     format='pdf')

    print('Show case ended.')


