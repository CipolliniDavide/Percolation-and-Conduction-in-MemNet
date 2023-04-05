from script_Opt.Class_SciPySparse.mem_parameters import mem_param, sim_param, volt_param, train_param, \
    env_param
from script_Opt.Class_SciPySparse.utils import utils
from script_Opt.Class_SciPySparse.mem_parameters import sim_param, volt_param, net_param, mem_param
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.utils import utils, create_dataset, getListOfFiles
from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend, align_yaxis


import numpy as np
import argparse
from tqdm import tqdm, contrib
import copy
from os.path import isfile, join, abspath
from os import pardir
import time
from easydict import EasyDict as edict
from matplotlib import pyplot as plt

def plot_twinx(x_data, x_label, y_data, y_label, y2_data=[], y2_label=None, save_path=None, figname=None,
               y_scale=['lin', 'lin'], title=None, curve_labels_y=[''], curve_labels_y2=[''], show=False,
               ax=None, y1_ticks=None, y2_ticks=None, x_ticks=None,
               colors=['blue', 'red']):
    # from Visual.visual_utils import align_yaxis, set_legend, set_ticks_label

    if type(y_data) is not list:
        y_data = [y_data]
    if type(y2_data) is not list:
        y2_data = [y2_data]

    if y1_ticks is None:
        y1_ticks = [np.min(y_data), 0, np.max(y_data)]
    else:
        pass

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))
    axes = [ax]
    # make a plot
    if y_scale[0] == 'log':
        ax.semilogy(x_data, y_data, color="red", marker="o", markersize=.5)
    else:
        for (y, cur_lab, c) in zip(y_data, curve_labels_y, colors):
            ax.plot(x_data, y, color=c, label=cur_lab, marker='o',
                linewidth=3, markersize=4)
    if x_ticks is None:
        set_ticks_label(ax=ax, ax_label=x_label, ax_type='x', valfmt="{x:.1f}", data=x_data, num=5)
    else:
        set_ticks_label(ax=ax, ax_label=x_label, ax_type='x', valfmt="{x:.1f}", data=x_data, ticks=x_ticks, num=5)
    if len(y2_data) != 0:
        if y2_ticks is None:
            y2_ticks = [np.min(y2_data), 0, np.max(y2_data)]
        else:
            pass
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        axes.append(ax2)
        # make a plot with different y-axis using second axis object
        if y_scale[1] == 'log':
            ax2.semilogy(x_data, y2_data, color="green", marker="o", markersize=.5)
        else:
            for (y, cur_lab) in zip(y2_data, curve_labels_y2):
                ax2.plot(x_data, y, color="green",  marker='o',
                linewidth=3, markersize=4)
        set_ticks_label(ax=ax2, ax_label=y2_label, ax_type='y', data=y2_data, num=5,
                        ticks=y2_ticks, valfmt="{x:.1f}",
                        fontdict_label={'color':'green'})
    if title:
        plt.title(title)
    ax.grid()
    if len(y2_data) > 0:
        align_yaxis(ax, ax2)
    if len(y_data) > 1:
        set_legend(ax=ax)
        set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                        ticks=y1_ticks, fontdict_label={'color': c})
    set_ticks_label(ax=ax, ax_label=y_label, ax_type='y', data=y_data, num=5,
                    ticks=y1_ticks, fontdict_label={'color': c})
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + figname)
    if show==True:
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-svp', '--save_path', #default='OutputGridAdiabatic_GoodOC', type=str)
                        default='OutputGridAdiabatic', type=str)
    parser.add_argument('-diag', '--diagonals', default=0, type=int)
    # parser.add_argument('-crt_sim_data', '--create_sim_data', default=0, type=int)
    args = parser.parse_args()

    linsize = 21
    m = linsize**2 - linsize

    volt = np.arange(0, .6, .001)
    kp = mem_param.kp0*np.exp(mem_param.eta_p*volt)
    kd = mem_param.kd0*np.exp(-mem_param.eta_d*volt)
    first_term = kp/(kd+kp)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volt, first_term, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label='Voltage [a.u.]', data=volt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{\frac{k_p}{k_p + k_d}}$', data=first_term, valfmt='{x:.1e}')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volt, kp, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label='Voltage [a.u.]', data=volt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{k_p}$', data=kp, valfmt='{x:.1e}')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volt, kd, marker='o')
    set_ticks_label(ax=ax, ax_type='x', ax_label='Voltage [a.u.]', data=volt)
    set_ticks_label(ax=ax, ax_type='y', ax_label=r'$\mathbf{k_d}$', data=kd, valfmt='{x:.1e}')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    plot_twinx(ax=ax,
               y_data=np.exp(mem_param.kp0*volt),
               y_label='kp',
               y2_data=np.exp(-mem_param.kd0*volt), y2_label='kd',
               x_data=volt,
               x_label="Voltage [a.u.]",
               y1_ticks=None,
               title='',
               colors=['blue', 'green'],
               y2_ticks=None,
               y_scale=['linear', 'linear'],
               # save_path=save_path,
               # figname=name_fig + '{:.1f}_meas_current.svg'.format(sweep_vel)
               )
    plt.show()

    a=0
