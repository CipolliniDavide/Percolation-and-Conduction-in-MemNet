from script_Opt.Class_SciPySparse.mem_parameters import sim_param, volt_param, net_param, mem_param
from script_Opt.Class_SciPySparse.MemNetwork_mixed import MemNet, Measure
from script_Opt.Class_SciPySparse.ControlSignal import ControlSignal
from script_Opt.Class_SciPySparse.utils import utils, create_dataset, getListOfFiles
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm, contrib
import networkx as nx
import sys
import copy
from glob import glob
from os.path import isfile, join, abspath
from os import pardir
import time
from scipy.sparse import csgraph, save_npz, load_npz

from create_plots_conductance import plot_ent_vs_fracMem
from script_Opt.Class_SciPySparse.visual_utils import set_ticks_label, set_legend, create_colorbar
import matplotlib.pyplot as plt

def plot_heatmap_ent(df, ykey, yloop,
                    aspect=1,
                    valfmt_ylab='{x:.0e}',
                    key='Ent', ylabel=r'$\mathbf{\sigma}$', save_path='./', name_fig='',
                    loc=3, normalize=False, error='sem'):
    frac_of_mem_el = np.round(np.sort(np.unique(np.array(df['frac_of_mem_elements']))), decimals=2)

    array = np.zeros((len(yloop), len(frac_of_mem_el)))
    array_specHeat = np.zeros((len(yloop), len(frac_of_mem_el)))
    # df_grr = df.groupby(by=['ratio', 'frac_of_static_elements']).mean()
    y_data = df
    for i, r in enumerate(yloop):
        # if i % 2 ==0:
        y = [y_data.loc[(y_data[ykey] == r) & (y_data['frac_of_mem_elements'] == fr)][key] for fr in frac_of_mem_el]
        if normalize:
            y_norm = [(y[i] - np.min(y)) / (np.max(y) - np.min(y)) for i in range(len(y))]
            yplot = y_norm
        else:
            yplot = y
        array[i] = np.array([yplot[i].mean() for i in range(len(y))])
        array_specHeat[i] = np.array([np.power(yplot[i], 2).mean() - np.power(yplot[i].mean(), 2) for i in range(len(y))])
    fig = plt.figure('Entropy', figsize=(10, 8))
    ax = fig.subplots(nrows=1, ncols=1)
    im = ax.imshow(1-array, origin='lower', interpolation='bicubic', aspect=aspect)
    set_ticks_label(ax=ax, ax_label='p\nMemristor density', data=frac_of_mem_el, ax_type='x', num=5, valfmt="{x:.1f}",
                    ticks=np.arange(len(frac_of_mem_el)), tick_lab=frac_of_mem_el)
    ticks = np.linspace(start=0, stop=len(yloop), num=5, endpoint=True, dtype=int)
    ticks[-1] = ticks[-1] - 1
    set_ticks_label(ax=ax, ax_label=ylabel, data=np.arange(len(yloop)), ax_type='y', #num=5,
                    valfmt=valfmt_ylab,
                    ticks=ticks,
                    tick_lab=yloop[ticks],
                    )
    cbar_edg = fig.colorbar(mappable=im)
    set_ticks_label(ax=cbar_edg.ax, ax_label=r'$\mathbf{1-\sigma}$',
                    data=1-array.reshape(-1),
                    ax_type='y',
                    num=5, valfmt="{x:.2f}")
                    # ticks=np.arange(len(frac_of_mem_el)), tick_lab=frac_of_mem_el)
    # create_colorbar(fig, ax, mapp=im,
    #                 array_of_values=1-array,
    #                 valfmt="{x:.2e}",
    #                 fontdict_cbar_label={'label': r'$\mathbf{1-\sigma}$'},
    #                 fontdict_cbar_tickslabel={'fontweight':'bold'}, fontdict_cbar_ticks=None)
    plt.tight_layout()
    plt.savefig(join(save_path + '{:s}_{:s}.svg'.format(name_fig, key)), format='svg', dpi=1200)
    # plt.show()
    plt.close('all')

    # fig = plt.figure('Entropy', figsize=(10, 8))
    # ax = fig.subplots(nrows=1, ncols=1)
    # im = ax.imshow(array_specHeat, origin='lower', interpolation='bicubic')
    # set_ticks_label(ax=ax, ax_label='p', data=frac_of_mem_el, ax_type='x', num=5, valfmt="{x:.2f}",
    #                 ticks=np.arange(len(frac_of_mem_el)), tick_lab=frac_of_mem_el)
    # set_ticks_label(ax=ax, ax_label=ylabel, data=yloop, ax_type='y', num=5,
    #                 valfmt="{x:.1e}",
    #                 aspect=.2,
    #                 ticks=np.arange(len(yloop)),
    #                 tick_lab=yloop,
    #                 # tick_lab=np.concatenate((np.array([2]),
    #                 #                       [1e2, 1e4, 1e7, 1e10]
    #                 # np.logspace(start=2, stop=10, num=4, endpoint=True)
    #                 # ))
    #                 )
    # create_colorbar(fig, ax, mapp=im, array_of_values=array_specHeat, valfmt="{x:.1e}",
    #                 fontdict_cbar_label={'label': r'$\mathbf{<\sigma^2> - <\sigma>^2 }$'},
    #                 fontdict_cbar_tickslabel={'fontweight': 'bold'}, fontdict_cbar_ticks=None)
    # plt.tight_layout()
    # plt.savefig(join(save_path + 'SpecHeat{:s}_{:s}.png'.format(key, name_fig)))
    # # plt.show()
    # plt.close('all')



def Small_net_entropy_from_conductances(adj_matrix):
    '''
    Measure entropy as defined by Small et al., i.e. mean of the entrodpy of the node degree distribution.
    Adj matrix must be triangular! We transform it back to symmetric in the code below
    :return:
    '''
    from sklearn.preprocessing import normalize
    w_normalized = normalize(adj_matrix + adj_matrix.T, norm='l1', axis=1)

    # log_p = copy.deepcopy(w_normalized)
    # log_p.data = np.log(log_p.data)
    # node_entropy = - w_normalized.multiply(log_p).sum(axis=1)
    # net_entropy = node_entropy.mean()

    net_entropy = - np.multiply(w_normalized.data, np.log(w_normalized.data)).sum()/adj_matrix.shape[0] #/np.log(adj_matrix.shape[0]-1)
    return net_entropy

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
    parser.add_argument('-svp', '--save_path', default='OutputGridAdiabaticII', type=str)
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

    number_of_nodes = args.linear_size ** 2
    number_of_edges = 2 * args.linear_size ** 2 - 2 * args.linear_size

    save_path_sim = join(root,
                         '{:}/L{:d}/NetSim_StartEnd/{:}/'.format(args.save_path,
                                                                args.linear_size,
                                                                net_param.weight_init))


    save_path_ds = join(root, '{:s}/L{:d}/Entropy_ADJ/DS/'.format(args.save_path, args.linear_size))
    print('Save to:\n\t{:s}'.format(save_path_ds))
    utils.ensure_dir(save_path_ds)


    if args.compute_dataset == 1:

        # m = Measure()
        list_of_fold = glob(save_path_sim + "Vbias*/frac*/ratio*/batch*/", recursive=True)
        # list_of_fold = glob('/Users/dav/PycharmProjects/MemNet/OutputGrid/NetSim_StartEnd/None/Vbias11.0/frac*/ratio*/batch*/',recursive=True)
        for i, fold in enumerate(tqdm(list_of_fold)):
            i_dic = utils.pickle_load(glob(fold+'/*.pickle')[0])
            vbias = float(fold.split('Vbias')[1].rsplit('/')[0])

            matx_l = [load_npz(m) for m in sorted(glob(fold + '/*.npz'))]
            tupl = (i_dic, {'Vbias': vbias,
                            'Ent': net_entropy_from_conductances(matx_l[1]),
                            'MaxEnt': net_entropy_from_conductances(matx_l[0]),
                            'Small_MaxEnt': Small_net_entropy_from_conductances(matx_l[0]),
                            'Small_Ent': Small_net_entropy_from_conductances(matx_l[1]),
                            }
                    )

            utils.pickle_save(obj=utils.merge_dict(tuple=tupl), filename=save_path_ds+'{:09d}.pickle'.format(i))

    df = create_dataset(load_dir=save_path_ds,  # .rsplit('/', 1)[0],
                        save_dir=None,
                        save_name=None,
                        sheet_name='Sheet1',
                        remove_nan_inf=False,
                        remove_labels=[])
    df['frac_of_static_elements'] = np.round(df['frac_of_static_elements'], decimals=2)
    df['frac_of_mem_elements'] = df.apply(lambda row: np.round(1-row.frac_of_static_elements, decimals=2), axis=1)
    df['CE'] = df.apply(lambda row: row.Ent / row.MaxEnt, axis=1)
    df['Small_CE'] = df.apply(lambda row: row.Small_Ent / row.Small_MaxEnt, axis=1)
    df['Ent'] = df.apply(lambda row: row.Ent / np.log(number_of_edges), axis=1)

    # df['G_wtap'] = df.apply(lambda row: row.cols * row.G / (row.g_min*row.ratio), axis=1)
    # df['G_star_diff'] = df.apply(lambda row: row.G - (cols/row.g_min*row.ratio), axis=1)
    # df['G_star_star'] = df.apply(lambda row: 1 - row.cols * row.G / (row.g_min * row.ratio), axis=1)
    # df['G_star_Ventra'] = df.apply(lambda row: (row.G - row.g_min) / (row.g_max - row.g_min), axis=1)

    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures/Vbias' + save_path_ds.split('Vbias')[1]

    df = df.loc[df['batch'] < 100]
    save_path_figures = save_path_ds.split('DS')[0] + 'FiguresBatch100/'
    save_path_figures = save_path_ds.split('DS')[0] + 'FiguresBatch100OnlyP0.3_0.7/'

    # save_path_figures = save_path_ds.split('DS')[0] + 'Figures/'
    utils.ensure_dir(save_path_figures)


    df = df[df['ratio'].isin([2, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])] #1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    df = df[df['frac_of_mem_elements'].isin([.3, .7])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]

    # df = df[df['ratio'].isin([1e4])]  # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])]
    for key in ['Ent', 'CE', 'Small_Ent', 'Small_CE']:
    ### P legend
        save_path_figures_PLeg = save_path_figures + 'Ratio_fixed/'
        save_path_figures_G_wtap = save_path_figures_PLeg + '{:s}/'.format(key)
        utils.ensure_dir(save_path_figures_G_wtap)

        for r in np.round(np.sort(np.unique(np.array(df['ratio'])))):
            save_path_heatmap = save_path_figures_G_wtap + '/HeatMap/'
            utils.ensure_dir(save_path_heatmap)
            plot_heatmap_ent(df=df[df['ratio'].isin([r])],
                             key=key,
                             ykey='Vbias',
                             valfmt_ylab='{x:.2f}',
                             aspect=.5,
                             ylabel='Voltage input\n'+r'$\mathbf{V}$ [a.u.]',
                             save_path=save_path_heatmap + 'Ratio{:.0e}'.format(r),
                             yloop=np.arange(1, 16, 1), #np.sort(df['Vbias'].unique()),
                             loc=3,
                             normalize=False,
                             error='sem')

            plot_ent_vs_fracMem(df=df[df['ratio'].isin([r])], # 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])],
                                key_y=key,
                                key_x='Vbias',
                                key_legend='frac_of_mem_elements',
                                ylabel='Entropy\n'+r'$\mathbf{\sigma}$',
                                save_path=save_path_figures_G_wtap,
                                normalize=False,
                                ymin_max=None,
                                xlabel='V [a.u.]\nVoltage input',
                                legendlabel='Memristor density\np',
                                legend_loop=np.round(np.sort(np.unique(np.array(df['frac_of_mem_elements']))), decimals=2),
                                x_loop=np.sort(np.unique(np.array(df['Vbias']))),
                                name_fig='Ratio{:.0e}'.format(r),
                                error='sem')

        ############################ Ratio legend ####################################################
        save_path_figures_RatioLeg = save_path_figures + 'P_fixed/'
        save_path_figures_G_star = save_path_figures_RatioLeg + '{:s}/'.format(key)
        utils.ensure_dir(save_path_figures_G_star)

        for f in np.round(np.sort(np.unique(np.array(df['frac_of_mem_elements']))), decimals=2):
            plot_ent_vs_fracMem(df=df[df['frac_of_mem_elements'].isin([f])],
                                key_y=key,
                                # normalize=True,
                                normalize=False,
                                key_x='Vbias',
                                key_legend='ratio',
                                ylabel='Entropy\n'+r'$\mathbf{\sigma}$',
                                save_path=save_path_figures_G_star,
                                xlabel='V [a.u.]\nVoltage input',
                                legendlabel=r'$\mathbf{G_{max}/G_{min}}$',
                                # frac_of_mem_el=np.round(np.sort(np.unique(np.array(df['frac_of_mem_elements']))), decimals=2),
                                # ratio=np.sort(df['ratio'].unique()),
                                legend_loop=np.sort(np.unique(np.array(df['ratio']))),
                                x_loop=np.sort(np.unique(np.array(df['Vbias']))),
                                name_fig='p_{:.2f}'.format(f),
                                error='sem')


        ########################### Fixed Voltage #########################
        save_path_figures_vbias = save_path_figures + 'Vbias_fixed/'

        save_path_figures_vbias_wtap = save_path_figures_vbias + '{:s}/'.format(key)
        utils.ensure_dir(save_path_figures_vbias_wtap)
        for v in np.round(np.sort(np.unique(np.array(df['Vbias']))), decimals=2):
            save_path_heatmap = save_path_figures_vbias_wtap + '/HeatMap/'
            utils.ensure_dir(save_path_heatmap)
            plot_heatmap_ent(df=df[df['Vbias'].isin([v])],
                             key=key,
                             ykey='ratio',
                             ylabel='Interaction strength\n'+r'$\mathbf{G_{max}/G_{min}}$',
                             save_path=save_path_heatmap + 'Vbias{:.2f}'.format(v),
                             yloop=np.sort(df['ratio'].unique()),
                             loc=3,
                             normalize=False,
                             error='sem')

            plot_ent_vs_fracMem(df=df[df['Vbias'].isin([v])],
                                key_y=key,
                                # normalize=True,
                                normalize=False,
                                key_x='frac_of_mem_elements',
                                key_legend='ratio',
                                ylabel='Entropy\n'+r'$\mathbf{\sigma}$',
                                save_path=save_path_figures_vbias_wtap,
                                xlabel='Memristor density\n'+'p',
                                legendlabel='Interaction strength\n'+r'$\mathbf{G_{max}/G_{min}}$',
                                # frac_of_mem_el=np.round(np.sort(np.unique(np.array(df['frac_of_mem_elements']))), decimals=2),
                                # ratio=np.sort(df['ratio'].unique()),
                                legend_loop=np.sort(np.unique(np.array(df['ratio']))),
                                x_loop=np.sort(np.unique(np.array(df['frac_of_mem_elements']))),
                                name_fig='Vbias{:.2f}'.format(v),
                                error='sem'
                                )

    a=0