'''
This class creates a net of MEMRISTORS & RESISTORS: memristors are modeled according to Miranda model.
Code uses SciPy csr_matrix to take advantage of sparsness and symmetry of the matrix.
Computation is made only on the upper triangular matrix.
Diagonals are supposed to be zero (no self link).
'''


import networkx as nx
import sys
import numpy as np
import time
# import jax.numpy as jnp
import copy
# import math
# from multiprocessing import Pool, cpu_count
import random
from scipy.sparse import csgraph, csc_matrix, csr_matrix, save_npz
from scipy import sparse
# from matplotlib import pyplot as plt
from .mem_parameters import mem_param, net_param, sim_param
from .Measure import Measure

class create_graph():

    def define_grid_graph_2(rows, cols, seed, diag=1):
        '''
        Reference : Kevin Montano et al 2022 Neuromorph. Comput. Eng. 2 014007
        Code based on https://github.com/ MilanoGianluca/Grid-Graph_Modeling_Memristive_Nanonetworks
        :param rows:
        :param cols:
        :param seed:
        :return:
        '''
        ##define a grid graph

        Ggrid = nx.grid_graph(dim=[rows, cols])
        if diag == 1:
            print('Add rnd diagonals to grid-graph')
            # Define random diagonals
            if seed:
                random.seed(seed)
            else:
                random.seed(time.time())

            for r in range(rows - 1):
                for c in range(cols - 1):
                    k = random.randint(0, 1)
                    if k == 0:
                        Ggrid.add_edge((c, r), (c + 1, r + 1))
                    else:
                        Ggrid.add_edge((c + 1, r), (c, r + 1))

        ##define a graph with integer nodes and positions of a grid graph
        G = nx.convert_node_labels_to_integers(Ggrid, first_label=0, ordering='default', label_attribute='coord')
        # import itertools
        # grid = list(itertools.product(range(nrows), range(ncols)))
        # coordinates = [(n, {'coord': grid[n]}) for n in G.nodes]
        # nx.set_node_attributes(G, values=dict(coordinates), name='coord')
        return G


class MemNet(Measure):
    def __init__(self, mem_param, net_param, gnd, src, diag=1):
        self.gnd = gnd
        self.src = src
        self.mem_param = mem_param
        self.net_param = net_param

        if mem_param and net_param:
            self.number_of_nodes = int(self.net_param.rows * self.net_param.cols)

            self.G_root = create_graph.define_grid_graph_2(rows=net_param.rows, cols=net_param.cols,
                                                           seed=net_param.seed, diag=diag)
            self.Adj = nx.to_scipy_sparse_array(self.G_root, format='csr')
            self.number_of_edges = self.Adj.data.sum() // 2
            self.adj_indexes = np.argwhere(self.Adj > 0)
            self.triangular_adj_indexes = np.argwhere(sparse.triu(self.Adj, k=0) > 0)
            if not (self.net_param.frac_of_static_elements < 1 and self.net_param.frac_of_static_elements>=0):
                raise ValueError('Range of net_param.frac_of_static_elements should be in range [0,1).')
            self.number_of_dyn_edges = int((1 - self.net_param.frac_of_static_elements) * self.number_of_edges)
            # if self.number_of_dyn_edges != self.number_of_edges:
            print('Mixed Mem-resistor:\n\t{:.1f}% memristor\n\t{:.1f}% resistor'.format(self.number_of_dyn_edges/self.number_of_edges*100,
                                                                             self.net_param.frac_of_static_elements*100))
            self.update_edge_weights = self.update_DynEdge_weights
            self.dynamic_el_index = random.sample(range(self.number_of_edges), k=self.number_of_dyn_edges)
            # else:
            #     self.update_edge_weights = self.update_AllEdge_weights

            # dynamic_elemts = self.triangular_adj_indexes[self.dynamic_el_index]
            self.reset()

    def reset(self):
        self.G = copy.deepcopy(self.G_root)
        self.initialize_graph_attributes()
        # return 1 / self.effective_resistance(nodeA=self.src[0], nodeB=self.gnd[0])

    def initialize_graph_attributes(self):
        # Initialize edge value
        if self.net_param.weight_init == 'rand':
            self.Condmat = csr_matrix((np.random.uniform(low=self.mem_param.g_min, high=self.mem_param.g_max,
                                                         size=self.number_of_edges),
                                       (self.triangular_adj_indexes.transpose()[0, :],
                                        self.triangular_adj_indexes.transpose()[1, :])),
                                      shape=self.Adj.shape)
        else:
            self.Condmat = sparse.triu(self.Adj, k=0) * self.mem_param.g0

        if self.net_param.weight_init == 'good_OC':
            # Good Ohmic resistors
            mask_static_edges = np.ones(self.number_of_edges, dtype=bool)
            mask_static_edges[self.dynamic_el_index] = False
            self.Condmat.data[mask_static_edges] = self.mem_param.g_max

        self.g = np.zeros(self.number_of_dyn_edges)

        # Node Voltage respect to ground
        self.node_Voltage = np.zeros(self.number_of_nodes)

    def delete_from_csr(self, mat, row_indices=[], col_indices=[]):
        """
        Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
        WARNING: Indices of altered axes are reset in the returned matrix
        """
        if not isinstance(mat, csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")

        rows = []
        cols = []
        if row_indices:
            rows = list(row_indices)
        if col_indices:
            cols = list(col_indices)

        if len(rows) > 0 and len(cols) > 0:
            row_mask = np.ones(mat.shape[0], dtype=bool)
            row_mask[rows] = False
            col_mask = np.ones(mat.shape[1], dtype=bool)
            col_mask[cols] = False
            return mat[row_mask][:, col_mask]
        elif len(rows) > 0:
            mask = np.ones(mat.shape[0], dtype=bool)
            mask[rows] = False
            return mat[mask]
        elif len(cols) > 0:
            mask = np.ones(mat.shape[1], dtype=bool)
            mask[cols] = False
            return mat[:, mask]
        else:
            return mat

    def solve_MVNA_for_currents(self, sourcenode_list, groundnode_list, V_list, t):
        supply_list = np.array(sourcenode_list + groundnode_list[1:])
        num_supply = len(supply_list)

        # Definition of matrices
        matL = csr_matrix(csgraph.laplacian(self.Condmat + self.Condmat.T, normed=False))
        matL_reduced = self.delete_from_csr(mat=matL, row_indices=[groundnode_list[0]],
                                            col_indices=[groundnode_list[0]])
        matD = csr_matrix((num_supply, num_supply))
        # fill B
        # b_supp_list = [s if s < groundnode_list[0] else s-1 for s in supply_list]
        matB = csr_matrix((np.ones(num_supply), ([s if s < groundnode_list[0] else s-1 for s in supply_list],
                                                 np.arange(num_supply))),
                          shape=(self.number_of_nodes - 1, num_supply))
        # fill Z
        temp = [i + self.number_of_nodes - 1 for i, v in enumerate(supply_list) if v in sourcenode_list]
        matZ = csr_matrix((V_list[:, t], (temp, [0] * len(sourcenode_list))),
                          shape=(self.number_of_nodes - 1 + num_supply, 1))

        # Compose matrix
        submat1 = sparse.hstack((matL_reduced, matB))
        submat2 = sparse.hstack((matB.transpose(), matD))
        matY = sparse.vstack((submat1, submat2), format='csr')

        self.matX = sparse.linalg.spsolve(matY, matZ)

        # add voltage (respect to gnd) as a node attribute
        mask = np.ones(self.number_of_nodes, bool)
        mask[groundnode_list[0]] = False
        self.node_Voltage[mask] = self.matX[:self.number_of_nodes - 1].reshape(-1)
        # Qui da ottimizzare
        # with Pool(processes=cpu_count()) as pool:
        #     a = pool.map(self.f_deltaV, self.adj_indexes)

        dv = np.zeros(shape=int(self.number_of_edges))
        for i, ind in enumerate(self.triangular_adj_indexes):
            dv[i] = self.node_Voltage[ind[0]] - self.node_Voltage[ind[1]]

        self.dVmat = csr_matrix((dv,
                                 (self.triangular_adj_indexes.transpose()[0, :],
                                  self.triangular_adj_indexes.transpose()[1, :])),
                                shape=self.Adj.shape)

        return (self.Condmat + self.Condmat.T).multiply(self.dVmat - self.dVmat.T > 0)


    def MVNA(self, sourcenode_list, groundnode_list, V_list, t):
        '''

        :param sourcenode_list:
        :param groundnode_list:
        :param V_list:
        :param t:
        :return:
        '''

        supply_list = np.array(sourcenode_list + groundnode_list[1:])
        num_supply = len(supply_list)

        # Definition of matrices
        matL = csr_matrix(csgraph.laplacian(self.Condmat + self.Condmat.T, normed=False))
        matL_reduced = self.delete_from_csr(mat=matL, row_indices=[groundnode_list[0]], col_indices=[groundnode_list[0]])
        matD = csr_matrix((num_supply, num_supply))
        # fill B
        matB = csr_matrix((np.ones(num_supply), ([s if s < groundnode_list[0] else s-1 for s in supply_list],
                                                 #(supply_list,
                                                 np.arange(num_supply))),
                          shape=(self.number_of_nodes - 1, num_supply))
        # fill Z
        temp = [i + self.number_of_nodes - 1 for i, v in enumerate(supply_list) if v in sourcenode_list]
        matZ = csr_matrix((V_list[:, t], (temp, [0]*len(sourcenode_list))),
                          shape=(self.number_of_nodes - 1 + num_supply, 1))

        # Compose matrix
        submat1 = sparse.hstack((matL_reduced, matB))
        submat2 = sparse.hstack((matB.transpose(), matD))
        matY = sparse.vstack((submat1, submat2), format='csr')

        matX = sparse.linalg.spsolve(matY, matZ)

        # add voltage (respect to gnd) as a node attribute
        mask = np.ones(self.number_of_nodes, bool)
        mask[groundnode_list[0]] = False
        self.node_Voltage[mask] = matX[:self.number_of_nodes-1].reshape(-1)
        self.source_current = matX[self.number_of_nodes-1:]

        # Qui da ottimizzare
        # with Pool(processes=cpu_count()) as pool:
        #     a = pool.map(self.f_deltaV, self.adj_indexes)

        dv = np.zeros(shape=int(self.number_of_edges))
        for i, ind in enumerate(self.triangular_adj_indexes):
            dv[i] = self.node_Voltage[ind[0]] - self.node_Voltage[ind[1]]

        self.dVmat = csr_matrix((dv,
                                 (self.triangular_adj_indexes.transpose()[0, :],
                                  self.triangular_adj_indexes.transpose()[1, :])),
                                shape=self.Adj.shape)

        # Return matrix of conductance with directed edges according to current flow
        return (self.Condmat + self.Condmat.T).multiply(self.dVmat - self.dVmat.T > 0)

    def compute_Imat(self):
        I = csr_matrix((np.multiply(self.dVmat.data, self.Condmat.data),
                        (np.array(self.triangular_adj_indexes).transpose()[0, :],
                         np.array(self.triangular_adj_indexes).transpose()[1, :])),
                       shape=self.Adj.shape)
        I = I - I.T
        return I.multiply(I > 0)

    #################  - UPDATE EDGE WEIGHT (Miranda's model)   -    ##############
    def update_AllEdge_weights(self, delta_t):

        self.kp = self.mem_param.kp0 * np.exp(self.mem_param.eta_p * np.abs(self.dVmat.data))
        self.kd = self.mem_param.kd0 * np.exp(-self.mem_param.eta_d * np.abs(self.dVmat.data))

        # a = np.divide(self.kp, self.kp + self.kd)
        temp = np.multiply(1 + np.divide(self.kd, self.kp), self.g)
        b = np.multiply(1-temp, np.exp(- ((self.kp + self.kd) * delta_t)))
        self.g = np.multiply(np.divide(self.kp, self.kp + self.kd), 1 - b)
        self.Condmat.data = self.mem_param.g_min * (1 - self.g) + self.mem_param.g_max * self.g

        # if self.g.min()<0 or self.g.max()>1:
        #     raise ValueError('g out of interval [0,1].\n')
        # if self.Condmat.data.max() > self.mem_param.g_max or self.Condmat.data.min()< self.mem_param.g_min:
        #     raise ValueError('Conductance value out of range.\n')


    def update_DynEdge_weights(self, delta_t):
        # np.multiply can be used only on np.array (or matx.data of scipy.csr sparse matrix)!

        self.kp = self.mem_param.kp0 * np.exp(self.mem_param.eta_p * np.abs(self.dVmat.data[self.dynamic_el_index]))
        self.kd = self.mem_param.kd0 * np.exp(-self.mem_param.eta_d * np.abs(self.dVmat.data[self.dynamic_el_index]))
        #
        # self.kp = self.mem_param.kp0 * np.exp(self.mem_param.eta_p * self.dVmat.data[self.dynamic_el_index])
        # self.kd = self.mem_param.kd0 * np.exp(-self.mem_param.eta_d * self.dVmat.data[self.dynamic_el_index])

        # a = np.divide(self.kp, self.kp + self.kd)
        temp = np.multiply(1 + np.divide(self.kd, self.kp), self.g)
        # temp = 1 + np.multiply(np.divide(self.kd, self.kp), self.g)

        b = np.multiply(1-temp, np.exp(- ((self.kp + self.kd) * delta_t)))
        self.g = np.multiply(np.divide(self.kp, self.kp + self.kd), 1 - b)
        self.Condmat.data[self.dynamic_el_index] = self.mem_param.g_min * (1 - self.g) + self.mem_param.g_max * self.g

        # if self.g.min()<0 or self.g.max()>1:
        #     raise ValueError('g out of interval [0,1].\n')
        # if self.Condmat.data.max() > self.mem_param.g_max or self.Condmat.data.min()< self.mem_param.g_min:
        #     raise ValueError('Conductance value out of range.\n')

    def run(self, t_list, groundnode_list, sourcenode_list, V_list, save_path=None, save_mode_all=0):

        delta_t = t_list[1] - t_list[0]
        net_conductance = np.zeros(len(t_list))
        net_entropy = np.zeros(len(t_list))

        if save_path:
            print('Saving triu conductance matrix in:\n\t{:s}'.format(save_path))
            _ = self.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=0)
            net_conductance[0] = -self.source_current[0] / V_list[0][0]
            net_entropy[0] = self.net_entropy_from_conductances(self.Condmat)
            save_npz(matrix=self.Condmat, file=save_path+'t{:09d}.npz'.format(0))
            # np.save(arr=self.g, file=save_path + 'g_t{:09d}.npy'.format(0))

            # sys.stdout.write("\r\tNetwork Stimulation: {:d}/{:d}".format(1, len(t_list)))
            if save_mode_all == 1:
                for t in range(1, len(t_list)):
                    self.update_edge_weights(delta_t=delta_t)
                    _ = self.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=t)
                    # if (t % 2): pass
                    # else:
                    net_conductance[t] = -self.source_current[0] / V_list[0][t]
                    net_entropy[t] = self.net_entropy_from_conductances(self.Condmat)
                    # net_conductance[t] = 1 / self.effective_resistance(nodeA=self.src[0], nodeB=self.gnd[0])
                    save_npz(matrix=self.Condmat, file=save_path + '/t{:09d}.npz'.format(t))
                    # np.save(arr=self.g, file=save_path + 'g_t{:09d}.npy'.format(0))

            else:
                for t in range(1, len(t_list)):
                    self.update_edge_weights(delta_t=delta_t)
                    _ = self.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=t)
                    # if (t % 2): pass
                    # else:
                    # net_conductance[t] = 1 / self.effective_resistance(nodeA=self.src[0], nodeB=self.gnd[0])
                    net_conductance[t] = -self.source_current[0] / V_list[0][t]
                    net_entropy[t] = self.net_entropy_from_conductances(self.Condmat)

                save_npz(matrix=self.Condmat, file=save_path+'/t{:09d}.npz'.format(t))
                # np.save(arr=self.g, file=save_path + 'g_t{:09d}.npy'.format(t))
                # sys.stdout.write("\r\nNetwork Stimulation completed\n\n")

            np.save(file=save_path + 'net_conductance.npy', arr=net_conductance)
            np.save(file=save_path + 'net_entropy.npy', arr=net_entropy)

        else:
            # if new_start:
            H_list = [[] for t in range(len(t_list))]
            H_list[0] = self.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=0)
            # sys.stdout.write("\r\t\tNetwork Stimulation: {:d}/{:d}".format(1, len(t_list)))
            for t in range(1, len(t_list)):
                self.update_edge_weights(delta_t=delta_t)
                H_list[t] = self.MVNA(groundnode_list=groundnode_list, sourcenode_list=sourcenode_list, V_list=V_list, t=t)
            return H_list
        # sys.stdout.write("\r\t\tNetwork Stimulation: {:d}/{:d}".format(t+1, len(t_list)))

        # H = nx.DiGraph(H_list[0].todense())
        # # H = nx.DiGraph(I)
        # coordinates = [(node, (feat['coord'])) for node, feat in self.G.nodes(data=True)]
        # nx.set_node_attributes(H, dict(coordinates), 'coord')
        # from .visualize import visualize
        # visualize.plot_network(H, show=True, numeric_label=True)


if __name__ == "__main__":
    #root = '/home/davide/PycharmProjects/grid-Graph/Reinforce-Grid-Graph_Modeling_Memristive'
    #################################################
    t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)  # [s]
    ########## Define source and ground pads lists ##########
    src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [390]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    labels = [(gnd[0], 'Gnd'), (src[0], 'Src')]
    ###########################################################
    #plot_dependency_from_voltage()
    #chack_conductance()

    #V_list = np.array([np.array([1]*10), np.array([5]*10)]).transpose()
    src = [10]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [390]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]
    #net_param.rows = 20
    #net_param.cols= 20
    from mem_parameters import volt_param
    V_list = np.asarray([8 if t <= 2e-3 else 0.1 for t in t_list]).transpose()
    src = [2]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
    gnd = [22]  # [20, 0]#409] # define a list of ground nodes in the range [0, rows*cols-1]

    # Instantiate Classes
    net = MemNet(mem_param=mem_param, net_param=net_param, gnd=gnd, src=src)
    net.run(t_list=t_list, sourcenode_list=src, groundnode_list=gnd, V_list=V_list)
