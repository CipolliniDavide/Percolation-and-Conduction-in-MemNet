#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:25:44 2021

@author: hp
"""
import networkx as nx
    
def average_shortest_path_length_for_all(G):
    import math
    #tempgraph=G.copy();
    #tempgraph= G
    if nx.is_connected(G):
        # Normal case, the graph is connected
        average= nx.average_shortest_path_length(G)
        print('Connected graph. Average shortest path: %.4f' %average)
        return average
    else:
        # Try to see if the graph is not connected because of isolated nodes
        #G.remove_nodes_from(nx.isolates(G))
        #if nx.is_connected(G):
            # Compute the graph average path without isolated nodes
        #    average=nx.average_shortest_path_length(G);
        #else:
        # Compute the average shortest path for each subgraph and mean it!
        #subgraphs = [sbg for sbg in nx.connected_component_subgraphs(G) if len(sbg) > 1]
        subgraphs= [G.subgraph(c) for c in nx.connected_components(G) if len(c)>1]
        average=math.fsum(len(sg) * nx.average_shortest_path_length(sg) for sg in subgraphs) / sum(len(sg) for sg in subgraphs)
        print('Disconnected graph. Average shortest path: %.4f' %average)
        return average
    
def hist_shortest_path(G, save_path):
    from matplotlib import pyplot as plt
    lengths=nx.shortest_path(G)
    l_list= [lengths[node][target] for node in G.nodes() for target in lengths[node].keys()]
    print(len(l_list))
    h, b= plt.hist(l_list, bins=20)
    plt.title('Distribution of  Path Lenght across all Node Pairs| %s' %(save_path.split('/')[1]) )
    #plt.xticks(bins_edges)
    plt.xlabel('Path Lenght')
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #plt.ylabel('#')
    if save_path: plt.savefig(save_path+'shortest_path.png')
    plt.close()
    