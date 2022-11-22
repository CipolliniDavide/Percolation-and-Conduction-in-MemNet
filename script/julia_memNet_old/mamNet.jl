using Random, Graphs, MetaGraphs, GraphPlot, Cairo, SparseArrays

sp! = set_prop!
gp = get_prop

Random.seed!(1)


struct MemParam
    kp0::Float64
    kd0::Float64
    eta_p::Float64
    eta_d::Float64
    g_min::Float64
    g_max::Float64
    g0::Float64
end

struct MemNet
    gnd::Int
    src::Int
    mem_param::MemParam 
    #mem_param::Dict{String,Float32}
    #net_param::Dict{String,Float32}
    G::MetaGraph
    src_num::Int
    gnd_num::Int
end

# function to_directed(G::MetaGraph)
#     GDir = MetaDiGraph(DiGraph(adjacency_matrix(G)))
#     #G.graph.update(deepcopy(self.graph))
#     #G.add_nodes_from((n, deepcopy(d)) for n, d in self._node.items())
#     for ed in edges(G)
#         sp!(GDir)
#     end

    
#     G.add_edges_from(
#         (u, v, deepcopy(data))
#         for u, nbrs in self._adj.items()
#         for v, data in nbrs.items()
#     )
#     return G
# end

function get_lapl(G::MetaGraph,wk::Symbol, height, width)
    in::Vector{Int} = []
    out::Vector{Int} = []
    weights::Vector{Float64} = []
    for edge in edges(G)
        append!(in,src(edge))
        append!(out,dst(edge))
        append!(weights, gp(G,edge,wk) )
    end
    adj = sparse(in,out,weights,height,width)
    degree = Diagonal( sum(adj, dims=1)[1,:] )
    return degree - adj
end

function grid_graph(rows::Int, cols::Int = rows)
    g = grid([rows, cols])
    for r in 1:rows
        for c in 1:cols
            k = rand(0:1)
            if k == 0
                add_edge!(g, coordToIdx(c, r, rows), coordToIdx(c + 1, r + 1, rows))
            else
                add_edge!(g, coordToIdx(c + 1, r, rows), coordToIdx(c, r + 1, rows))
            end
        end
    end
return MetaGraph(g)
end

function coordToIdx(i::Int,j::Int, N::Int)
    return (i-1)*N+j
end


function initialize_graph_attributes(Net::MemNet, sourcenode_list, groundnode_list)
    Yin = Net.mem_param.g0
    #Net::MemNet.src_num = len(sourcenode_list)
    #Net::MemNet.gnd_num = len(groundnode_list)
    # add the initial conductance
    for ed in edges(Net.G)
        sp!(Net.G,ed,:Y,Yin)
        sp!(Net.G,ed,:X,0)
        sp!(Net.G,ed,:g,0)
        sp!(Net.G,ed,:deltaV,0)
        sp!(Net.G,ed,:Xlocal,0)
        #Net.G[u][v]['Y'] = Yin  # assign initial ammittance to all edges
        # Net.G[u][v]['Filament'] = false
        #Net.G[u][v]['X'] = 0
        #Net.G[u][v]['Xlocal'] = 0  # assign initial high resistance state in all junctions
        #Net.G[u][v]['deltaV'] = 0
        #Net.G[u][v]['g'] = 0
    end
    
    ##initialize
    for n in vertices(Net.G)
        sp!(Net.G,n,:pad,false)
        sp!(Net.G,n,:source_node,false)
        sp!(Net.G,n,:ground_node,false)
        #vertices(Net.G)[n]['pad'] = false
        #vertices(Net.G)[n]['source_node'] = false
        #vertices(Net.G)[n]['ground_node'] = false
        if n in groundnode_list
            sp!(Net.G,n,:ground_node,true)
            #vertices(Net.G)[n]['ground_node'] = true
        end
        if n in sourcenode_list
            sp!(Net.G,n,:source_node,true)
            #vertices(Net.G)[n]['source_node'] = true
        end
    end
end



function update_edge_weigths(Net::MemNet, delta_t)
    """
    :param Net.G: Net.G is modified
    :param delta_t:
    :return:
    """
    G = Net.G
    for ed in edges(Net.G)
        sp!(G, ed, :deltaV, abs(gp(G, src(ed), :V) - gp(G, dst(ed), :V)))
        #G[u][v]['deltaV'] = abs(G.nodes[u]['V'] - G.nodes[v]['V'])
        sp!(G, ed, :kp, Net.mem_param.kp0 * exp(Net.mem_param.eta_p * gp(G, ed, :deltaV) ))
        #G[u][v]['kp'] = Net.mem_param.kp0 * exp(Net.mem_param.eta_p * G[u][v]['deltaV'])
        sp!(G, ed, :kd, Net.mem_param.kd0 * exp(Net.mem_param.eta_d * gp(G, ed, :deltaV) ))
        #G[u][v]['kd'] = Net.mem_param.kd0 * exp(-Net.mem_param.eta_d * G[u][v]['deltaV'])
        miranda_formula = (gp(G, ed, :kp)/gp(g, ed, :kp) + gp(G, ed, :kd)) * 
        (1 - (1 - (1 + (gp(G, ed, :kd)/ gp(G, ed, :kp))*gp(G, ed, :g)))*exp(-(gp(G, ed, :kp)+gp(G, ed, :kd))*delta_t ))
        sp!(G, ed, :g, miranda_formula)
        #G[u][v]['g'] = (G[u][v]['kp'] / (G[u][v]['kp'] + G[u][v]['kd'])) * (
        #        1 - (1 - (1 + (G[u][v]['kd'] / G[u][v]['kp']) * G[u][v]['g']))
        #        * math.exp(- (G[u][v]['kp'] + G[u][v]['kd']) * delta_t))
        sp!(G, ed, :Y, Net.mem_param.g_min * (1 - gp(G, ed, :g)) + Net.mem_param.g_max * gp(G, ed, :g))
        #G[u][v]['Y'] = Net.mem_param.g_min * (1 - G[u][v]['g']) + Net.mem_param.g_max * G[u][v]['g']

    end
end

mem_param = MemParam(2.555173332603108574e-06,  #kp0
                     6.488388862524891465e+01,  # model kd_0
                     3.492155165334443012e+01,  # model eta_p
                     5.590601016803570467e+00,  # model eta_d
                     1.014708121672117710e-03,  # model g_min
                     2.723493729125820492e-03,  # model g_max
                     1.014708121672117710e-03,  # model g_0
                    )
g = grid_graph(5,5)
gsrc = [2]  # ,30] #[node_map(3, 1, rows, cols), 10] # define a list of source nodes in the range [0, rows*cols-1]
gnd = [22]
mem_net = MemNet(1,15, mem_param, g, 1, 1)
# initialize_graph_attributes(mem_net, src, gnd)

"""mem_param = Dict("kp0"=> 2.555173332603108574e-06,  # model kp_0
             "kd0"=> 6.488388862524891465e+01,  # model kd_0
             "eta_p"=> 3.492155165334443012e+01,  # model eta_p
            "eta_d"=> 5.590601016803570467e+00,  # model eta_d
            "g_min"=> 1.014708121672117710e-03,  # model g_min
            "g_max"=> 2.723493729125820492e-03,  # model g_max
            "g0"=> 1.014708121672117710e-03  # model g_0
            )
"""
