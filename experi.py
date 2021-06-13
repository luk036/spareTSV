# test

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def vdc(n, base=2):
    vdc, denom = 0.0,1.0
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc

def vdcorput(n, base=2):
    '''
    n - number of vectors
    base - seeds
    '''
    return [vdc(i, base) for i in range(n)]


def formGraph(T,pos,mu,eta,seed=None):
    ''' Form N by N grid of nodes, perturb by mu and connect nodes within eta.
        mu and eta are relative to 1/(N-1)
    '''
    if seed is not None:
        np.random.seed(seed)
    
    N = np.sqrt(T)
    mu = mu/(N-1)
    eta = eta/(N-1)
    
    # generate perterbed grid positions for the nodes
    # pos = [(i + mu*np.random.randn(), j + mu*np.random.randn())\
    #     for i in np.linspace(0,1,N) for j in np.linspace(1,0,N)]
    pos = dict(enumerate(pos))
    n = len(pos)
    
    # connect nodes with edges
    G = nx.random_geometric_graph(n,eta,pos=pos)   
    G = nx.DiGraph(G)
    return G


def showPaths(G,pos,N,edgeProbs=1.0,path=None,visibleNodes=None,guards=None):
    ''' Takes directed graph G, node positions pos, and edge probabilities.
        Optionally uses path (a list of edge indices) to plot the smuggler's path.
        
        edgeProbd gives the probabilities for all the edges, including hidden ones.
        
        path includes all the edges, including the hidden ones
        
        Gnodes and Rnodes denote the source and destination nodes, to be plotted green
        and red respectively.
        
        guards is a list of node indices for denoting guards with a black dot on the plot
    '''
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, aspect='equal')
    
    n = G.number_of_nodes()
    if visibleNodes is None:
        visibleNodes = G.nodes()
    primalNodes = range(0,N)
    spareNodes = range(N,n)
    # draw the regular interior nodes in the graph
    nx.draw_networkx_nodes(G,pos,nodelist=primalNodes,node_color='c',node_size=50,ax=ax)
    nx.draw_networkx_nodes(G,pos,nodelist=spareNodes,node_color='r',node_size=50,ax=ax)
            
    # draw guard nodes
    if guards is not None:
        nx.draw_networkx_nodes(G,pos,nodelist=guards,node_color='.0',node_size=100,ax=ax)
      
        
    if path is None:
        alpha = 1
    else:
        alpha = .15
        
    # start messing with edges
    edge2ind = {e:i for i,e in enumerate(G.edges())}
    ind2edge = {i:e for i,e in enumerate(G.edges())}
    
    # only display edges between non-dummy nodes
    visibleEdges = [i for i in range(len(edge2ind)) if ind2edge[i][0] in visibleNodes and ind2edge[i][1] in visibleNodes]
    
    edgelist = [ind2edge[i] for i in visibleEdges]
    
    if isinstance(edgeProbs,float):
        edgeProbs = [edgeProbs]*G.number_of_edges()
        
    p = [edgeProbs[i] for i in visibleEdges]
    
    # draw edges of graph, make transparent if we're drawing a path over them
    edges = nx.draw_networkx_edges(G,pos,edge_color=p,width=1,
                                   edge_cmap=plt.cm.RdYlGn,arrows=False,edgelist=edgelist,edge_vmin=0.0,
                                   edge_vmax=1.0,ax=ax,alpha=alpha)
        
    # draw the path, only between visible nodes
    if path is not None:
        #visiblePath = [i for i in path if ind2edge[i][0] in visibleNodes and ind2edge[i][1] in visibleNodes]
        #path_pairs = [ind2edge[i] for i in visiblePath]
        #path_colors = [edgeProbs[i] for i in visiblePath]
        edges = nx.draw_networkx_edges(G,pos,edge_color='b',width=1,
                                       edge_cmap=plt.cm.RdYlGn,edgelist=path,arrows=True,edge_vmin=0.0,
                                   edge_vmax=1.0)
    
    ## fig.colorbar(edges,label='??? graph')

    ax.axis([-0.05,1.05,-0.05,1.05])
    #ax.axis('tight')
    #ax.axis('equal')
    ax.axis('off')
    
    return fig, ax


N = 155
M = 40
r = 4

T = N+M
xbase = 2
ybase = 3
x = [i for i in vdcorput(T, xbase)]
y = [i for i in vdcorput(T, ybase)]
pos = list(zip(x,y))
G = formGraph(T,pos,.12,1.6,seed=5)
n = G.number_of_nodes()
pos2 = dict(enumerate(pos))
fig, ax = showPaths(G,pos2,N)
plt.show()

## Add a sink, connect all spareTSV to it.
## pos = pos + [(1.5,.5)]
for u,v in G.edges():
    h = np.array(pos[u]) - np.array(pos[v])
    G[u][v]['weight'] = int(np.sqrt(np.dot(h,h))*100)
    ## G[u][v]['weight'] = 1
    G[u][v]['capacity'] = r
for i in range(N):
    G.nodes[i]['demand'] = -1
for i in range(N,T):
    G.nodes[i]['demand'] = 0
G.add_node(T, demand=N)
# G.add_edges_from([(i, T) for i in range(N,T)]) # weight = 0
for i in range (N,T):
    G.add_edge(i,T,capacity=r) # weight = 0

try:
    flowCost, flowDict = nx.network_simplex(G)
    pathlist = [(u, v) for u in flowDict for v in flowDict[u] if flowDict[u][v] > 0 if v < T]
    #print(pathlist)
    pos2 = dict(enumerate(pos))
    G.remove_node(T)
    fig, ax = showPaths(G,pos2,N,path=pathlist)
    plt.show()
except nx.NetworkXUnfeasible:
    print("Solution Infeasible!")


