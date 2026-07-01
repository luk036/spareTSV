# Show multi-in/out violations (eta=1.1, seed=5)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from spareTSV import formGraph, showPaths, vdcorput, setup_network_flow
from digraphx.mcf import cycle_canceling_mcf


seed, N, M, r, mu, eta = 5, 155, 40, 4, 0.12, 1.1
T = N + M
x = [i for i in vdcorput(T, 2)]
y = [i for i in vdcorput(T, 3)]
pos = list(zip(x, y))
gra = formGraph(T, pos, mu, eta, seed=seed)
pos2 = dict(enumerate(pos))

# Initial graph
fig, ax = showPaths(gra, pos2, N)
fig.savefig("multi-inout-initial.svg")
plt.show()

sink = setup_network_flow(gra, pos, primal_count=N, capacity=r)

g, demands = {}, {}
for u in gra.nodes():
    d = gra.nodes[u].get("demand", 0)
    if d != 0 or u == sink:
        demands[u] = d
for u, v, data in gra.edges(data=True):
    if u not in g:
        g[u] = {}
    g[u][v] = {
        "weight": data.get("weight", 0),
        "capacity": data.get("capacity", 1),
    }

result = cycle_canceling_mcf(g, demands)
flow_cost, flow_dict = result
print(f"Flow cost: {flow_cost}")

pathlist = [
    (u, v)
    for u in flow_dict
    for v, f in flow_dict[u].items()
    if f > 0 and v != sink
]

# Compute in/out degrees
in_deg = {}
out_deg = {}
for u, v in pathlist:
    out_deg[u] = out_deg.get(u, 0) + 1
    in_deg[v] = in_deg.get(v, 0) + 1

# Find multi-in/out nodes
bad_nodes = {
    n for n in set(in_deg) | set(out_deg)
    if in_deg.get(n, 0) > 1 and out_deg.get(n, 0) > 1
}

print(f"Multi-in/out nodes: {sorted(bad_nodes)}")
for n in sorted(bad_nodes):
    ins = [u for u, v in pathlist if v == n]
    outs = [v for u, v in pathlist if u == n]
    print(f"  node {n}: in={ins} out={outs}")

# Classify edges
normal_edges = []
bad_in_edges = []
bad_out_edges = []
for u, v in pathlist:
    if v in bad_nodes:
        bad_in_edges.append((u, v))
    elif u in bad_nodes:
        bad_out_edges.append((u, v))
    else:
        normal_edges.append((u, v))

# Remove sink for visualization
gra.remove_node(sink)

# Solution graph
fig, ax = showPaths(gra, pos2, N, path=normal_edges)

# Draw incoming edges to bad nodes in cyan
if bad_in_edges:
    nx.draw_networkx_edges(
        gra, pos2, edgelist=bad_in_edges,
        edge_color="#00ffff", width=2,
        arrows=True, arrowstyle="->", arrowsize=12,
        ax=ax,
    )

# Draw outgoing edges from bad nodes in magenta
if bad_out_edges:
    nx.draw_networkx_edges(
        gra, pos2, edgelist=bad_out_edges,
        edge_color="#ff00ff", width=2,
        arrows=True, arrowstyle="->", arrowsize=12,
        ax=ax,
    )

# Highlight bad nodes with black circles
if bad_nodes:
    nx.draw_networkx_nodes(
        gra, pos2, nodelist=list(bad_nodes),
        node_color="#000000", node_size=50, ax=ax,
    )

fig.savefig("multi-inout-solution.svg")
plt.show()
