# Show violations in spare TSV solution (OLD method, seed=99)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from spareTSV import formGraph, showPaths, vdcorput, setup_network_flow
from digraphx.mcf import cycle_canceling_mcf


def nx_to_dict_graph(gra, sink):
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
    return g, demands


seed, N, M, r, mu, eta = 99, 150, 45, 5, 0.10, 1.4
T = N + M
xbase, ybase = 2, 3
x = [i for i in vdcorput(T, xbase)]
y = [i for i in vdcorput(T, ybase)]
pos = list(zip(x, y))
gra = formGraph(T, pos, mu, eta, seed=seed)
pos2 = dict(enumerate(pos))

# Initial graph
fig, ax = showPaths(gra, pos2, N)
fig.savefig("spareTSV-seed99-initial.svg")
plt.close(fig)

sink = setup_network_flow(gra, pos, primal_count=N, capacity=r)
g, demands = nx_to_dict_graph(gra, sink)
result = cycle_canceling_mcf(g, demands)

flow_cost, flow_dict = result
print(f"Flow cost: {flow_cost}")

pathlist = [
    (u, v)
    for u in flow_dict
    for v, f in flow_dict[u].items()
    if f > 0 and v != sink
]

# Identify violating edges (nodes with multiple outgoing edges)
outgoing_count = {}
for u, v in pathlist:
    outgoing_count[u] = outgoing_count.get(u, 0) + 1

good_path = []
bad_path = []
for u, v in pathlist:
    if outgoing_count[u] > 1:
        bad_path.append((u, v))
    else:
        good_path.append((u, v))

# Count
outgoing = {}
for u, v in pathlist:
    outgoing[u] = v
violations = len(pathlist) - len(outgoing)
print(f"Path edges: {len(pathlist)}, violations: {violations}")

# Show violating nodes
violators = {u for u, c in outgoing_count.items() if c > 1}
print(f"Violating nodes: {sorted(violators)}")
for u in sorted(violators):
    edges = [(u, v) for (a, v) in pathlist if a == u]
    print(f"  node {u}: -> {edges}")

    gra.remove_node(sink)

    fig, ax = showPaths(gra, pos2, N, path=good_path)
    if bad_path:
        nx.draw_networkx_edges(
            gra, pos2, edgelist=bad_path,
            edge_color="#ff00ff", width=2,
            arrows=True, arrowstyle="->", arrowsize=12,
            ax=ax,
        )
    fig.savefig("spareTSV-seed99-solution.svg")
    plt.close(fig)
