# Test case: s=12, constraint eliminates violation

import random

import matplotlib.pyplot as plt
import networkx as nx
from digraphx.mcf import cycle_canceling_mcf

from spareTSV import formGraph, setup_network_flow

N, M, r = 34, 15, 3
mu, eta, seed = 0.11, 1.70, 12
T = N + M
random.seed(123)
pts = [(random.random(), random.random()) for _ in range(T)]
random.seed(123)  # re-seed so formGraph gets same state
gra = formGraph(T, pts, mu, eta, seed=seed)
sink = setup_network_flow(gra, pts, primal_count=N, capacity=r)

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

# OLD method (no constraint)
fc_o, fd_o = cycle_canceling_mcf(g, demands)
# NEW method (with constraint)
fc_n, fd_n = cycle_canceling_mcf(g, demands, sink=sink)

print(f"OLD: cost={fc_o}")
print(f"NEW: cost={fc_n}")

# Save OLD solution
pathlist_o = [(u, v) for u in fd_o for v, f in fd_o[u].items() if f > 0 and v != sink]
pathlist_n = [(u, v) for u in fd_n for v, f in fd_n[u].items() if f > 0 and v != sink]

# Count violations in OLD
in_d, out_d = {}, {}
for u, v in pathlist_o:
    out_d[u] = out_d.get(u, 0) + 1
    in_d[v] = in_d.get(v, 0) + 1
bad_o = {
    n for n in set(in_d) | set(out_d) if in_d.get(n, 0) > 1 and out_d.get(n, 0) > 1
}
print(f"OLD violations (multi-in/out): {sorted(bad_o)}")
for n in sorted(bad_o):
    ins = [u for u, v in pathlist_o if v == n]
    outs = [v for u, v in pathlist_o if u == n]
    print(f"  node {n}: in={ins} out={outs}")

# Count violations in NEW
in_d2, out_d2 = {}, {}
for u, v in pathlist_n:
    out_d2[u] = out_d2.get(u, 0) + 1
    in_d2[v] = in_d2.get(v, 0) + 1
bad_n = {
    n for n in set(in_d2) | set(out_d2) if in_d2.get(n, 0) > 1 and out_d2.get(n, 0) > 1
}
print(f"NEW violations (multi-in/out): {sorted(bad_n)}")

# Draw solutions
pos2 = dict(enumerate(pts))
gra.remove_node(sink)

# OLD figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect("equal")
ax.axis([-0.05, 1.05, -0.05, 1.05])
ax.axis("off")

nx.draw_networkx_nodes(
    gra, pos2, nodelist=range(N), node_color="#4fc3f7", node_size=50, ax=ax
)
nx.draw_networkx_nodes(
    gra, pos2, nodelist=range(N, T), node_color="#ef5350", node_size=50, ax=ax
)

normal_o = [(u, v) for u, v in pathlist_o if u not in bad_o]
bad_o_edges = [(u, v) for u, v in pathlist_o if u in bad_o]

nx.draw_networkx_edges(
    gra, pos2, edgelist=normal_o, edge_color="#2196f3", width=2, arrows=True, ax=ax
)
if bad_o_edges:
    nx.draw_networkx_edges(
        gra,
        pos2,
        edgelist=bad_o_edges,
        edge_color="#ff00ff",
        width=3,
        arrows=True,
        ax=ax,
    )
if bad_o:
    nx.draw_networkx_nodes(
        gra, pos2, nodelist=list(bad_o), node_color="#000000", node_size=80, ax=ax
    )

fig.savefig("test-s12-old.svg")
plt.close(fig)

# NEW figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect("equal")
ax.axis([-0.05, 1.05, -0.05, 1.05])
ax.axis("off")

nx.draw_networkx_nodes(
    gra, pos2, nodelist=range(N), node_color="#4fc3f7", node_size=50, ax=ax
)
nx.draw_networkx_nodes(
    gra, pos2, nodelist=range(N, T), node_color="#ef5350", node_size=50, ax=ax
)
nx.draw_networkx_edges(
    gra, pos2, edgelist=pathlist_n, edge_color="#81c784", width=2, arrows=True, ax=ax
)

fig.savefig("test-s12-new.svg")
plt.close(fig)
print("Saved: test-s12-old.svg, test-s12-new.svg")
