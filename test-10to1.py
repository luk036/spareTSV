# Test constraint at 10:1 ratio (N=30 primal, M=3 spare)
import matplotlib

matplotlib.use("Agg")
from digraphx.mcf import cycle_canceling_mcf  # noqa: E402

from spareTSV import formGraph, setup_network_flow, vdcorput  # noqa: E402


def stats(fd):
    pl = [(u, v) for u in fd for v, f in fd[u].items() if f > 0 and v != sink]
    in_d, out_d = {}, {}
    for u, v in pl:
        out_d[u] = out_d.get(u, 0) + 1
        in_d[v] = in_d.get(v, 0) + 1
    mo = sum(1 for n, c in out_d.items() if c > 1)
    mb = sum(
        1 for n in set(in_d) | set(out_d) if in_d.get(n, 0) > 1 and out_d.get(n, 0) > 1
    )
    return mo, mb


N, M, r = 30, 3, 30  # 10:1 ratio, high capacity
T = N + M
x = [i for i in vdcorput(T, 2)]
y = [i for i in vdcorput(T, 3)]
pos = list(zip(x, y))

for seed in range(5, 15):
    for eta in [1.2, 1.5, 1.8, 2.0, 2.5]:
        gra = formGraph(T, pos, 0.12, eta, seed=seed)
        e = gra.number_of_edges()
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
        r_old = cycle_canceling_mcf(g, demands)
        r_new = cycle_canceling_mcf(g, demands, sink=sink)
        if r_old is None:
            print(f"seed={seed} eta={eta}: INFEASIBLE")
            continue
        fc_o, fd_o = r_old
        fc_n, fd_n = r_new
        mo_o, mb_o = stats(fd_o)
        mo_n, mb_n = stats(fd_n)
        print(
            f"seed={seed} eta={eta} e={e}: "
            f"old mb={mb_o} mo={mo_o} cost={fc_o} | "
            f"new mb={mb_n} mo={mo_n} cost={fc_n}"
        )
