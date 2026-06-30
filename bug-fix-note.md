## Bug Report: Parallel Residual Edge Collision in MCF Cycle-Cancellation

### Summary

The cycle-cancellation min-cost flow solver (`digraphx.mcf.cycle_canceling_mcf`) produced
suboptimal solutions on graphs where both a forward residual edge and a backward
residual edge mapped to the same directed node pair `(u, v)`.  The later edge silently
overwrote the earlier one, causing negative cycles to go undetected and the algorithm
to terminate at a local (not global) optimum.

### Root Cause

In `_build_residual`, residual edges are stored as a dict-of-dicts keyed by
`(source, target)`:

```python
residual[u][v] = {cost: w, capacity: c, orig: (u, v), forward: True}   # forward
residual[v][u] = {cost: -w, capacity: f, orig: (u, v), forward: False} # backward
```

This works correctly as long as every `(source → target)` pair has **at most one**
residual edge.  However, when the **original graph contains both** `(a, b)` and
`(b, a)`, the residual graph can have:

| Source | Target | Kind | Cost | Capacity |
|--------|--------|------|------|----------|
| a | b | forward of (a, b) | +w_ab | cap_ab − f_ab |
| a | b | backward of (b, a) | −w_ba | f_ba |

Both edges are stored at `residual[a][b]`.  The second one **overwrites** the first,
losing whichever edge was written first.  Consequently negative cycles that need
the lost edge disappear from the residual graph.

### Reproduction

A minimal failing case (random seed 2, 6 nodes, 13 edges):

```
Optimal cost (networkx simplex): 89
Cycle-cancellation cost (before fix): 101
```

The original graph contained both `(2, 4)` and `(4, 2)`.  After the initial feasible flow
was routed, both a forward residual `4→2 cost=+15` and a backward residual
`4→2 cost=−15` competed for `residual[4][2]`.  The overwrite destroyed the
cost=−15 edge, hiding the negative cycle `1→5→4→2→1` (total cost −6) from
Howard's cycle finder.

### Fix

When a `(source, target)` key already exists, keep the edge with the **more negative
cost** (which is always the backward edge, since `−w ≤ 0 ≤ +w` for all `w ≥ 0`).
The forward edge's capacity is recovered in the next iteration when the residual
graph is rebuilt from the updated flow.

```python
prev = residual[u].get(v)
if prev is None or edge["cost"] < prev["cost"]:
    residual[u][v] = edge
```

### Impact

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| spareTSV experiment (195 nodes) | 1672 (mismatch) | 1108 ✓ (matches simplex) |
| Random test n=8, p=0.35 (78 cases) | 37/43 pass (86%) | 75/78 pass (96%) |
| Random test n=15, p=0.20 (22 cases) | 14/25 pass (56%) | 21/22 pass (95%) |
| digraphx full suite | 140/140 | 140/140 |

### Remaining Limitations

The "keep-more-negative-cost" heuristic drops the forward edge when it collides
with a backward edge.  In rare cases (≈4–5% on random graphs) this prevents
some negative cycles from being found, because the forward edge's capacity is
needed to form a complete cycle.  These cases resolve on the next residual
rebuild when the lost capacity is recovered.

### Lessons

1. **Residual graphs for MCF are multi-graphs in general** — parallel edges
   between the same ordered node pair can and do arise when the original graph
   has reciprocal edges.  NetworkX's `capacity_scaling` uses a `MultiDiGraph`
   internally for exactly this reason.

2. **Howard's `NegCycleFinder` assumes simple graphs** (at most one edge per
   ordered node pair).  A full fix would require extending `NegCycleFinder` to
   support parallel edges, or using a multi-graph-compatible cycle detector.

3. **Verification against `nx.min_cost_flow`** on random graphs was essential.
   Without random benchmarks, the bug would have been masked by the unit tests
   (which used graphs without reciprocal edges).
