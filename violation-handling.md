## Violation Handling in Spare-TSV Cycle-Cancellation Descent

### Problem

The spare-TSV min-cost flow problem has a vertex-disjointness constraint: each
non-sink node may have at most one outgoing edge in the solution path set.
Equivalently, if `e(v1, v2)` is part of the solution, then `e(v1, v3)` cannot
also be part of the solution.

Without this constraint, the standard MCF formulation (cycle-cancellation
descent via `NegCycleFinderQ` with Howard's algorithm) produces optimal-cost
solutions that may violate the vertex-disjointness condition on some problem
instances.

### Case Study: seed=99

- Parameters: N=150 primal, M=45 spare, r=5 capacity, mu=0.10, eta=1.4
- Number of violations: 1 (old method, no constraint)
- Violating node: 136 (primal node)
- Two outgoing edges: 136 → 24 and 136 → 100

### How violations arise

1. **BFS initial routing**: The `_find_feasible_flow` function uses a greedy
   BFS from each supply node to a demand node.  For primal nodes with supply > 1
   or when the BFS routes multiple supply units through the same intermediate
   node, the same node may end up with multiple outgoing flow edges.  This is
   the primary source of violations.

2. **Cycle-cancellation amplification**: Even when the BFS flow has few
   violations, subsequent cycle cancellations can introduce additional ones.
   A negative cycle may add a forward edge from a node that already has an
   outgoing edge, creating a new violation.

### The VertexFilter approach

A `VertexFilter` functor was introduced to track "used" nodes (those with an
outgoing flow edge from an *accepted* cycle) and reject negative cycles that
would add a new outgoing edge to an already-used node:

- `cycle_uses_used_node()`: checks whether any forward edge in a cycle
  originates from a node already in the used set.

- `accept_cycle()`: marks forward-edge source nodes as used, and removes
  nodes from used when all their outgoing flow drops to zero.

- `used` set persists **across iterations** (moved outside the while loop)
  to prevent re-adding edges to previously-used nodes.

### Results

| Method | Cost | Violations | Notes |
|--------|------|------------|-------|
| Old (no constraint) | 977 | 1 | BFS violation at node 136 persists |
| New (with VertexFilter) | 1153 | 3 | Worse cost; BFS violations can't be fixed |

The persistent used set prevents **new** violations from being introduced during
cycle cancellation (the node-136 violation is a BFS artifact, not introduced by
cancellation).  However it also prevents cycles that would **fix** existing
BFS violations, because any cycle involving a used node is rejected.  This
causes the algorithm to converge to a suboptimal solution with higher cost and
more unresolved violations.

### Why the algorithm is not "smart enough"

1. **The used set is a global, irreversible lock**: once a node is marked as
   used, no cycle involving that node as a forward-edge source can be accepted —
   even if the cycle would reduce the total number of violations or improve
   cost.

2. **BFS initial flow is unconstrained**: the greedy BFS creates violations
   that the cycle-cancellation phase cannot fix under the current filter.

3. **One cycle per relax pass**: Howard's algorithm yields one negative cycle
   per relaxation pass.  If the first cycle found is rejected by the filter,
   subsequent cycles in the same pass (which might have different edge
   structures) are explored only if the generator continues.

4. **No "fix-up" mechanism**: there is no logic to detect and resolve existing
   violations during cycle cancellation.  A cycle that removes one outgoing
   edge from a violated node while adding another would be accepted (net
   change = 0 for that node), but such cycles are rare.

### Figures

- `spareTSV-seed99-initial.svg`: Initial geometric graph (cyan=primal, red=spare)
- `spareTSV-seed99-solution.svg`: MCF solution with violations highlighted
  - Blue arrows: valid path edges (one per node)
  - Magenta arrows: violating edges (node 136 has two outgoing edges)

### Potential improvements

1. **Constraint-aware initial flow**: modify BFS routing to avoid creating
   nodes with multiple outgoing edges from the start.

2. **Soft constraint**: allow temporary violations during cycle cancellation,
   enforcing the constraint only at termination (via a post-processing
   cleanup).

3. **Successor-based cycle finding**: `NegCycleFinderQ.howard_succ()` uses
   successor relaxation, which may find cycles that fix violations more
   effectively than predecessor-based Howard's.

4. **Multiple candidate cycles per pass**: accept the first cycle that
   *reduces* the number of violations (or the one that increases cost the
   least), rather than rejecting all violating cycles outright.
