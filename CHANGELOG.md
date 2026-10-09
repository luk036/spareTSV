# Changelog

All notable changes to spareTSV will be documented in this file.

## Version 0.1 (2026-10-09)

Initial PyPI release of `spareTSV` — spare through-silicon-via (TSV) network optimization in Python.

### Features
- **Graph modeling**: `formGraph` builds NxN geometric grids (`nx.random_geometric_graph`) with Van der Corput quasi-random positions (`vdc`, `vdcorput`, `vdcorput_iter`). (#303ccdb)
- **Flow optimization**: `setup_network_flow` configures node demands/capacities and adds a sink; `solve_network_flow` runs `nx.network_simplex` to find spare-to-primal paths, returning `None` on infeasible problems. (#0b3e0b9, #569bf70)
- **Visualization**: `showPaths` renders primal/spare nodes and flow paths with matplotlib, plus multi-in/out visualization and ratio-test scripts. (#ee720d8, #dcf91d6)
- **Modern package layout**: Migrated to a PyScaffold-style `src/spareTSV/` package with `tests/`, `tox`, and CI. (#303ccdb)

### Performance
- **Reduced allocation churn** in the spare-TSV hot path. (#3754391)

### Testing & Code Quality
- **100% coverage**: Added unit tests and a `vdcorput_iter` coverage test (96% → 100%). (#d14176c, #5ea3ee7)
- **Flaky-Tk fix**: Forced the matplotlib `Agg` backend in the test suite. (#29d491c)

### Documentation
- **AGENTS.md**: Added agent guidelines for the repository. (#289844f)
- **SVG diagrams**: Refreshed the project diagrams. (#37e6529, #badad40)
- **Violation analysis**: Added a bug-fix report plus old/new comparison scripts. (#dcf91d6, #be01c2a)

### Build & CI
- **Updated GitHub Actions versions**. (#ac7326b)
- **Pre-commit compliance**: Fixed the JSON / flake8 / isort / black violations. (#ea6dced)
