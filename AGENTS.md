# AGENTS.md - Agent Guidelines for spareTSV

## Build/Lint/Test Commands

### Testing
```bash
# Run all tests
pytest
# Or via tox
tox

# Run single test
pytest tests/test_spare_tsv.py::test_vdc

# Run with verbose output (default) and coverage (default addopts)
pytest --cov spareTSV --cov-report term-missing --verbose
```

### Linting & Formatting
```bash
# Run all pre-commit hooks (recommended before committing)
pre-commit run --all-files

# Individual tools
black src/spareTSV tests/   # Format code (line length 256)
isort src/spareTSV tests/   # Sort imports (Black profile)
flake8 src/spareTSV tests/  # Lint (max_line_length=256)
```

### Build
```bash
tox -e build                 # Build sdist + wheel
tox -e clean                 # Remove build artifacts
python -m build .            # Alternative build
```

## Code Style Guidelines

### Project Structure
- Source: `src/spareTSV/` (namespace package) - a single small module `spare_tsv.py`
- Tests: `tests/` with one test file `test_spare_tsv.py`
- Version managed via setuptools_scm (version_scheme=no-guess-dev)

### Naming Conventions
- **Functions**: camelCase for setup/IO helpers (e.g., `formGraph`, `showPaths`, `setup_network_flow`), lowercase for sequence utilities (`vdc`, `vdcorput`, `vdcorput_iter`)
- **Parameters**: camelCase (e.g., `edgeProbs`, `visibleNodes`, `primal_count`, `sink_node`)
- **Variables**: lowercase or short names (e.g., `gra`, `pos`, `N`, `eta`)
- **Constants**: UPPER_CASE where used (e.g., `sink` node conventions)

### Type Hints
- Currently minimal/absent in the source module - optional in this small package
- Prefer adding type hints to new functions where reasonable

### Docstrings
- **Style**: One-line docstrings for simple helpers; NumPy-style `Parameters` sections for complex functions (e.g., `showPaths`)
- Include concise `Returns`/`Yields` notes where helpful

### Imports
- **Order**: stdlib → third-party → local
- Third-party: `matplotlib.pyplot`, `networkx` (as `nx`), `numpy` (as `np`)

### Error Handling
- Use `try/except` around `nx.network_simplex` catching `nx.NetworkXUnfeasible`
- Return `None` for infeasible problems instead of raising (see `solve_network_flow`)
- Keep preconditions simple; minimal explicit assertion usage

### Testing Patterns
- **Framework**: pytest with coverage enabled in `setup.cfg`
- Shared fixtures in `tests/conftest.py` (`sample_positions`, `small_graph`)
- Plain `assert` statements, `test_` prefix

### Python Version
- CI targets Python 3.10 (python-app workflow)
- `python_requires` >= 3.10 (importlib-metadata only for <3.10)

### Pre-commit Hooks (Active)
- trailing-whitespace
- check-added-large-files
- check-ast
- check-json / check-yaml / check-xml
- check-merge-conflict
- debug-statements
- end-of-file-fixer
- requirements-txt-fixer
- mixed-line-ending (auto-fix)
- isort
- black
- flake8

### Configuration Files
- `setup.cfg`: Package metadata, pytest options (--cov spareTSV, --verbose), flake8 (max_line_length=256)
- `pyproject.toml`: Build system (setuptools_scm)
- `tox.ini`: Test environments (default, build, clean)
- `.isort.cfg`: Import sorting (Black profile, known_first_party=spareTSV)
- `.coveragerc`: Coverage reporting (branch coverage, excludes repr/debug/asserts)
- `environment.yml` / `requirements.txt`: Dependency pins

## Key Project Context

spareTSV is a small library for spare through-silicon via (TSV) network optimization:
- **Graph modeling**: Builds NxN geometric grids (`formGraph` via `nx.random_geometric_graph`) with Van der Corput quasi-random positions (`vdc`, `vdcorput`)
- **Flow optimization**: `setup_network_flow` configures node demands/capacities and adds a sink; `solve_network_flow` runs `nx.network_simplex` to find spare-to-primal paths
- **Visualization**: `showPaths` renders primal/spare nodes and flow paths with matplotlib
- **Key dependencies**: `networkx` (graphs, simplex solver), `numpy` (RNG), `matplotlib` (plotting)
