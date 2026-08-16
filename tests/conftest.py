"""Shared test fixtures for spareTSV."""

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend for deterministic headless tests


@pytest.fixture
def sample_positions():
    """Generate Van der Corput positions for a test graph."""
    from spareTSV import vdc

    T = 12  # 9 primal + 3 spare
    xbase, ybase = 2, 3
    x = [vdc(i, xbase) for i in range(T)]
    y = [vdc(i, ybase) for i in range(T)]
    return list(zip(x, y))


@pytest.fixture
def small_graph(sample_positions):
    """A small test geometric graph."""
    from spareTSV import formGraph

    return formGraph(12, sample_positions, 0.12, 1.6, seed=5)
