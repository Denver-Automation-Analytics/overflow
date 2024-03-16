import pytest
import numpy as np
from overflow.fix_flats import flat_edges
from overflow.constants import (
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_WEST,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_DIRECTION_SOUTH_WEST,
    FLOW_DIRECTION_SOUTH,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_NORTH,
)


@pytest.fixture(name="dem")
def fixture_dem():
    """Create a dem for testing.

    Returns:
        np.ndarray: A dem with a flat that drains
    """
    return np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, -1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )


@pytest.fixture(name="fdr")
def fixture_fdr():
    """Create a dem for testing.

    Returns:
        np.ndarray: A dem with a flat that drains
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
            ],
        ],
        dtype=np.uint8,
    )


@pytest.fixture(name="expected_high_edges")
def fixture_expected_high_edges():
    """Create the expected high edges for the test dem.

    Returns:
        list: A list of tuples containing the expected high edges
    """
    return [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 1),
        (3, 1),
        (4, 1),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
        (5, 4),
    ]


@pytest.fixture(name="expected_low_edges")
def fixture_expected_low_edges():
    """Create the expected low edges for the test dem.

    Returns:
        list: A list of tuples containing the expected low edges
    """
    return [
        (5, 1),
        (5, 2),
        (5, 3),
    ]


def test_flat_edges(dem, fdr, expected_high_edges, expected_low_edges):
    """Test the flat_edges function.

    Args:
        dem (np.ndarray): A dem with a flat that drains
        fdr (np.ndarray): A flow direction raster
        expected_high_edges (list): A list of tuples containing the expected high edges
        expected_low_edges (list): A list of tuples containing the expected low edges
    """
    high_edges, low_edges = flat_edges(dem, fdr)
    assert sorted(high_edges) == sorted(expected_high_edges)
    assert sorted(low_edges) == sorted(expected_low_edges)
