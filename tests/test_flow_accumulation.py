import pytest
import numpy as np
from overflow.flow_accumulation import single_tile_flow_accumulation
from overflow.constants import (
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_NORTH,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_WEST,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_EXTERNAL,
)


@pytest.fixture(name="fdr")
def fixture_fdr():
    """Example FDR from R. Barnes Paper
    https://arxiv.org/pdf/1608.04431.pdf
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
            ],
            [
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
            ],
            [
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
            ],
            [
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH_WEST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_WEST,
            ],
        ],
        dtype=np.int64,
    )


def test_single_tile_flow_accumulation(fdr: np.ndarray):
    """Test a single tile flow accumulation"""
    fac, links = single_tile_flow_accumulation(fdr)
    expected_fac = np.array(
        [
            [1, 27, 1, 1, 2, 11, 1],
            [3, 21, 2, 1, 5, 4, 1],
            [2, 20, 1, 1, 3, 2, 1],
            [1, 2, 17, 14, 1, 1, 1],
            [2, 1, 1, 1, 13, 2, 1],
            [1, 1, 1, 1, 6, 4, 1],
            [1, 1, 1, 3, 1, 2, 1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(fac, expected_fac)
    # assert the perimeter cells in links are as expected
    np.testing.assert_array_equal(links[0, 0], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[0, 1], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[0, 2], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[0, 3], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[0, 4], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[0, 5], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[0, 6], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[1, 0], (0, 1))
    np.testing.assert_array_equal(links[2, 0], (0, 1))
    np.testing.assert_array_equal(links[3, 0], (0, 1))
    np.testing.assert_array_equal(links[4, 0], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[5, 0], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[6, 0], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[1, 6], (0, 5))
    np.testing.assert_array_equal(links[2, 6], (0, 5))
    np.testing.assert_array_equal(links[3, 6], (0, 5))
    np.testing.assert_array_equal(links[4, 6], (0, 1))
    np.testing.assert_array_equal(links[5, 6], (0, 1))
    np.testing.assert_array_equal(links[6, 6], (0, 1))
    np.testing.assert_array_equal(links[6, 1], FLOW_EXTERNAL)
    np.testing.assert_array_equal(links[6, 2], (0, 1))
    np.testing.assert_array_equal(links[6, 3], (0, 1))
    np.testing.assert_array_equal(links[6, 4], (0, 1))
    np.testing.assert_array_equal(links[6, 5], (0, 1))
