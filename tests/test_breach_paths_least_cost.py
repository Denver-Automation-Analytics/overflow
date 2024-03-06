import numpy as np
import pytest
from overflow.breach_paths_least_cost import breach_paths_least_cost_chunk

# pylint does not understand pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def dem_with_pit():
    """A numpy array representing a DEM with a pit at the center and a breach path to the edge."""
    return np.array(
        [
            [1, 1, -2, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def dem_with_pit_and_nodata():
    """A numpy array representing a DEM with a pit at the center and a breach path to the edge."""
    return np.array(
        [
            [1, 1, -9999, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )


def test_breach_paths_least_cost_chunk(dem_with_pit):
    """Test that the expected breach path is created."""
    pits = np.array([[2, 2]])
    breach_paths_least_cost_chunk(pits, dem_with_pit, -9999)
    expected_dem = np.array(
        [
            [1, 1, -2, 1, 1],
            [1, 2, -1, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    assert np.allclose(dem_with_pit, expected_dem)


def test_breach_paths_least_cost_chunk_with_nodata(dem_with_pit_and_nodata):
    """Test that the expected breach path is created."""
    pits = np.array([[2, 2]])
    breach_paths_least_cost_chunk(pits, dem_with_pit_and_nodata, -9999)
    expected_dem = np.array(
        [
            [1, 1, -9999, 1, 1],
            [1, 2, -0.01, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    assert np.allclose(dem_with_pit_and_nodata, expected_dem)
