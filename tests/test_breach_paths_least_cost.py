import numpy as np
import pytest
from overflow.breach_paths_least_cost import breach_paths_least_cost_chunk

# pylint does not understand pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture
def dem_array():
    """A numpy array representing a DEM with a pit at the center and a breach path to the edge."""
    return np.array(
        [
            [1, 1, -2, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ]
    )


def test_breach_paths_least_cost_chunk(dem_array):
    """Test that the expected breach path is created."""
    pits = np.array([[2, 2]])
    breach_paths_least_cost_chunk(pits, dem_array, -9999)
    expected_dem = np.array(
        [
            [1, 1, -2, 1, 1],
            [1, 2, -1, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    assert np.allclose(dem_array, expected_dem)
