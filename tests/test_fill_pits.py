import numpy as np
import pytest
from numba.typed import Dict
from numba import types
from overflow.fill_pits import priority_flood_tile, make_sides, handle_edge, handle_corner


@pytest.fixture(name="dem_values")
def fixture_dem_values():
    """Create a small 5x5 raster for testing"""
    np.random.seed(32)
    return np.random.randint(10, 50, size=(5, 5)).astype(np.float32)


@pytest.fixture(name="expected_filled_dem_values")
def fixture_expected_filled_dem_values(dem_values):
    """Create a small 5x5 raster for testing"""
    filled_dem = dem_values.copy()
    filled_dem[1, 1] = 14
    filled_dem[2, 2] = 14
    filled_dem[2, 3] = 14
    return filled_dem


@pytest.fixture(name="expected_filled_dem_labels")
def fixture_expected_filled_dem_labels():
    """Create a small 5x5 raster for testing"""
    return np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 3, 3, 3, 2],
        ]
    ).astype(int)

@pytest.fixture(name="dem_values_with_nodata")
def fixture_dem_values_with_nodata():
    """Create a small 5x5 raster for testing"""
    dem = np.array(
        [
            [-9999, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, -9999, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
        ]
    ).astype(np.float32)
    return dem

@pytest.fixture(name="expected_filled_dem_values_with_nodata")
def fixture_expected_filled_dem_values_with_nodata():
    """Create a small 5x5 raster for testing"""
    return np.array(
        [
            [-9999, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, 20, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
        ]
    ).astype(np.float32)

@pytest.fixture(name="expected_filled_dem_labels_with_nodata")
def fixture_expected_filled_dem_labels_with_nodata():
    """Create a small 5x5 raster for testing"""
    return np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
        ]
    ).astype(int)


def test_fill_tile_top_left(
    dem_values: np.ndarray,
    expected_filled_dem_values: np.ndarray,
    expected_filled_dem_labels: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(top=True, left=True)
    labels, graph = priority_flood_tile(dem_values, sides)
    assert np.allclose(dem_values, expected_filled_dem_values)
    assert np.array_equal(labels, expected_filled_dem_labels)
    assert len(graph) == 2
    assert graph[(2, 3)] == 20.0
    assert graph[(1, 2)] == 15.0


def test_fill_tile_bottom_left(
    dem_values: np.ndarray,
    expected_filled_dem_values: np.ndarray,
    expected_filled_dem_labels: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(bottom=True, left=True)
    labels, graph = priority_flood_tile(dem_values, sides)
    assert np.allclose(dem_values, expected_filled_dem_values)
    assert np.array_equal(labels, expected_filled_dem_labels)
    assert len(graph) == 3
    assert graph[(2, 3)] == 20.0
    assert graph[(1, 2)] == 21.0
    assert graph[(1, 3)] == 15.0

def test_fill_tile_with_nodata_top(
    dem_values_with_nodata: np.ndarray,
    expected_filled_dem_values_with_nodata: np.ndarray,
    expected_filled_dem_labels_with_nodata: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(top=True)
    labels, graph = priority_flood_tile(dem_values_with_nodata, sides)
    assert np.allclose(dem_values_with_nodata, expected_filled_dem_values_with_nodata)
    assert np.array_equal(labels, expected_filled_dem_labels_with_nodata)
    assert len(graph) == 1
    assert graph[(1, 2)] == -np.inf

def test_fill_tile_with_nodata_right(
    dem_values_with_nodata: np.ndarray,
    expected_filled_dem_values_with_nodata: np.ndarray,
    expected_filled_dem_labels_with_nodata: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides(right=True)
    labels, graph = priority_flood_tile(dem_values_with_nodata, sides)
    assert np.allclose(dem_values_with_nodata, expected_filled_dem_values_with_nodata)
    assert np.array_equal(labels, expected_filled_dem_labels_with_nodata)
    assert len(graph) == 1
    assert graph[(1, 2)] == 50

def test_fill_tile_with_nodata_no_edge(
    dem_values_with_nodata: np.ndarray,
    expected_filled_dem_values_with_nodata: np.ndarray,
    expected_filled_dem_labels_with_nodata: np.ndarray,
):
    """Test filling single tile"""
    sides = make_sides()
    labels, graph = priority_flood_tile(dem_values_with_nodata, sides)
    assert np.allclose(dem_values_with_nodata, expected_filled_dem_values_with_nodata)
    assert np.array_equal(labels, expected_filled_dem_labels_with_nodata)
    assert len(graph) == 0

def test_handle_edge():
    """Test handle_edge function produces expected graph"""
    dem_a = np.array([1, 2, 3, 4, 5])
    labels_a = np.array([2, 2, 3, 3, 2])
    dem_b = np.array([5, 4, 3, 2, 1])
    labels_b = np.array([5, 5, 6, 6, 5])
    graph =  Dict.empty(
        key_type=types.Tuple([types.int64, types.int64]),
        value_type=types.float32,
    )
    graph[(2,5)] = 5
    no_data = -9999

    handle_edge(dem_a, labels_a, dem_b, labels_b, graph, no_data)

    expected_graph = {(2, 5): 4, (2, 6): 3, (3, 5): 4, (3, 6): 3}
    assert len(graph) == 4
    for key, value in expected_graph.items():
        assert graph[key] == value

def test_handle_corner():
    """Test handle_corner function produces expected graph"""
    elev_a = 5
    label_a = 2
    elev_b = 1
    label_b = 5
    graph = Dict.empty(
        key_type=types.Tuple([types.int64, types.int64]),
        value_type=types.float32,
    )
    graph[2,5] = 6
    expected_graph = {(2, 5): 5}
    handle_corner(elev_a, label_a, elev_b, label_b, graph)
    assert len(graph) == 1
    for key, value in expected_graph.items():
        assert graph[key] == value
