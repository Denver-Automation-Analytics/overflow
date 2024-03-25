import pytest
import numpy as np
from osgeo import gdal
from overflow.fix_flats import (
    flat_edges,
    label_flats,
    away_from_higher,
    towards_lower,
    resolve_flats,
    d8_masked_flow_dirs,
    fix_flats,
)
from overflow.constants import (
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_NODATA,
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


@pytest.fixture(name="expected_flat_labels")
def fixture_expected_flat_labels():
    """Create the expected labels for the flat area of the test dem.

    Returns:
        np.ndarray: A 2D array containing the expected labels for the flat area
    """
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        np.uint32,
    )


@pytest.fixture(name="expected_final_flat_mask")
def fixture_expected_final_flat_mask():
    """Create the expected final flat mask for the test dem.

    Returns:
        np.ndarray: A 2D array containing the expected final flat mask
    """
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 12, 12, 12, 12, 12, 0],
            [0, 10, 9, 9, 9, 10, 0],
            [0, 8, 7, 6, 7, 8, 0],
            [0, 6, 5, 5, 5, 8, 0],
            [0, 2, 2, 2, 6, 8, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        np.uint32,
    )


@pytest.fixture(name="dem_zhou_2022")
def fixture_dem_zhou_2022():
    """DEM from the worked example in Zhou et al. 2022.

    Returns:
        np.ndarray: A 2D array DEM
    """
    return np.array(
        [
            [5, 4, 4, 5, 9, 7, 2, 4],
            [9, 3, 3, 3, 3, 3, 3, 7],
            [7, 3, 3, 3, 3, 3, 3, 5],
            [8, 3, 3, 3, 3, 3, 3, 6],
            [9, 3, 3, 3, 3, 3, 3, 2],
            [6, 3, 3, 3, 3, 3, 3, 6],
            [5, 3, 3, 3, 3, 3, 3, 5],
            [1, 8, 9, 5, 6, 6, 7, 4],
        ],
        np.uint32,
    )


@pytest.fixture(name="dem_zhou_2022_filepath")
def fixture_dem_zhou_2022_filepath(dem_zhou_2022):
    """DEM from the worked example in Zhou et al. 2022.

    Yeilds:
        str: A path to a DEM file
    """
    dem_path = "/vsimem/dem_zhou_2022.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = dem_zhou_2022.shape
    dataset = driver.Create(dem_path, cols, rows, 1, gdal.GDT_Float32)
    # set nodata value
    dataset.GetRasterBand(1).SetNoDataValue(-9999)
    dataset.GetRasterBand(1).WriteArray(dem_zhou_2022)
    dataset = None
    yield dem_path
    gdal.Unlink(dem_path)


@pytest.fixture(name="fdr_zhou_2022")
def fixture_fdr_zhou_2022():
    """FDR from the worked example in Zhou et al. 2022.

    Returns:
        np.ndarray: A 2D array DEM
    """
    return np.array(
        [
            [
                FLOW_DIRECTION_NORTH_WEST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_NORTH_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_NORTH,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_NORTH_EAST,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_UNDEFINED,
                FLOW_DIRECTION_EAST,
            ],
            [
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_EAST,
            ],
        ],
        np.uint32,
    )


@pytest.fixture(name="fdr_zhou_2022_filepath")
def fixture_fdr_zhou_2022_filepath(fdr_zhou_2022):
    """FDR from the worked example in Zhou et al. 2022.

    Yeilds:
        str: A path to a FDR file
    """
    fdr_path = "/vsimem/fdr_zhou_2022.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = fdr_zhou_2022.shape
    dataset = driver.Create(fdr_path, cols, rows, 1, gdal.GDT_Byte)
    # set nodata value
    dataset.GetRasterBand(1).SetNoDataValue(FLOW_DIRECTION_NODATA)
    dataset.GetRasterBand(1).WriteArray(fdr_zhou_2022)
    dataset = None
    yield fdr_path
    gdal.Unlink(fdr_path)


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


def test_label_flats(dem, expected_flat_labels):
    """Test the label_flats function.

    Args:
        dem (np.ndarray): A dem with a flat that drains
        expected_flat_labels (np.ndarray): A 2D array containing the expected labels for the flat area
    """
    labels = np.zeros(dem.shape, dtype=np.uint32)
    flat_row, flat_col = 2, 2
    new_label = 1
    label_flats(dem, labels, new_label, flat_row, flat_col)
    assert np.array_equal(labels, expected_flat_labels)


def test_away_from_higher(fdr, expected_high_edges, expected_flat_labels):
    """Test the away_from_higher function.

    Args:
        fdr (np.ndarray): A flow direction raster
        expected_high_edges (list): A list of tuples containing the expected high edges
        expected_flat_labels (np.ndarray): A 2D array containing the expected labels for the flat area
    """
    flat_mask = np.zeros(fdr.shape, dtype=np.int32)
    flat_height = np.zeros((1), dtype=np.int32)
    high_edges = expected_high_edges
    labels = expected_flat_labels
    away_from_higher(labels, flat_mask, fdr, high_edges, flat_height)
    assert flat_height[0] == 3
    expected_flat_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 1, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 1, 2, 2, 2, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        np.int32,
    )
    assert np.array_equal(flat_mask, expected_flat_mask)


def test_towards_lower(
    fdr, expected_low_edges, expected_flat_labels, expected_final_flat_mask
):
    """Test the towards_lower function.

    Args:
        fdr (np.ndarray): A flow direction raster
        expected_low_edges (list): A list of tuples containing the expected low edges
        expected_flat_labels (np.ndarray): A 2D array containing the expected labels for the flat area
    """
    flat_mask = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 1, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 1, 2, 2, 2, 1, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        np.int32,
    )  # from the previous test we know the flat mask is this
    flat_height = np.zeros((1), dtype=np.int32)
    flat_height[0] = 3  # from the previous test we know the flat height is 3
    low_edges = expected_low_edges
    labels = expected_flat_labels
    towards_lower(labels, flat_mask, fdr, low_edges, flat_height)
    assert np.array_equal(flat_mask, expected_final_flat_mask)


def test_resolve_flats(dem, fdr, expected_final_flat_mask, expected_flat_labels):
    """Test the resolve_flats function.

    Args:
        dem (np.ndarray): A dem with a flat that drains
        fdr (np.ndarray): A flow direction raster
    """
    flat_mask, labels = resolve_flats(dem, fdr)
    assert np.array_equal(flat_mask, expected_final_flat_mask)
    assert np.array_equal(labels, expected_flat_labels)


def test_d8_masked_flow_dirs(fdr, expected_final_flat_mask, expected_flat_labels):
    """Test the d8_masked_flow_dirs function."""
    test_fdr = fdr.copy()
    d8_masked_flow_dirs(expected_final_flat_mask, test_fdr, expected_flat_labels)
    expected_fdr = np.array(
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
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_WEST,
            ],
            [
                FLOW_DIRECTION_EAST,
                FLOW_DIRECTION_SOUTH_EAST,
                FLOW_DIRECTION_SOUTH,
                FLOW_DIRECTION_SOUTH_WEST,
                FLOW_DIRECTION_WEST,
                FLOW_DIRECTION_NORTH_WEST,
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
    assert np.array_equal(test_fdr, expected_fdr)


def test_fix_flats(dem_zhou_2022_filepath, fdr_zhou_2022_filepath):
    fix_flats(dem_zhou_2022_filepath, fdr_zhou_2022_filepath, chunk_size=4)
