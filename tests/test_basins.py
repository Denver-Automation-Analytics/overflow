import pytest
import numpy as np
from osgeo import gdal
from numba.typed import Dict
from numba.types import UniTuple
from numba import int64
from overflow.basins import label_watersheds, label_watersheds_tiled
from overflow.constants import FLOW_DIRECTION_NODATA


@pytest.fixture(name="test_fdr")
def fixture_test_fdr():
    """Create a sample FDR for testing."""
    fdr = np.array(
        [
            [3, 4, 0, 1],
            [2, 3, 1, 2],
            [6, 5, 7, 6],
            [5, 4, 0, 7],
        ],
        dtype=np.ubyte,
    )
    return fdr


@pytest.fixture(name="expected_watersheds")
def fixture_expected_watersheds():
    """The expected watersheds from the test fdr."""
    watersheds = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ],
        dtype=np.int64,
    )
    return watersheds


@pytest.fixture(name="test_fdr_with_drainage_points")
def fixture_test_fdr_with_drainage_points():
    """Create a sample FDR for testing with drainage points."""
    fdr = np.array(
        [
            [7, 7, 6, 5, 5],
            [7, 7, 6, 5, 5],
            [7, 7, 6, 5, 5],
            [7, 7, 6, 5, 5],
            [0, 0, 6, 4, 4],
        ],
        dtype=np.ubyte,
    )
    return fdr


@pytest.fixture(name="test_fdr_with_drainage_points_filepath")
def fixture_test_fdr_with_drainage_points_filepath(test_fdr_with_drainage_points):
    """Create a filepath for test_fdr_with_drainage_points."""
    filepath = "/vsimem/test_fdr.tif"
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = test_fdr_with_drainage_points.shape
    dataset = driver.Create(filepath, cols, rows, 1, gdal.GDT_Byte)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(FLOW_DIRECTION_NODATA)
    band.WriteArray(test_fdr_with_drainage_points)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None
    yield filepath
    gdal.Unlink(filepath)


@pytest.fixture(name="test_drainage_points")
def fixture_test_drainage_points():
    """Create a sample drainage points."""
    drainage_points = Dict.empty(UniTuple(int64, 2), int64)
    drainage_points[(2, 2)] = 0
    drainage_points[(3, 2)] = 0
    return drainage_points


@pytest.fixture(name="expected_watersheds_with_drainage_points")
def fixture_expected_watersheds_with_drainage_points():
    """The expected watersheds from the test_fdr_with_drainage_points."""
    watersheds = np.array(
        [
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 2],
            [3, 2, 1, 2, 3],
            [3, 3, 2, 3, 3],
            [3, 3, 3, 3, 3],
        ],
        dtype=np.int64,
    )
    return watersheds


def test_label_watersheds(test_fdr, expected_watersheds):
    """Test the label_watersheds function."""
    drainage_points = Dict.empty(UniTuple(int64, 2), int64)
    watersheds, next_id = label_watersheds(test_fdr, drainage_points)
    assert np.array_equal(watersheds, expected_watersheds)
    assert next_id == 5


def test_label_watersheds_with_drainage_points(
    test_fdr_with_drainage_points,
    test_drainage_points,
    expected_watersheds_with_drainage_points,
):
    """Test the label_watersheds function with drainage points."""
    watersheds, next_id = label_watersheds(
        test_fdr_with_drainage_points, test_drainage_points
    )
    assert np.array_equal(watersheds, expected_watersheds_with_drainage_points)
    assert next_id == 4
    assert test_drainage_points[(2, 2)] == 1
    assert test_drainage_points[(3, 2)] == 2


def test_label_watersheds_tiled(
    test_fdr_with_drainage_points_filepath,
    test_drainage_points,
    expected_watersheds_with_drainage_points,
):
    """Test the label_watersheds_tiled function."""
    output_path = "/vsimem/watersheds.tif"
    label_watersheds_tiled(
        test_fdr_with_drainage_points_filepath,
        test_drainage_points,
        output_path,
        chunk_size=3,
    )
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    watersheds = band.ReadAsArray()
    assert np.array_equal(watersheds, expected_watersheds_with_drainage_points)
