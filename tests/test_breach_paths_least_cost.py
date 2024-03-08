import numpy as np
import pytest
from osgeo import gdal
from overflow.breach_paths_least_cost import (
    breach_all_pits_in_chunk_least_cost,
    breach_paths_least_cost,
    EPSILON_GRADIENT,
)

# pylint does not understand pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture(name="dem_with_pit")
def fixture_dem_with_pit():
    """A numpy array representing a DEM with a pit at the center and a breach path to the edge."""
    return np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )


@pytest.fixture(name="dem_with_pit_and_nodata")
def fixture_dem_with_pit_and_nodata():
    """A numpy array representing a DEM with a pit at the center and a breach path to the edge."""
    return np.array(
        [
            [2, 2, 2, -9999, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )


@pytest.fixture(name="dem_from_file")
def fixture_dem_from_file(dem_with_pit):
    """Create a raster filepath for testing."""
    output_path = "/vsimem/test_raster_breach.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        output_path, dem_with_pit.shape[0], dem_with_pit.shape[1], 1, gdal.GDT_Float32
    )
    band = dataset.GetRasterBand(1)
    band.WriteArray(dem_with_pit)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


def test_breach_paths_least_cost_chunk(dem_with_pit):
    """Test that the expected breach path is created."""
    pits = np.array([[3, 3]])
    breach_all_pits_in_chunk_least_cost(pits, dem_with_pit, -9999)
    expected_dem = np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, -4 / 3, 2, 2, 2],
            [2, 2, 2, -2 / 3, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(dem_with_pit, expected_dem)


def test_breach_paths_least_cost_chunk_with_nodata(dem_with_pit_and_nodata):
    """Test that the expected breach path is created."""
    pits = np.array([[3, 3]])
    breach_all_pits_in_chunk_least_cost(pits, dem_with_pit_and_nodata, -9999)
    expected_dem = np.array(
        [
            [2, 2, 2, -9999, 2, 2, 2],
            [2, 2, 2, -2 * EPSILON_GRADIENT, 2, 2, 2],
            [2, 2, 2, -EPSILON_GRADIENT, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(dem_with_pit_and_nodata, expected_dem)


def test_dem_from_file(dem_from_file):
    output_path = "/vsimem/dem_from_file_breached.tif"
    breach_paths_least_cost(dem_from_file, output_path, chunk_size=7, search_radius=3)
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    expected_dem = np.array(
        [
            [2, 2, 2, -2, 2, 2, 2],
            [2, 2, 2, -4 / 3, 2, 2, 2],
            [2, 2, 2, -2 / 3, 2, 2, 2],
            [2, 2, 2, 0, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    assert np.allclose(array, expected_dem)


def test_dem_real():
    input_path = "data/dem1_5070.tif"
    output_path = "data/dem1_5070_breached.tif"
    breach_paths_least_cost(input_path, output_path, chunk_size=1000)
