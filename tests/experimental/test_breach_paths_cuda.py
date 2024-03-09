import numpy as np
import pytest
from osgeo import gdal
from overflow.experimental.breach_paths_least_cost_cuda import (
    breach_paths_least_cost_cuda,
)


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


def test_breach_paths_cuda(dem_from_file):
    """Test that the expected breach path is created."""
    output_path = "/vsimem/test_raster_breach_path.tif"
    breach_paths_least_cost_cuda(
        dem_from_file,
        output_path,
        chunk_size=7,
        search_radius=3,
        max_pits=1000,
    )
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
    dataset = gdal.Open(output_path)
    band = dataset.GetRasterBand(1)
    actual_dem = band.ReadAsArray()
    np.testing.assert_array_almost_equal(actual_dem, expected_dem)
    gdal.Unlink(output_path)
