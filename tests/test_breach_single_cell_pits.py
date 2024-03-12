import pytest
import numpy as np
from osgeo import gdal
from overflow.breach_single_cell_pits import (
    breach_single_cell_pits_in_chunk,
    breach_single_cell_pits,
)


@pytest.fixture(name="raster_file_path")
def fixture_raster_file_path():
    """Create a random raster band for testing.


    Yields:
        gdal.Band: A raster band of size 5x5 with random float32 data.
    """
    output_path = "/vsimem/test_raster_breach.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, 5, 5, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.array(
        [
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
            [10, 9, 4, 7, 6],
            [10, 9, 8, 7, 3],
        ]
    )
    band.WriteArray(array)
    band.SetNoDataValue(-np.inf)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


def test_breach_single_cell_pits(raster_file_path):

    expected = np.array(
        [
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
            [10, 9, 8, 7, 6],
            [10, 9, 4, 3, 6],
            [10, 9, 8, 7, 3],
        ]
    )
    results_path = "/vsimem/test_breach_single_cell_pits.tif"
    breach_single_cell_pits(raster_file_path, results_path, chunk_size=5)
    result = gdal.Open(results_path)
    band = result.GetRasterBand(1)
    band = result.ReadAsArray(0, 0, result.RasterXSize, result.RasterYSize).astype(int)
    assert np.array_equal(band, expected)
