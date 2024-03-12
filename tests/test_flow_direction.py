import pytest
import numpy as np
from osgeo import gdal
from overflow.util.raster import raster_chunker
from overflow.generate_flow_direction_raster import flow_direction


@pytest.fixture
def raster_file_path():
    """Create a raster band for testing.

    Yields:
        gdal.Band: A raster band of size 5x5.
    """
    output_path = "/vsimem/test_raster_FDIR.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, 5, 5, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    # row 1 - all cells should flow to the right, final cell is undefined
    # row 2 - all cells should flow north east except final 2 cells drain north
    # all cells bordering 4 should drain into 4
    array = np.array(
        [
            [5, 4, 3, 2, 1],
            [5, 5, 5, 5, 5],
            [5, 5, 5, 5, 5],
            [5, 5, 4, 5, 5],
            [5, 5, 5, 5, 5],
        ]
    )
    band.WriteArray(array)
    band.SetNoDataValue(0)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


def test_flow_direction(raster_file_path):

    expected = np.array(
        [
            [1, 1, 1, 1, 255],
            [128, 128, 128, 64, 64],
            [255, 2, 4, 8, 255],
            [255, 1, 255, 16, 255],
            [255, 128, 64, 32, 255],
        ]
    )
    results_path = "/vsimem/test_flow_direction_results.tif"
    flow_direction(raster_file_path, results_path, chunk_size=5)
    result = gdal.Open(results_path)
    band = result.GetRasterBand(1)
    band = result.ReadAsArray(0, 0, result.RasterXSize, result.RasterYSize).astype(int)
    assert np.array_equal(band, expected)
