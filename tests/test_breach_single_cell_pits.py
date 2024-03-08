import pytest
import numpy as np
from osgeo import gdal
from overflow.util.raster import raster_chunker
from overflow.breach_single_cell_pits import (
    breach_single_cell_pits_in_chunk,
    breach_single_cell_pits,
)


@pytest.fixture
def raster_file_path():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 100x100 with random float32 data.
    """
    output_path = "/vsimem/test_raster_breach.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, 5, 5, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.array(
        [
            [100, 101, 90, 97, 90],
            [103, 102, 80, 96, 95],
            [94, 95, 96, 95, 94],
            [97, 98, 95, 94, 90],
            [95, 90, 85, 40, 92],
        ]
    )
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


def test_breach_cingle_cell_pits_in_chunk():

    chunk = np.array(
        [
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, 100, 101, 90, 97, 90, -999, -999],
            [-999, -999, 103, 102, 80, 96, 95, -999, -999],
            [-999, -999, 94, 95, 96, 95, 94, -999, -999],
            [-999, -999, 97, 98, 95, 94, 90, -999, -999],
            [-999, -999, 95, 90, 85, 40, 92, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
        ]
    )
    expected = np.array(
        [
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, 100, 90, 90, 97, 90, -999, -999],
            [-999, -999, 97, 87, 80, 85, 95, -999, -999],
            [-999, -999, 94, 95, 96, 95, 94, -999, -999],
            [-999, -999, 97, 92, 95, 94, 90, -999, -999],
            [-999, -999, 95, 90, 85, 40, 92, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
            [-999, -999, -999, -999, -999, -999, -999, -999, -999],
        ]
    )
    nodata_value = -999
    _ = breach_single_cell_pits_in_chunk(chunk, nodata_value)
    assert (chunk & expected).all()


def test_breach_single_cell_pits(raster_file_path):

    expected = np.array(
        [
            [100, 90, 90, 97, 90],
            [97, 87, 80, 85, 95],
            [94, 95, 96, 95, 94],
            [97, 92, 95, 94, 90],
            [95, 90, 85, 40, 92],
        ]
    )
    results_path = "/vsimem/test_breach_single_cell_pits.tif"
    breach_single_cell_pits(raster_file_path, results_path, chunk_size=5)
    result = gdal.Open(results_path)
    band = result.GetRasterBand(1)
    band = result.ReadAsArray(0, 0, result.RasterXSize, result.RasterYSize).astype(int)
    assert (band & expected).all()
