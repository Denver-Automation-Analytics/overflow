import pytest
import numpy as np
from osgeo import gdal
from overflow.util.raster import raster_chunker
from overflow.generate_flow_direction_raster import flow_direction_from_chunks


@pytest.fixture
def raster_file_path():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 100x100 with random float32 data.
    """
    output_path = "/vsimem/test_raster_FDIR.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, 5, 5, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.array(
        [
            [100, 90, 90, 97, 90],
            [97, 92, 80, 85, 95],
            [94, 95, 96, 95, 94],
            [97, 92, 95, 94, 90],
            [95, 90, 85, 40, 92],
        ]
    )
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()
    dataset = None
    yield output_path
    gdal.Unlink(output_path)


def test_breach_single_cell_pits(raster_file_path):

    expected = np.array(
        [
            [1, 2, 4, 8, 8],
            [1, 1, 255, 16, 16],
            [128, 128, 64, 32, 32],
            [1, 2, 2, 4, 8],
            [1, 1, 1, 255, 16],
        ]
    )
    results_path = "/vsimem/test_flow_direction_results.tif"
    flow_direction_from_chunks(raster_file_path, results_path, chunk_size=5)
    result = gdal.Open(results_path)
    band = result.GetRasterBand(1)
    band = result.ReadAsArray(0, 0, result.RasterXSize, result.RasterYSize).astype(int)
    assert (band & expected).all()
