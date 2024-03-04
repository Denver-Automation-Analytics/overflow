import numpy as np
import pytest
from osgeo import gdal
from overflow.util.raster import raster_chunker, read_raster_with_bounds_handling

# pylint does not understand pytest fixtures
# pylint: disable=redefined-outer-name
# allow protected access for testing
# pylint: disable=protected-access

@pytest.fixture
def raster_band():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 100x100 with random float32 data.
    """
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", 100, 100, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array = np.random.rand(100, 100).astype(np.float32)
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    yield band


def test_read_raster_as_array_in_bounds(raster_band: gdal.Band):
    """Test reading a raster in bounds"""
    array = read_raster_with_bounds_handling(10, 10, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array == raster_band.ReadAsArray(10, 10, 20, 20))


def test_read_raster_as_array_out_of_bounds(raster_band):
    """Test reading a raster out of bounds"""
    array = read_raster_with_bounds_handling(90, 90, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array[10:, 10:] == raster_band.GetNoDataValue())


def test_read_raster_as_array_negative_offset(raster_band):
    """Test reading a raster with a negative offset"""
    array = read_raster_with_bounds_handling(-10, -10, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array[:10, :10] == raster_band.GetNoDataValue())


def test_read_raster_as_array_offset_larger_than_raster(raster_band):
    """Test reading a raster with an offset larger than the raster"""
    array = read_raster_with_bounds_handling(100, 100, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array == raster_band.GetNoDataValue())
    array = read_raster_with_bounds_handling(200, 200, 20, 20, raster_band)
    assert array.shape == (20, 20)
    assert np.all(array == raster_band.GetNoDataValue())


def test_read_raster_as_array_zero_size(raster_band):
    """Test reading a raster with a zero size"""
    array = read_raster_with_bounds_handling(10, 10, 0, 20, raster_band)
    assert array.shape == (20, 0)
    array = read_raster_with_bounds_handling(10, 10, 20, 0, raster_band)
    assert array.shape == (0, 20)
    array = read_raster_with_bounds_handling(10, 10, 0, 0, raster_band)
    assert array.shape == (0, 0)


def test_read_raster_as_array_negative_size(raster_band):
    """Test reading a raster with a negative size"""
    with pytest.raises(AssertionError):
        read_raster_with_bounds_handling(10, 10, -20, 20, raster_band)
    with pytest.raises(AssertionError):
        read_raster_with_bounds_handling(10, 10, 20, -20, raster_band)
    with pytest.raises(AssertionError):
        read_raster_with_bounds_handling(10, 10, -20, -20, raster_band)


def test_raster_chunker_even(raster_band):
    """Test the raster chunker with a chunk size that evenly divides the raster size."""
    chunk_size = 10
    buffer_size = 2

    # create temp in memory raster band the same size as the original
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("test_raster_chunker_even", 100, 100, 1, gdal.GDT_Float32)
    temp_band = dataset.GetRasterBand(1)
    # fill with no data
    array = np.full((100, 100), raster_band.GetNoDataValue(), dtype=np.float32)
    temp_band.WriteArray(array)
    temp_band.FlushCache()

    for chunk in raster_chunker(raster_band, chunk_size, buffer_size):
        # Check the size of the chunk
        assert chunk.data.shape == (
            chunk_size + 2 * buffer_size,
            chunk_size + 2 * buffer_size,
        )
        # check the size of the unbuffered data
        assert chunk._get_unbuffered_data(raster_band).shape == (chunk_size, chunk_size)
        # for the first row of chunks, there should be nodata in the top buffer region
        if chunk.row == 0:
            assert np.all(chunk.data[:buffer_size, :] == raster_band.GetNoDataValue())
        # for the first column of chunks, there should be nodata in the left buffer region
        if chunk.col == 0:
            assert np.all(chunk.data[:, :buffer_size] == raster_band.GetNoDataValue())
        # for the last row of chunks, there should be nodata in the bottom buffer region
        if chunk.row == 9:
            assert np.all(chunk.data[-buffer_size:, :] == raster_band.GetNoDataValue())
        # for the last column of chunks, there should be nodata in the right buffer region
        if chunk.col == 9:
            assert np.all(chunk.data[:, -buffer_size:] == raster_band.GetNoDataValue())
        # for the inner chunks, the data should match the original raster
        if 0 < chunk.row < 9 and 0 < chunk.col < 9:
            assert np.all(
                chunk.data
                == raster_band.ReadAsArray(
                    chunk.col * chunk_size - buffer_size,
                    chunk.row * chunk_size - buffer_size,
                    chunk_size + 2 * buffer_size,
                    chunk_size + 2 * buffer_size,
                )
            )
        # write the chunk to the temp band
        chunk.write(temp_band)
    # flush the temp band to make sure the data is written
    temp_band.FlushCache()
    # read the temp band and check that it matches the original raster
    assert np.all(temp_band.ReadAsArray() == raster_band.ReadAsArray())


def test_raster_chunker_odd(raster_band):
    """Test the raster chunker with a chunk size that does not evenly divide the raster size."""
    chunk_size = 11
    buffer_size = 2

    # create temp in memory raster band the same size as the original
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("test_raster_chunker_odd", 100, 100, 1, gdal.GDT_Float32)
    temp_band = dataset.GetRasterBand(1)
    # fill with no data
    array = np.full((100, 100), raster_band.GetNoDataValue(), dtype=np.float32)
    temp_band.WriteArray(array)
    temp_band.FlushCache()

    for chunk in raster_chunker(raster_band, chunk_size, buffer_size):
        # Check the size of the chunk
        assert chunk.data.shape == (
            chunk_size + 2 * buffer_size,
            chunk_size + 2 * buffer_size,
        )
        if chunk.row < 9 and chunk.col < 9:
            # unbuffered size is always the chunk size in this case
            assert chunk._get_unbuffered_data(raster_band).shape == (
                chunk_size,
                chunk_size,
            )
        else:
            # unbuffered size is smaller than the chunk size in this case
            ub_shape = chunk._get_unbuffered_data(raster_band).shape
            if chunk.col == 9 and chunk.row < 9:
                assert ub_shape == (chunk_size, 1)
            elif chunk.row == 9 and chunk.col < 9:
                assert ub_shape == (1, chunk_size)
            else:
                assert ub_shape == (1, 1)
        # write the chunk to the temp band
        chunk.write(temp_band)
    # flush the temp band to make sure the data is written
    temp_band.FlushCache()
    # read the temp band and check that it matches the original raster
    assert np.all(temp_band.ReadAsArray() == raster_band.ReadAsArray())
