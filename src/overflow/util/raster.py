from typing import Iterator
from osgeo import gdal
import numpy as np

gdal.UseExceptions()


def gdal_data_type_to_numpy_data_type(gdal_dtype: int) -> np.dtype:
    """Map a GDAL data type to a numpy data type.

    Args:
        gdal_dtype (int): The GDAL data type to map.

    Returns:
        np.dtype: The numpy data type that corresponds to the input GDAL data type.
    """
    # map GDAL data type to numpy data type
    gdal_numpy_dtype_mapping = {
        "Byte": np.uint8,
        "UInt16": np.uint16,
        "Int16": np.int16,
        "UInt32": np.uint32,
        "Int32": np.int32,
        "Float32": np.float32,
        "Float64": np.float64,
        "CInt16": np.complex64,
        "CInt32": np.complex128,
        "CFloat32": np.complex64,
        "CFloat64": np.complex128,
    }
    return gdal_numpy_dtype_mapping[gdal.GetDataTypeName(gdal_dtype)]


def read_raster_with_bounds_handling(
    x_offset: int, y_offset: int, x_size: int, y_size: int, raster_band: gdal.Band
) -> np.ndarray:
    """Read a chunk of a raster band and return it as a numpy array. This function allows for reading
       out of bounds regions. If the window, or part of the window, extends beyond the edge of the raster,
       the out of bounds region will be filled with the nodata value for the band.

    Args:
        x_offset (int): The x offset of the chunk to read.
        y_offset (int): The y offset of the chunk to read.
        x_size (int): The number of columns in the chunk.
        y_size (int): The number of rows in the chunk.
        raster_band (gdal.Band): The raster band to read from.

    Returns:
        np.ndarray: The chunk of the raster band as a numpy array. The shape of the array will be: (y_size, x_size).
    """
    assert x_size >= 0, "x_size must be positive"
    assert y_size >= 0, "y_size must be positive"
    # Get the no data value from the raster band
    no_data_value = raster_band.GetNoDataValue()
    assert no_data_value is not None, "The raster band has no no data value"
    # Get the GDAL data type from the raster band
    gdal_dtype = raster_band.DataType

    # Convert the GDAL data type to a numpy data type
    np_dtype = gdal_data_type_to_numpy_data_type(gdal_dtype)

    # Create an array filled with the no data value
    window_data = np.full((y_size, x_size), no_data_value, dtype=np_dtype)

    # Calculate the offsets and sizes for reading the array
    # x_offset_adjusted and y_offset_adjusted are ensuring that the reading process doesn't start before the
    # beginning of the band. If x_offset or y_offset are negative, they are set to 0.
    x_offset_adjusted = max(x_offset, 0)
    y_offset_adjusted = max(y_offset, 0)

    # if x or y offsets are adjusted, the size of the window needs to be adjusted
    # by the same amount. This is to ensure that the returned window size remains the same
    # as the requested window size.
    x_size_adjusted = x_size - (x_offset_adjusted - x_offset)
    y_size_adjusted = y_size - (y_offset_adjusted - y_offset)

    # Calculate how much of the band remains after the offset.
    # If the offset is beyond the end of the band, this will be 0.
    x_remaining = max(raster_band.XSize - x_offset_adjusted, 0)
    y_remaining = max(raster_band.YSize - y_offset_adjusted, 0)

    # win_xsize and win_ysize are calculating the size of the window to read. They are ensuring that the
    # window doesn't extend beyond the end of the band by choosing the smaller of given y_size_adjusted
    # and y_size_adjusted and the size remaining.
    win_xsize = min(x_size_adjusted, x_remaining)
    win_ysize = min(y_size_adjusted, y_remaining)

    # Write band data to window data at the requested offsets
    window_data[
        y_offset_adjusted - y_offset : y_offset_adjusted - y_offset + win_ysize,
        x_offset_adjusted - x_offset : x_offset_adjusted - x_offset + win_xsize,
    ] = raster_band.ReadAsArray(
        xoff=x_offset_adjusted,
        yoff=y_offset_adjusted,
        win_xsize=win_xsize,
        win_ysize=win_ysize,
    )

    return window_data


class RasterChunk:
    """A class to represent a chunk of a raster band including an overlapping buffer region on all edges."""

    def __init__(
        self,
        row: int,
        col: int,
        size: int,
        buffer_size: int,
    ):
        self.data = None
        self.row = row
        self.col = col
        self.size = size
        self.buffer_size = buffer_size

    def from_numpy(self, data: np.ndarray):
        """Create a RasterChunk object from a numpy array.

        Args:
            data (np.ndarray): The numpy array to create the RasterChunk object from.
        """
        self.data = data

    def read(self, band: gdal.Band):
        """Read a chunk of a raster band including an overlapping buffer region on all edges.
           If part of the chunk, including the buffer region, extends beyond the edge of the raster,
           the out of bounds region will be filled with nodata. The chunk will be stored as a numpy array
           in the data attribute of the RasterChunk object.

        Args:
            band (gdal.Band): The raster band to read from.
        """
        x_offset = self.col * self.size - self.buffer_size
        y_offset = self.row * self.size - self.buffer_size
        x_size = self.size + 2 * self.buffer_size
        y_size = self.size + 2 * self.buffer_size
        self.data = read_raster_with_bounds_handling(
            x_offset, y_offset, x_size, y_size, band
        )

    def _get_unbuffered_data(self, band: gdal.Band) -> np.ndarray:
        y_remaining = max(band.YSize - self.row * self.size, 0)
        x_remaining = max(band.XSize - self.col * self.size, 0)
        unbuffered_y_size = min(self.size, y_remaining)
        unbuffered_x_size = min(self.size, x_remaining)
        return self.data[
            self.buffer_size : self.buffer_size + unbuffered_y_size,
            self.buffer_size : self.buffer_size + unbuffered_x_size,
        ]

    def write(self, band: gdal.Band):
        """Write a chunk to a raster band. The chunk must have been read from a band of the same size.
        The chunk will have only it's unbuffered data written to the band and will not write out of bounds regions.

        Args:
            band (gdal.Band): The raster band to write to.
        """
        if self.data is not None:
            band.WriteArray(
                self._get_unbuffered_data(band),
                xoff=self.col * self.size,
                yoff=self.row * self.size,
            )
        else:
            raise ValueError("The chunk has not been read yet.")


def raster_chunker(
    band: gdal.Band,
    chunk_size: int,
    chunk_buffer_size: int,
) -> Iterator[RasterChunk]:
    """Generator that yields chunks of a raster.

    Args:
        band  (gdal.Band): The raster band to read from.
        chunk_row_size (int): The number of rows in each chunk.
        chunk_col_size (int): The number of columns in each chunk.
        buffer_row_size (int): The number of rows in the buffer region.
        buffer_col_size (int): The number of columns in the buffer region.

    Yields:
        Iterator[Tuple[int, int, np.ndarray]]: An iterator that yields tuples. Each tuple contains
        the buffered chunk of the raster band as a numpy array, the x offset of the chunk, the y offset of the chunk,
        the number of columns in the chunk, and the number of rows in the chunk.
    """
    # Calculate the number of chunks in each dimension
    n_chunks_row = (band.YSize + chunk_size - 1) // chunk_size
    n_chunks_col = (band.XSize + chunk_size - 1) // chunk_size
    # Iterate over the chunks
    for chunk_row in range(n_chunks_row):
        for chunk_col in range(n_chunks_col):
            # Read the chunk and yield it
            chunk = RasterChunk(
                chunk_row,
                chunk_col,
                chunk_size,
                chunk_buffer_size,
            )
            chunk.read(band)
            yield chunk
