import math
import numpy as np
from numba import njit, prange
from osgeo import gdal
from .util.raster import raster_chunker


@njit(parallel=True)
def generate_flow_direction_raster(dem, nodata_value) -> np.ndarray:
    """
    Define the 8 directions using a list of tuples
     32   |   64   |  128
     ------------------
     16   |  255   |  1
     ------------------
     8    |   4   |  2

    This function is used to calculate flow direction in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with flow direction values.

    Parameters
    ----------
    chunk (np.ndarray) : Digital Elevation Model (DEM) chunk.

    Returns
    -------
    np.ndarray
        A chunk of a DEM with flow direction values.
    """
    # order is optimized for cache access
    d8_directions_array = [
        32,  # North West
        64,  # North
        128,  # North East
        16,  # West
        1,  # East
        8,  # South West
        4,  # South
        2,  # South East
    ]

    # locations defined as (row,col), position in array matches d8_directions_array
    directions = [
        (-1, -1),  # North West
        (-1, 0),  # North
        (-1, 1),  # North East
        (0, -1),  # West
        (0, 1),  # East
        (1, -1),  # South west
        (1, 0),  # South
        (1, 1),  # South East
    ]
    # np.empty is faster than np.full
    fdr = np.zeros(dem.shape, dtype=np.uint8)
    # Get the shape of the chunk
    rows, cols = dem.shape

    # Loop through each cell in the chunk

    # pylint: disable=not-an-iterable
    for row in prange(1, rows - 1):
        for col in range(1, cols - 1):
            if dem[row, col] != nodata_value:
                max_slope = -np.inf
                max_index = -1
                all_non_positive = True

                for i, (dy, dx) in enumerate(directions):
                    slope = calculate_slope(dem, row, col, dy, dx, nodata_value)
                    if slope > max_slope:
                        max_slope = slope
                        max_index = i
                    if slope > 0:
                        all_non_positive = False

                if all_non_positive:
                    fdr[row, col] = 255
                else:
                    fdr[row, col] = d8_directions_array[max_index]

    return fdr


@njit()
def calculate_slope(
    dem: np.ndarray, row: int, col: int, dy: int, dx: int, nodata_value: float
) -> float:
    """
    Calculate the slope between the cell and its neighbors.

    Parameters
    ----------
    dem (np.ndarray) : Digital Elevation Model (DEM).
    i, j (int) : Coordinates of the cell.
    dx, dy (int) : Direction to the neighbor.
    nodata_value (float) : Value representing no data.

    Returns
    -------
    float
        The slope between the cell and its neighbor. Positive slopes indicate downhill flow.
    """
    if dem[row + dy, col + dx] == nodata_value:
        return -np.inf

    return (dem[row, col] - dem[row + dy, col + dx]) / (
        math.sqrt(2) if dx != 0 and dy != 0 else 1
    )


def flow_direction(input_path, output_path, chunk_size=1000):
    """
    Generates a flow direction raster from a DEM chunks of a given size.
    """
    input_raster = gdal.Open(input_path)
    projection = input_raster.GetProjection()
    transform = input_raster.GetGeoTransform()

    band = input_raster.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        output_path,
        input_raster.RasterYSize,
        input_raster.RasterXSize,
        1,
        gdal.GDT_Byte,
    )

    dataset.SetProjection(projection)
    dataset.SetGeoTransform(transform)
    output_band = dataset.GetRasterBand(1)

    for chunk in raster_chunker(band, chunk_size=chunk_size, chunk_buffer_size=1):

        result = generate_flow_direction_raster(chunk.data, nodata_value)
        chunk.from_numpy(result)
        chunk.write(output_band)
