import math
import numpy as np
from numba import njit, prange
from osgeo import gdal
from util.raster import raster_chunker


@njit(parallel=True)
def generate_flow_direction_raster(chunk, nodata_value) -> np.ndarray:
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
    d8_directions_dict = {0: 128, 1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64}
    dx = [1, 1, 1, 0, -1, -1, -1, 0]
    dy = [-1, 0, 1, 1, 1, 0, -1, -1]
    chunk_copy = chunk.copy()
    # Get the shape of the chunk
    rows, cols = chunk.shape
    # Loop through each cell in the chunk

    # pylint: disable=not-an-iterable
    for row in prange(1, rows - 1):
        for col in range(1, cols - 1):

            slopes = []
            if chunk[row, col] != nodata_value:
                for k in range(8):
                    # Check if the neighbor is a nodata value, used to keep array index constant
                    if chunk[row + dy[k], col + dx[k]] == nodata_value:
                        slopes.append(-9999)
                    else:
                        if k in [1, 3, 5, 7]:
                            # Calculate the slope between the cell and its non-diagonal neighbors
                            slope_calc = (
                                chunk[row, col] - chunk[row + dy[k], col + dx[k]]
                            )
                            slopes.append(slope_calc)
                        else:
                            # Calculate the slope between the cell and its diagonal neighbors
                            slope_calc = (
                                chunk[row, col] - chunk[row + dy[k], col + dx[k]]
                            ) / (math.sqrt(2))
                            slopes.append(slope_calc)

                all_same = True
                for slope in slopes:
                    if slope != slopes[0]:
                        all_same = False
                        break
                if all_same:
                    chunk_copy[row, col] = 255
                else:
                    # Get the maximum slope
                    m = max(slopes)
                    # If the maximum slope is negative, set the flow direction to 255
                    if m < 0:
                        chunk_copy[row, col] = 255
                    # Otherwise, set the flow direction to the corresponding D8 direction
                    else:
                        index_max = slopes.index(m)
                        chunk_copy[row, col] = d8_directions_dict[index_max]

    return chunk_copy


def flow_direction(input_path, output_path, chunk_size=1000):
    """Generates a flow direction raster from a DEM chunks of a given size."""
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


flow_direction(
    "/workspaces/overflow/data/MergedLarger.tif",
    "/workspaces/overflow/data/FDR_LargerOld3.tif",
    chunk_size=2000,
)