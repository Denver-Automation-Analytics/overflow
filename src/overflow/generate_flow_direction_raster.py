import math
import numpy as np
from numba import njit, prange

from .setup_bands import setup_bands
from .util.raster import raster_chunker


@njit(parallel=True)
def generate_flow_direction_raster(
    chunk, cell_size, nodata_value
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function is used to calculate flow direction in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with flow direction values.

    Parameters
    ----------
    chunk : np.ndarray
        A chunk of a DEM.

    Returns
    -------
    np.ndarray
        A chunk of a DEM with flow direction values.
    """
    d8_directions_dict = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128, 8: 255}
    dx = [1, 1, 1, 0, -1, -1, -1, 0]
    dy = [-1, 0, 1, 1, 1, 0, -1, -1]

    chunk_copy = chunk.copy()
    # Get the shape of the chunk
    rows, cols = chunk.shape
    # Loop through each cell in the chunk

    # pylint: disable=not-an-iterable
    for row in prange(2, rows - 2):
        for col in range(2, cols - 2):
            z = chunk[row, col]
            if z != nodata_value:
                slopes = []
                for k in range(8):
                    if chunk[row + dy[k], col + dx[k]] != nodata_value:
                        if k % 2 != 0 or k != 0:
                            slopes.append(
                                (z - chunk[row + dy[k], col + dx[k]]) / cell_size
                            )
                        else:
                            slopes.append(
                                (z - chunk[row + dy[k], col + dx[k]])
                                / math.sqrt(cell_size**2 + cell_size**2)
                            )
            if z != nodata_value:
                m = max(slopes)
                index_max = slopes.index(m)
                chunk_copy[row, col] = d8_directions_dict[index_max]

    return chunk_copy


def flow_direction_from_chunks(input_path, output_path, chunk_size=1000):
    """_summary_ - to do

    Args:
        input_path (_type_): _description_
        output_path (_type_): _description_
        chunk_size (int, optional): _description_. Defaults to 1000.
    """
    with setup_bands(input_path, output_path) as bands:
        input_band, output_band, nodata_value = bands
        cell_size = 1

        for chunk in raster_chunker(
            input_band, chunk_size=chunk_size, chunk_buffer_size=2
        ):
            result = generate_flow_direction_raster(chunk.data, cell_size, nodata_value)
            chunk.from_numpy(result)
            chunk.write(output_band)


flow_direction_from_chunks(
    "/workspaces/overflow/data/test7.tif",
    "/workspaces/overflow/data/testFDIR11.tif",
    chunk_size=1000,
)
