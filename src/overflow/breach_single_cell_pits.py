import numpy as np
from numba import njit, prange

from overflow.setup_bands import setup_bands
from .util.raster import raster_chunker


@njit(parallel=True)
def breach_single_cell_pits_in_chunk(
    chunk, nodata_value
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function is used to breach single cell pits in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with breached single cell pits.

    Parameters
    ----------
    chunk : np.ndarray
        A chunk of a DEM.

    Returns
    -------
    np.ndarray
        A chunk of a DEM with breached single cell pits.
    """
    dx = [1, 1, 1, 0, -1, -1, -1, 0]
    dy = [-1, 0, 1, 1, 1, 0, -1, -1]
    dx2 = [2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2, -2, -1, 0, 1]
    dy2 = [-2, -1, 0, 1, 2, 2, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2]
    breachcell = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 0]
    # Create a copy of the chunk

    rows, cols = chunk.shape
    # Loop through each cell in the chunk
    unsolved_pits_raster = np.zeros_like(chunk, dtype=np.int8)
    # pylint: disable=not-an-iterable
    for row in prange(2, rows - 2):
        for col in range(2, cols - 2):
            z = chunk[row, col]
            if z != nodata_value:
                flag = True
                for k in range(8):
                    zn = chunk[row + dy[k], col + dx[k]]
                    if zn < z and zn != nodata_value:
                        flag = False
                        break

                if flag:
                    unsolved_pits_raster[row, col] = 1

    pit_indicies = np.argwhere(unsolved_pits_raster == 1)

    for row, col in pit_indicies:
        z = chunk[row, col]
        for k in range(16):
            zn = chunk[row + dy2[k], col + dx2[k]]
            if zn < z and zn != nodata_value:
                solved = True
                chunk[row + dy[breachcell[k]], col + dx[breachcell[k]]] = (z + zn) / 2
        if solved:
            unsolved_pits_raster[row, col] = 0

    return unsolved_pits_raster


def breach_single_cell_pits(input_path, output_path, chunk_size=2000):
    """
    input_raster = gdal.Open(input_path)
    projection = input_raster.GetProjection()
    transform = input_raster.GetGeoTransform()

    band = input_raster.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        output_path,
        input_raster.RasterXSize,
        input_raster.RasterYSize,
        1,
        gdal.GDT_Float32,
    )
    dataset.SetProjection(projection)
    dataset.SetGeoTransform(transform)
    output_band = dataset.GetRasterBand(1)
    output_band.SetNoDataValue(nodata_value)
    """

    with setup_bands(input_path, output_path) as bands:
        band, output_band, nodata_value = bands
        for chunk in raster_chunker(band, chunk_size=chunk_size, chunk_buffer_size=2):
            _ = breach_single_cell_pits_in_chunk(chunk.data, nodata_value)
            chunk.write(output_band)
