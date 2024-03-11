import numpy as np
from numba import njit, prange
from osgeo import gdal
from util.raster import raster_chunker


@njit(parallel=True)
def flow_accumulation(chunk, nodata_value) -> tuple[np.ndarray, np.ndarray]:
    direction_dict = {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
        255: (0, 0),
    }
    rows, cols = chunk.shape
    chunk_copy = np.zeros_like(chunk, dtype=np.int8)
    for row in prange(2, rows - 2):
        for col in range(2, cols - 2):
            try:
                path = direction_dict[int(chunk[row, col])]
            except:
                continue
            if chunk[row + path[0], col + path[1]] != nodata_value:
                chunk_copy[row + path[0], col + path[1]] = 1
            else:
                chunk_copy[row, col] = 1
    return chunk_copy


def flow_accumulation_from_chunks(input_path, output_path, chunk_size=2000):
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
        gdal.GDT_Float32,
    )

    dataset.SetProjection(projection)
    dataset.SetGeoTransform(transform)
    output_band = dataset.GetRasterBand(1)

    for chunk in raster_chunker(band, chunk_size=chunk_size, chunk_buffer_size=2):
        result = flow_accumulation(chunk.data, nodata_value)
        chunk.from_numpy(result)
        chunk.write(output_band)


flow_accumulation_from_chunks(
    "/workspaces/overflow/data/FDIR_Exp3.tif",
    "/workspaces/overflow/data/FDIR_Full3.tif",
)
