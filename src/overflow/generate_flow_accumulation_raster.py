import numpy as np
from numba import njit, prange
from osgeo import gdal
from util.raster import raster_chunker


@njit(parallel=True)
def flow_accumulation(
    fdr, ridge_indicies, flow_accumulation_raster, nodata_value
) -> tuple[np.ndarray, np.ndarray]:
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
    rows, cols = fdr.shape

    # pylint: disable=not-an-iterable
    for row in prange(1, rows - 1):
        for col in range(1, cols - 1):
            if (
                fdr[row, col] == nodata_value
            ):  # if cell is nodata, it will not be considered in flow accumulation
                ridge_indicies[row, col] = 1
            else:

                if fdr[row, col] in [
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                    255,
                ]:  # check for error cells in flow direction raster
                    path = direction_dict[
                        int(fdr[row, col])
                    ]  # get the direction of flow from the flow direction raster
                    ridge_indicies[row + path[0], col + path[1]] = (
                        1  # set the cell that receives flow to 1
                    )
                else:
                    ridge_indicies[row, col] = 1
    flow_accumulation_raster = increment_flow_accumulation(
        ridge_indicies, flow_accumulation_raster, fdr, nodata_value
    )
    return flow_accumulation_raster


@njit()
def increment_flow_accumulation(
    ridge_indicies, flow_accumulation_raster, fdr, nodata_value
):
    direction_dict = {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
        250: (0, 0),
    }

    ridge_cells = np.argwhere(
        ridge_indicies == 0
    )  # location of all cells that do not receive flow

    FAC_Row, FAC_Col = (
        flow_accumulation_raster.shape
    )  # get the shape of the flow accumulation raster, used to check for out of bounds
    for i, j in ridge_cells:
        acc_row, acc_col = i, j
        acc_increment = 1
        accum_dict = {}
        while (
            fdr[acc_row, acc_col] != nodata_value
        ):  # while the cell is not a nodata cell follow flow direction
            if int(fdr[acc_row, acc_col]) not in direction_dict:
                break
            else:
                acc_path = direction_dict[
                    int(fdr[acc_row, acc_col])
                ]  # get the direction of flow from the flow direction raster
                if (
                    acc_row + acc_path[0] < 0  # check for out of bounds
                    or acc_col + acc_path[1] < 0
                    or acc_row + acc_path[0] >= FAC_Row
                    or acc_col + acc_path[1] >= FAC_Col
                ):
                    break

                else:
                    if (
                        acc_row,
                        acc_col,
                    ) in accum_dict:  # if cell already present in dictionary, break loop
                        break
                    elif (
                        fdr[acc_row, acc_col] == 255
                    ):  # if cell is a sink or flat, keep going in same direction as previously established
                        accum_dict[(acc_row, acc_col)] = acc_increment
                        acc_increment += 1
                    else:
                        accum_dict[(acc_row, acc_col)] = (
                            acc_increment  # if cell not sink or flat, increment flow accumulation and add cell index to dictionary
                        )

                        acc_row = acc_row + acc_path[0]
                        acc_col = acc_col + acc_path[1]
                        acc_increment += 1

        if (
            len(accum_dict) > 200
        ):  # establish lower bound for stream accumulation length
            for key in accum_dict:
                flow_accumulation_raster[key[0], key[1]] = int(accum_dict[key])

    return flow_accumulation_raster


def flow_accumulation_from_chunks(input_path, output_path, chunk_size=6000):
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

    for chunk in raster_chunker(band, chunk_size=chunk_size, chunk_buffer_size=1):
        ridge_indicies = np.zeros(chunk.data.shape)
        flow_accumulation_raster = np.zeros(chunk.data.shape)
        result = flow_accumulation(
            chunk.data, ridge_indicies, flow_accumulation_raster, nodata_value
        )
        chunk.from_numpy(result)
        chunk.write(output_band)
        del ridge_indicies, flow_accumulation_raster


flow_accumulation_from_chunks(
    "/workspaces/overflow/data/FDR_NoData4.tif",
    "/workspaces/overflow/data/FAC21.tif",
)
