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

    ridge_cells = np.argwhere(
        ridge_indicies == 0
    )  # location of all cells that do not receive flow

    ridge_cells_dict = assign_index_dict(ridge_cells, fdr)
    flow_accumulation_raster = increment_flow_accumulation(
        ridge_cells_dict, flow_accumulation_raster, fdr, nodata_value
    )
    return flow_accumulation_raster


@njit()
def assign_index_dict(ridge_cells, fdr):
    ridge_cells_dict = {}
    for i in ridge_cells:
        ridge_cells_dict[(i[0], i[1])] = fdr[i[0], i[1]]
    return {(row, col): fdr[row, col] for row, col in ridge_cells}


@njit()
def increment_flow_accumulation(
    ridge_cells_dict, flow_accumulation_raster, fdr, nodata_value
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
        255: (-1, 0),
        179: (0, 0),
    }

    FAC_Row, FAC_Col = (
        flow_accumulation_raster.shape
    )  # get the shape of the flow accumulation raster, used to check for out of bounds

    for acc_row, acc_col in ridge_cells_dict:
        cyclical_array_check = []

        acc_increment = 1
        accum_dict = {}
        cyclical = False
        while (
            fdr[acc_row, acc_col] != nodata_value
        ):  # while the cell is not a nodata cell follow flow direction
            if fdr[acc_row, acc_col] not in direction_dict:
                break

            if cyclical:
                pass
            else:
                acc_path = direction_dict[
                    fdr[acc_row, acc_col]
                ]  # get the direction of flow from the flow direction raster
            if (
                acc_row + acc_path[0] < 0  # check for out of bounds
                or acc_col + acc_path[1] < 0
                or acc_row + acc_path[0] >= FAC_Row
                or acc_col + acc_path[1] >= FAC_Col
            ):
                break

            if (
                acc_row,
                acc_col,
            ) in accum_dict:  # if cell already present in dictionary proceed, if cells become part of cyclical progression apply if statement below
                accum_dict[(acc_row, acc_col)] = accum_dict[(acc_row, acc_col)] + 1
                cyclical_array_check.append((acc_row, acc_col))
                if (
                    cyclical_array_check.count((acc_row, acc_col)) > 6
                ):  # check if cell is part of a cyclical progression, if so move row col position by -2 to escape
                    accum_dict[(acc_row - 1, acc_col - 1)] = acc_increment + 1

                    acc_row = acc_row - 2
                    acc_col = acc_col - 2
                    acc_increment += 1
                    if fdr[acc_row, acc_col] != nodata_value:
                        acc_path = direction_dict[fdr[acc_row, acc_col]]
                        cyclical = True
                        break

                else:
                    acc_row = acc_row + acc_path[0]
                    acc_col = acc_col + acc_path[1]
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
            for (row, col), value in accum_dict.items():
                flow_accumulation_raster[row, col] = int(value)

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
    "/workspaces/overflow/data/fdr_large.tif",
    "/workspaces/overflow/data/FAC37.tif",
)
