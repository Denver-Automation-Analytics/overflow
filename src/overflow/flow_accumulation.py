from enum import Enum
import numpy as np
from numba import njit
from osgeo import gdal
from numba.typed import Dict
from numba.core import types
from .util.raster import raster_chunker, RasterChunk, neighbor_generator
from .constants import (
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_NODATA,
    FLOW_ACCUMULATION_NODATA,
    FLOW_TERMINATES,
    FLOW_EXTERNAL,
    NEIGHBOR_OFFSETS,
    DEFAULT_CHUNK_SIZE,
)
from .util.numba_datastructures import ResizableFIFOQueue


class Side(Enum):
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    LEFT = 4


class Corner(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM_LEFT = 4


@njit
def get_next_cell(
    flow_direction: np.ndarray, row: int, col: int
) -> tuple[int, int, int]:
    """Return the next (downstream) row, column, and value in the flow direction raster.

    Args:
        flow_direction (np.ndarray): Flow direction raster
        row (int): The row of the current cell
        col (int): The col of the current cell

    Returns:
        tuple[int, int, int]: the (row, col, val) of the next (downstream) cell
    """
    fdr_value = flow_direction[row, col]
    d_row, d_col = NEIGHBOR_OFFSETS[fdr_value]
    next_row = row + d_row
    next_col = col + d_col
    rows, cols = flow_direction.shape
    is_outside_tile = (
        next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols
    )
    if not is_outside_tile:
        return next_row, next_col, flow_direction[next_row, next_col]
    return next_row, next_col, FLOW_DIRECTION_NODATA


@njit
def perimeter_indices(shape):
    """Return the indices of the perimeter cells of a 2D array."""
    rows, cols = shape
    indices = []
    for i in range(rows):
        indices.append((i, 0))
        indices.append((i, cols - 1))
    for j in range(1, cols - 1):
        indices.append((0, j))
        indices.append((rows - 1, j))
    return indices


@njit
def follow_path(flow_direction, row, col, links):
    """Follow the flow path from a cell on the perimeter of the tile and
    populate the links array with the row and col of the perimeter cell that each cell drains to.
    If the cell drains directly to the edge of the tile, the FLOW_EXTERNAL value is used.
    If the cell terminates within the tile, the FLOW_TERMINATES value is used.
    If the cell drains through the tile, the row and col of the perimeter cell it drains to are used.
    This is Algorithm 2 from https://arxiv.org/pdf/1608.04431.pdf R. Barnes

    Args:
        flow_direction (np.ndarray): Flow direction raster
        row (int): the row of the cell on the perimeter
        col (int): the col of the cell on the perimeter
        links (np.ndarray): 3D array of shape (rows, cols, 2)
    """
    init_row = row
    init_col = col
    rows, cols = flow_direction.shape
    while True:
        value = flow_direction[row, col]
        if value == FLOW_DIRECTION_NODATA:
            break
        next_row, next_col, next_val = get_next_cell(flow_direction, row, col)
        is_outside_tile = (
            next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols
        )
        if is_outside_tile:
            if row == init_row and col == init_col:
                # cell drains immediately to the edge of the tile
                links[init_row, init_col, 0] = FLOW_EXTERNAL[0]
                links[init_row, init_col, 1] = FLOW_EXTERNAL[1]
                links[init_row, init_col, 2] = flow_direction[row, col]

            else:
                # cell drains through the tile
                links[init_row, init_col, 0] = row
                links[init_row, init_col, 1] = col
                links[init_row, init_col, 2] = -255

            break
        if next_val == FLOW_DIRECTION_NODATA or next_val == FLOW_DIRECTION_UNDEFINED:
            # cell terminates within the tile
            links[init_row, init_col, 0] = FLOW_TERMINATES[0]
            links[init_row, init_col, 1] = FLOW_TERMINATES[1]
            links[init_row, init_col, 2] = -255

            break
        row, col = next_row, next_col


@njit
def single_tile_flow_accumulation(
    flow_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate flow accumulation for a single tile.
       This is Algorithm 1 from https://arxiv.org/pdf/1608.04431.pdf R. Barnes

    Args:
        flow_direction (np.ndarray): Flow direction raster with codes from FlowDirection constants

    Returns:
        tuple[np.ndarray, np.ndarray]: (flow_accumulation, links)
         - flow_accumulation: The flow accumulation raster
         - links: The link raster identifying where each perimeter cell drains to
    """
    # the output flow accumulation raster
    flow_accumulation = np.zeros(flow_direction.shape, dtype=np.int64)
    # a dependency raster (number of neighbors that flow into each cell)
    inflow_count = np.zeros(flow_direction.shape, dtype=np.uint8)

    # Calculate inflow count
    for row, col in np.ndindex(flow_direction.shape):
        value = flow_direction[row, col]

        if value == FLOW_DIRECTION_NODATA:
            flow_accumulation[row, col] = FLOW_ACCUMULATION_NODATA
            continue
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        inflow_count[next_row, next_col] += 1

    queue = []

    # Populate initial queue will cells that have no inflow
    for row, col in np.ndindex(inflow_count.shape):
        _, _, next_value = get_next_cell(flow_direction, row, col)
        if inflow_count[row, col] == 0:
            queue.append((row, col))

    # main loop

    while queue:
        row, col = queue.pop(0)
        if flow_accumulation[row, col] == FLOW_ACCUMULATION_NODATA:
            continue
        flow_accumulation[row, col] += 1
        next_row, next_col, next_value = get_next_cell(flow_direction, row, col)
        if next_value == FLOW_DIRECTION_NODATA:
            continue
        flow_accumulation[next_row, next_col] += flow_accumulation[row, col]

        inflow_count[next_row, next_col] -= 1

        if inflow_count[next_row, next_col] == 0:
            queue.append((next_row, next_col))

    # links is a 3d numpy array containing the row and col on the
    # perimeter that each cell utlimatly drains to.
    # Only cells on the perimeter of the tile are considered.
    # Cells that drain directly to the edge of the tile are
    # marked with FLOW_EXTERNAL. Cells that terminate within the tile
    # are marked with FLOW_TERMINATES. Cells that drain through the tile
    # are marked with the row and col of the perimeter cell they drain to.

    links = (
        np.ones(
            shape=(flow_direction.shape[0], flow_direction.shape[1], 3), dtype=np.int64
        )
        * -255
    )
    for row, col in perimeter_indices(flow_direction.shape):
        follow_path(flow_direction, row, col, links)
    return flow_accumulation, links


def construct_local_accumulation_graph(links, global_row, global_col, global_graph):
    links_accumulation = {}
    links_direction = {}
    links_position = {}

    rows, cols, _ = links.shape
    for i in range(rows):
        for j in range(cols):
            if links[i, j, 0] > 0:
                links_accumulation[(links[i, j, 0], links[i, j, 1])] = (
                    links_accumulation.get((links[i, j, 0], links[i, j, 1]), 0) + 1
                )
    for i in range(rows):
        for j in range(cols):
            if links[i, j, 0] == FLOW_EXTERNAL[0]:
                links_direction[(i, j)] = links[i, j, 2]
    not_in_count = [
        l for l in links_direction.keys() if l not in links_accumulation.keys()
    ]
    for key in not_in_count:
        del links_direction[key]
    for key in links_direction.keys():
        key_array = np.array(key)
        d_row, d_col = NEIGHBOR_OFFSETS[links_direction[key]]
        to_row = key_array[0] + global_row + d_row
        to_col = key_array[1] + global_col + d_col
        global_graph[(key[0] + global_row, key[1] + global_col)] = (to_row, to_col)
        if key_array[0] == 0:
            links_position[key] = Side.TOP
        elif key_array[0] == rows - 1:
            links_position[key] = Side.BOTTOM
        elif key_array[1] == 0:
            links_position[key] = Side.LEFT
        elif key_array[1] == cols - 1:
            links_position[key] = Side.RIGHT
        elif key_array[0] == 0 and key_array[1] == 0:
            links_position[key] = Corner.TOP_LEFT
        elif key_array[0] == 0 and key_array[1] == cols - 1:
            links_position[key] = Corner.TOP_RIGHT
        elif key_array[0] == rows - 1 and key_array[1] == cols - 1:
            links_position[key] = Corner.BOTTOM_RIGHT
        elif key_array[0] == rows - 1 and key_array[1] == 0:
            links_position[key] = Corner.BOTTOM_LEFT
    return links_accumulation, links_direction, links_position, global_graph


def generate_flow_accumulation(
    input_path: str, output_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE
):
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
    global_graph = {}
    for chunk in raster_chunker(band, chunk_size=chunk_size, chunk_buffer_size=2):
        global_row = chunk.row * chunk_size
        global_col = chunk.col * chunk_size
        accumulation_tile, links = single_tile_flow_accumulation(chunk.data)
        links_accumulation, links_direction, links_position, global_graph = (
            construct_local_accumulation_graph(
                links, global_row, global_col, global_graph
            )
        )

    return global_graph

    # chunk.from_numpy(accumulation_tile)
    # chunk.write(output_band)
