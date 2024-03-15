import math
import heapq
from numba import njit, prange
from osgeo import gdal
import numpy as np
from .util.raster import GridCell, raster_chunker
from .breach_single_cell_pits import breach_single_cell_pits_in_chunk
from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_RADIUS,
    DEFAULT_MAX_PITS,
    UNVISITED_INDEX,
    EPSILON_GRADIENT,
)


@njit
def process_neighbor(
    pit_index: int,
    neighbor_row: int,
    neighbor_col: int,
    row_offset: int,
    col_offset: int,
    current_cell_cost: float,
    initial_elevation: float,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    heap: list[GridCell],
    cost_multiplier: float,
    current_row: int,
    current_col: int,
) -> None:
    """
    Process a neighboring cell in a Digital Elevation Model (DEM) grid for pathfinding.

    This function calculates the cost of moving to a neighboring cell and updates the
    cost and previous cell information if the calculated cost is lower than the current
    cost of the neighbor. It also enqueues the neighbor into a priority queue if the
    cost is updated.

    Parameters:
        - pit_index (int): Index of the pit being processed. Used to index the cost, prev_col, and prev_row arrays.
        - next_row (int): Row index of the neighboring cell.
        - next_col (int): Column index of the neighboring cell.
        - row_offset (int): Offset of the row index of the neighboring cell in the search window.
        - col_offset (int): Offset of the column index of the neighboring cell in the search window.
        - current_cell_cost (float): Current accumulated cost to reach the current cell.
        - initial_elevation (float): Initial elevation of the pit cell.
        - dem (np.ndarray): Digital Elevation Model (DEM) array.
        - dem_no_data_value (float): No data value in the DEM.
        - costs_array (np.ndarray): 3D array storing the accumulated costs of reaching
          each cell in the grid.
        - prev_rows_array (np.ndarray): 3D array storing the row indices of the previous
          cells for each cell in the grid.
        - prev_cols_array (np.ndarray): 3D array storing the column indices of the previous
          cells for each cell in the grid.
        - heap (list[GridCell]): Priority queue storing cells to be processed.
        - cost_multiplier (float): Multiplier factor for calculating cost considering diagonal movement.
        - current_row (int): Row index of the current cell.
        - current_col (int): Column index of the current cell.

    Returns:
        None
    """
    next_elevation = dem[neighbor_row, neighbor_col]
    if next_elevation == dem_no_data_value or math.isnan(next_elevation):
        next_elevation = -np.inf  # nodata cells are treated as most negative elevation
    if next_elevation != -np.inf:
        next_cost = current_cell_cost + cost_multiplier * (
            next_elevation - initial_elevation
        )
    else:
        next_cost = current_cell_cost
    # if the cost is less than the current cost of the neighbor
    if (
        next_cost
        < costs_array[pit_index, neighbor_row + row_offset, neighbor_col + col_offset]
    ):
        # update the cost and previous cell of the neighbor
        costs_array[pit_index, neighbor_row + row_offset, neighbor_col + col_offset] = (
            next_cost
        )
        prev_rows_array[
            pit_index, neighbor_row + row_offset, neighbor_col + col_offset
        ] = current_row
        prev_cols_array[
            pit_index, neighbor_row + row_offset, neighbor_col + col_offset
        ] = current_col
        # enqueue the neighbor
        heapq.heappush(heap, GridCell(neighbor_row, neighbor_col, next_cost))


@njit
def reconstruct_path(
    pit_index: int,
    breach_point_row: int,
    breach_point_col: int,
    final_elevation: float,
    init_elevation: float,
    dem: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    row_offset: int,
    col_offset: int,
) -> None:
    """
    Reconstruct the least-cost path from the final breach point to the pit cell.

    This function reconstructs the least-cost path from the final breach point back to the pit cell using the
    information stored in the previous cell arrays. It applies gradient to the DEM to create the breach path.

    Parameters:
        - pit_index (int): Index of the pit being processed. Used to index the prev_row and prev_col arrays.
        - breach_point_row (int): Row index of the final breach point.
        - breach_point_col (int): Column index of the final breach point.
        - final_elevation (float): Elevation of the final breach point.
        - init_elevation (float): Initial elevation of the pit cell.
        - dem (np.ndarray): Digital Elevation Model (DEM) array.
        - prev_rows_array (np.ndarray): 3D array storing the row indices of the previous
          cells for each cell in the grid.
        - prev_cols_array (np.ndarray): 3D array storing the column indices of the previous
          cells for each cell in the grid.
        - row_offset (int): Offset of the row index of the final breach point in the search window.
        - col_offset (int): Offset of the column index of the final breach point in the search window.

    Returns:
        None
    """
    path = []
    row, col = breach_point_row, breach_point_col
    while UNVISITED_INDEX not in (row, col):
        path.append((row, col))
        row, col = (
            prev_rows_array[pit_index, row + row_offset, col + col_offset],
            prev_cols_array[pit_index, row + row_offset, col + col_offset],
        )
    # remove last cell in path since we don't want to modify the pit cell
    path.pop()
    path_length = len(path)
    for j, (path_row, path_col) in enumerate(path):
        # apply gradient to the dem to create the breach path
        if final_elevation == -np.inf:
            # we're breaching to a nodata cell, so don't modify the first cell
            if j > 0:
                # we're breaching to a nodata cell, so assume a small gradient
                dem[path_row, path_col] = min(
                    (init_elevation - (path_length - j) * EPSILON_GRADIENT),
                    dem[path_row, path_col],
                )
        else:
            dem[path_row, path_col] = min(
                (
                    final_elevation
                    + (init_elevation - final_elevation) * j / path_length
                ),
                dem[path_row, path_col],
            )


@njit(parallel=True)
def breach_pits_in_chunk_least_cost(
    pits: np.ndarray,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    output_dem: np.ndarray,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
):
    """
    Compute least-cost paths for breach points in parallel within a given search radius.

    This function computes the least-cost paths for breach points in parallel within a specified search radius. It uses
    Dijkstra's algorithm to explore the neighborhood of each breach point until a breach point is found or the search
    window is exhausted. Once a breach point is found, it reconstructs the least-cost path from the final breach point
    back to the pit cell.

    Parameters:
        pits (np.ndarray): Array of pit coordinates (row, column) representing the starting points for the least-cost
                           path computation.
        dem (np.ndarray): Digital Elevation Model (DEM) array.
        dem_no_data_value (float): No data value in the DEM.
        search_radius (int, optional): Search radius around each pit point within which the least-cost path is computed.
                                       Defaults to DEFAULT_SEARCH_RADIUS.

    Returns:
        None

    Raises:
        ValueError: If the search_radius is not a positive integer.

    Notes:
        - This function modifies the output_dem in-place to create the least-cost paths.
        - Parallel execution is utilized for processing multiple pits simultaneously.
        - The search_radius must be a positive integer. Only pits that can be solved within the search radius will be
            breached.
        - The DEM should have valid elevation values, and dem_no_data_value should be set accordingly for nodata cells.
    """
    if search_radius <= 0 or not isinstance(search_radius, int):
        raise ValueError("search_radius must be a positive integer")
    search_window_size = 2 * search_radius + 1
    neighbors = [
        (0, 1),  # Right
        (1, 0),  # Down
        (0, -1),  # Left
        (-1, 0),  # Up
        (-1, -1),  # Upper Left
        (-1, 1),  # Upper Right
        (1, -1),  # Lower Left
        (1, 1),  # Lower Right
    ]
    # list of locals for each thread
    breach_point_found = np.zeros(pits.shape[0], dtype=np.bool_)
    current_row = np.full(pits.shape[0], -1, dtype=np.int64)
    current_col = np.full(pits.shape[0], -1, dtype=np.int64)
    init_elevation = np.full(pits.shape[0], math.nan, dtype=np.float32)
    row_offset = np.full(pits.shape[0], -1, dtype=np.int64)
    col_offset = np.full(pits.shape[0], -1, dtype=np.int64)
    # pylint: disable=not-an-iterable
    for i in prange(pits.shape[0]):
        # initialize variables for the search
        current_row[i] = pits[i, 0]
        current_col[i] = pits[i, 1]
        current_cost = 0
        init_elevation[i] = dem[current_row[i], current_col[i]]
        row_offset[i] = search_radius - current_row[i]
        col_offset[i] = search_radius - current_col[i]
        costs_array[
            i, current_row[i] + row_offset[i], current_col[i] + col_offset[i]
        ] = 0
        heap = [GridCell(current_row[i], current_col[i], current_cost)]
        heapq.heapify(heap)
        breach_point_found[i] = False
        while len(heap) > 0:
            # dequeue the cell with the lowest cost
            cell = heapq.heappop(heap)
            current_cost, current_row[i], current_col[i] = (
                cell.cost,
                cell.row,
                cell.column,
            )
            # if this cell can be breached, stop
            if (
                dem[current_row[i], current_col[i]] < init_elevation[i]
                or dem[current_row[i], current_col[i]] == dem_no_data_value
                or math.isnan(dem[current_row[i], current_col[i]])
            ):
                breach_point_found[i] = True
                break
            # if the heap size is too large, stop
            if len(heap) >= search_window_size**2:
                break  # pit is unsolvable with max heap size
            # for each neighbor of the current cell
            for dr, dc in neighbors:
                next_row, next_col = current_row[i] + dr, current_col[i] + dc
                # Calculate the cost considering diagonal movement
                multiplier = 1 if dr == 0 or dc == 0 else math.sqrt(2)
                is_in_bounds = (
                    # check if the cell is inside the DEM
                    0 <= next_row < dem.shape[0]
                    and 0 <= next_col < dem.shape[1]
                    # check if the cell is inside the search window
                    and 0 <= next_row + row_offset[i] < search_window_size
                    and 0 <= next_col + col_offset[i] < search_window_size
                )
                if is_in_bounds:
                    process_neighbor(
                        i,
                        next_row,
                        next_col,
                        row_offset[i],
                        col_offset[i],
                        current_cost,
                        init_elevation[i],
                        dem,
                        dem_no_data_value,
                        costs_array,
                        prev_rows_array,
                        prev_cols_array,
                        heap,
                        multiplier,
                        current_row[i],
                        current_col[i],
                    )
    for i in range(pits.shape[0]):
        if breach_point_found[i]:
            final_elevation = dem[current_row[i], current_col[i]]
            final_elevation = (
                final_elevation if final_elevation != dem_no_data_value else -np.inf
            )
            reconstruct_path(
                i,
                current_row[i],
                current_col[i],
                final_elevation,
                init_elevation[i],
                output_dem,
                prev_rows_array,
                prev_cols_array,
                row_offset[i],
                col_offset[i],
            )
    # reset the costs and previous cells to their initial values
    costs_array.fill(np.inf)
    prev_rows_array.fill(UNVISITED_INDEX)
    prev_cols_array.fill(UNVISITED_INDEX)


def breach_all_pits_in_chunk_least_cost(
    pits: np.ndarray,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    max_pits: int = DEFAULT_MAX_PITS,
):
    """
    Compute least-cost paths for all breach points within a given search radius. This function
    preallocates memory for the cost and backlink rasters and wraps breach_pits_least_cost
    to split the pits into chunks and avoid memory overflow.
    """
    # split the pits into chunks to avoid memory overflow
    output_dem = dem.copy()
    for i in range(0, pits.shape[0], max_pits):
        chunk_pits = pits[i : i + max_pits]
        breach_pits_in_chunk_least_cost(
            chunk_pits,
            dem,
            dem_no_data_value,
            costs_array,
            prev_rows_array,
            prev_cols_array,
            output_dem,
            search_radius,
        )
    return output_dem


def allocate_memory_for_costs_and_prev_cells(
    search_window_size: int,
    max_pits: int,
):
    """
    Allocate memory for the costs and previous cells arrays.
    """
    chunk_costs_array = np.full(
        (max_pits, search_window_size, search_window_size),
        np.inf,
        dtype=np.float32,
    )
    chunk_prev_rows_array = np.full(
        (max_pits, search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    chunk_prev_cols_array = np.full(
        (max_pits, search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    return chunk_costs_array, chunk_prev_rows_array, chunk_prev_cols_array


def breach_paths_least_cost(
    input_path,
    output_path,
    chunk_size=DEFAULT_CHUNK_SIZE,
    search_radius=DEFAULT_SEARCH_RADIUS,
    max_pits=DEFAULT_MAX_PITS,
):
    """Main function to breach paths in a DEM using least cost algorithm.
    This function will tile the input DEM into chunks and breach the paths in
    each chunk in parallel using the least cost algorithm.
    """
    input_ds = gdal.Open(input_path)
    projection = input_ds.GetProjection()
    geotransform = input_ds.GetGeoTransform()
    input_band = input_ds.GetRasterBand(1)
    input_nodata = input_band.GetNoDataValue()
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(
        output_path, input_ds.RasterXSize, input_ds.RasterYSize, 1, gdal.GDT_Float32
    )
    output_ds.SetProjection(projection)
    output_ds.SetGeoTransform(geotransform)
    output_band = output_ds.GetRasterBand(1)
    output_band.SetNoDataValue(input_nodata)

    # allocate memory for the costs and previous cells
    # two parameters drive the memory requirements here

    # search_radius: This parameter determines the size of the search window around each pit.
    # The search window is a square with side length 2 * search_radius + 1. Therefore, the memory
    # required for storing the cost, previous row, and previous column arrays is proportional to
    # max_pits * (2 * search_radius + 1) ** 2.

    # max_pits: This parameter determines the maximum number of pits that can be processed in parallel.
    # More pits will require more memory for storing the cost, previous row, and previous column
    # arrays, but will also allow more efficient use of the CPU. This should not
    # be set larger than the number of available processing cores.
    search_window_size = 2 * search_radius + 1
    chunk_costs_array, chunk_prev_rows_array, chunk_prev_cols_array = (
        allocate_memory_for_costs_and_prev_cells(search_window_size, max_pits)
    )

    ## chunk_size: This parameter determines the size of the chunks into which the input raster is
    ## divided for processing. Larger chunk sizes will require more memory but can potentially improve
    ## performance by processing more data in RAM at once
    ## The memory required for storing the chunk data is proportional to chunk_size ** 2.
    for chunk in raster_chunker(
        input_band, chunk_size=chunk_size, chunk_buffer_size=search_radius
    ):
        pits_raster = breach_single_cell_pits_in_chunk(chunk.data, input_nodata)
        pits_array = np.argwhere(pits_raster == 1)
        breached_dem = breach_all_pits_in_chunk_least_cost(
            pits_array,
            chunk.data,
            input_nodata,
            chunk_costs_array,
            chunk_prev_rows_array,
            chunk_prev_cols_array,
            search_radius,
            max_pits,
        )
        chunk.from_numpy(breached_dem)
        chunk.write(output_band)
    input_ds = None
    output_ds = None
