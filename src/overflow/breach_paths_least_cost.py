import math
import heapq
from numba import njit, prange
from osgeo import gdal
import numpy as np
from .util.raster import GridCell, raster_chunker, DEFAULT_CHUNK_SIZE
from .breach_single_cell_pits import breach_single_cell_pits_in_chunk

# constants
DEFAULT_SEARCH_RADIUS = 200
DEFAULT_MAX_PITS = 24
UNVISITED_INDEX = -1
EPSILON_GRADIENT = 1e-5  # small value to apply to gradient of breaching to nodata cells


@njit
def process_neighbor(
    i: int,
    next_row: int,
    next_col: int,
    row_offset: int,
    col_offset: int,
    current_cost: float,
    init_elevation: float,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    heap: list[GridCell],
    multiplier: float,
    current_row: int,
    current_col: int,
):
    """
    Process the neighbor cell for least-cost path computation.

    This function calculates the cost of moving to a neighboring cell and updates the costs array and previous cell
    arrays if the calculated cost is lower than the existing cost for that cell. It also enqueues the neighbor cell into
    the priority queue if necessary.

    Parameters:
        i (int): Index of the pit being processed.
        next_row (int): Row index of the neighboring cell.
        next_col (int): Column index of the neighboring cell.
        row_offset (int): Offset of the row index of the neighboring cell in the search window.
        col_offset (int): Offset of the column index of the neighboring cell in the search window.
        current_cost (float): Current accumulated cost to reach the current cell.
        init_elevation (float): Initial elevation of the pit cell.
        dem (np.ndarray): Digital Elevation Model (DEM) array.
        dem_no_data_value (float): No data value in the DEM.
        costs_array (np.ndarray): Array to store the accumulated costs to reach each cell.
        prev_rows_array (np.ndarray): Array to store the previous row indices of the path.
        prev_cols_array (np.ndarray): Array to store the previous column indices of the path.
        heap (list[GridCell]): Priority queue storing cells to be processed.
        multiplier (float): Multiplier factor for calculating cost considering diagonal movement.
        current_row (int): Row index of the current cell.
        current_col (int): Column index of the current cell.

    Returns:
        None
    """
    next_elevation = dem[next_row, next_col]
    if next_elevation == dem_no_data_value or math.isnan(next_elevation):
        next_elevation = -np.inf  # nodata cells are treated as most negative elevation
    if next_elevation != -np.inf:
        next_cost = current_cost + multiplier * (next_elevation - init_elevation)
    else:
        next_cost = current_cost
    # if the cost is less than the current cost of the neighbor
    if next_cost < costs_array[i, next_row + row_offset, next_col + col_offset]:
        # update the cost and previous cell of the neighbor
        costs_array[i, next_row + row_offset, next_col + col_offset] = next_cost
        prev_rows_array[i, next_row + row_offset, next_col + col_offset] = current_row
        prev_cols_array[i, next_row + row_offset, next_col + col_offset] = current_col
        # enqueue the neighbor
        heapq.heappush(heap, GridCell(next_row, next_col, next_cost))


@njit
def reconstruct_path(
    i: int,
    row: int,
    col: int,
    final_elevation: float,
    init_elevation: float,
    dem: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
    row_offset: int,
    col_offset: int,
):
    """
    Reconstruct the least-cost path from the final breach point to the pit cell.

    This function reconstructs the least-cost path from the final breach point back to the pit cell using the
    information stored in the previous cell arrays. It applies gradient to the DEM to create the breach path.

    Parameters:
        i (int): Index of the pit being processed.
        row (int): Row index of the final breach point.
        col (int): Column index of the final breach point.
        final_elevation (float): Elevation of the final breach point.
        init_elevation (float): Initial elevation of the pit cell.
        dem (np.ndarray): Digital Elevation Model (DEM) array.
        prev_rows_array (np.ndarray): Array storing the previous row indices of the path.
        prev_cols_array (np.ndarray): Array storing the previous column indices of the path.
        row_offset (int): Offset of the row index of the final breach point in the search window.
        col_offset (int): Offset of the column index of the final breach point in the search window.

    Returns:
        None
    """
    path = []
    while UNVISITED_INDEX not in (row, col):
        path.append((row, col))
        row, col = (
            prev_rows_array[i, row + row_offset, col + col_offset],
            prev_cols_array[i, row + row_offset, col + col_offset],
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
                if not is_in_bounds:
                    continue
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
    search_window_size = 2 * search_radius + 1
    chunk_costs_array, chunk_prev_rows_array, chunk_prev_cols_array = (
        allocate_memory_for_costs_and_prev_cells(search_window_size, max_pits)
    )

    for chunk in raster_chunker(
        input_band, chunk_size=chunk_size, chunk_buffer_size=search_radius
    ):
        if chunk.col == 3 and chunk.row == 1:
            pass
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