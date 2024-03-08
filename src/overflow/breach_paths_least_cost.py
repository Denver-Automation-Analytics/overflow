import math
import heapq
from numba import njit, prange
import numpy as np
from .util.raster import GridCell

# constants
DEFAULT_SEARCH_RADIUS = 100
DEFAULT_MAX_PITS = 100
UNVISITED_INDEX = -1


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
    next_cost = current_cost + multiplier * (next_elevation - init_elevation)
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
                dem[path_row, path_col] = init_elevation - (path_length - j) * 0.01
        else:
            dem[path_row, path_col] = (
                final_elevation + (init_elevation - final_elevation) * j / path_length
            )


@njit(parallel=True)
def breach_pits_least_cost(
    pits: np.ndarray,
    dem: np.ndarray,
    dem_no_data_value: float,
    costs_array: np.ndarray,
    prev_rows_array: np.ndarray,
    prev_cols_array: np.ndarray,
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
        - This function modifies the DEM in-place to create the least-cost paths.
        - Parallel execution is utilized for processing multiple pits simultaneously, enhancing performance,
          but it may encounter race conditions on occasion that may leave pits unsolved.
        - The search_radius must be a positive integer.
        - The DEM should have valid elevation values, and dem_no_data_value should be set accordingly for nodata cells.
    """
    if search_radius <= 0 or not isinstance(search_radius, int):
        raise ValueError("search_radius must be a positive integer")
    search_window_size = 2 * search_radius + 1
    # pylint: disable=not-an-iterable
    for i in prange(pits.shape[0]):
        # initialize variables for the search
        current_row = pits[i, 0]
        current_col = pits[i, 1]
        current_cost = 0
        init_elevation = dem[current_row, current_col]
        row_offset = search_radius - current_row
        col_offset = search_radius - current_col
        costs_array[i, current_row + row_offset, current_col + col_offset] = 0
        heap = [GridCell(current_row, current_col, current_cost)]
        heapq.heapify(heap)
        breach_point_found = False
        while len(heap) > 0:
            # dequeue the cell with the lowest cost
            cell = heapq.heappop(heap)
            current_cost, current_row, current_col = cell.cost, cell.row, cell.column
            # if this cell can be breached, stop
            if (
                dem[current_row, current_col] < init_elevation
                or dem[current_row, current_col] == dem_no_data_value
                or math.isnan(dem[current_row, current_col])
            ):
                breach_point_found = True
                break
            # if the heap size is too large, stop
            if len(heap) >= search_window_size**2:
                break  # pit is unsolvable with max heap size
            # for each neighbor of the current cell
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_row, next_col = current_row + dr, current_col + dc
                # Calculate the cost considering diagonal movement
                multiplier = 1 if dr == 0 or dc == 0 else math.sqrt(2)
                is_in_bounds = (
                    # check if the cell is inside the DEM
                    0 <= next_row < dem.shape[0]
                    and 0 <= next_col < dem.shape[1]
                    # check if the cell is inside the search window
                    and 0 <= next_row + row_offset < search_window_size
                    and 0 <= next_col + col_offset < search_window_size
                )
                if is_in_bounds:
                    process_neighbor(
                        i,
                        next_row,
                        next_col,
                        row_offset,
                        col_offset,
                        current_cost,
                        init_elevation,
                        dem,
                        dem_no_data_value,
                        costs_array,
                        prev_rows_array,
                        prev_cols_array,
                        heap,
                        multiplier,
                        current_row,
                        current_col,
                    )
        if breach_point_found:
            final_elevation = dem[current_row, current_col]
            final_elevation = (
                final_elevation if final_elevation != dem_no_data_value else -np.inf
            )
            reconstruct_path(
                i,
                current_row,
                current_col,
                final_elevation,
                init_elevation,
                dem,
                prev_rows_array,
                prev_cols_array,
                row_offset,
                col_offset,
            )
    # reset the costs and previous cells to their initial values
    costs_array.fill(np.inf)
    prev_rows_array.fill(UNVISITED_INDEX)
    prev_cols_array.fill(UNVISITED_INDEX)


def breach_all_pits_least_cost(
    pits: np.ndarray,
    dem: np.ndarray,
    dem_no_data_value: float,
    search_radius: int = DEFAULT_SEARCH_RADIUS,
    max_pits: int = DEFAULT_MAX_PITS,
):
    """
    Compute least-cost paths for all breach points within a given search radius. This function wraps
    breach_pits_least_cost and splits the pits into chunks to avoid memory overflow.
    """
    # allocate memory for the costs and previous cells
    search_window_size = 2 * search_radius + 1
    chunk_costs_array = np.full(
        (min(pits.shape[0], max_pits), search_window_size, search_window_size),
        np.inf,
        dtype=np.float32,
    )
    chunk_prev_rows_array = np.full(
        (min(pits.shape[0], max_pits), search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    chunk_prev_cols_array = np.full(
        (min(pits.shape[0], max_pits), search_window_size, search_window_size),
        UNVISITED_INDEX,
        dtype=np.int64,
    )
    # split the pits into chunks to avoid memory overflow
    for i in range(0, pits.shape[0], max_pits):
        chunk_pits = pits[i : i + max_pits]
        breach_pits_least_cost(
            chunk_pits,
            dem,
            dem_no_data_value,
            chunk_costs_array,
            chunk_prev_rows_array,
            chunk_prev_cols_array,
            search_radius,
        )
