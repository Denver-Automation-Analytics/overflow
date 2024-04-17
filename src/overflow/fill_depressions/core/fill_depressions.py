from heapq import heappush, heappop
import numpy as np
from numba import njit, float32
from overflow.util.raster import (
    open_dataset,
    create_dataset,
    GridCellFloat32 as GridCell,
)
from overflow.constants import (
    EDGE_LABEL,
    NEIGHBOR_OFFSETS,
    TOP,
    RIGHT,
    BOTTOM,
    LEFT,
)
from overflow.fill_depressions.core.watershed_graph import WatershedGraph
from overflow.util.queue import GridCellFloat32Queue as Queue


@njit
def make_sides(top=False, right=False, bottom=False, left=False):
    # Initialize variable
    s = 0
    # Set flags based on input

    if top:
        s |= TOP
    if right:
        s |= RIGHT
    if bottom:
        s |= BOTTOM
    if left:
        s |= LEFT
    return s


@njit
def priority_flood_tile(
    dem: np.ndarray, sides: int = 0, no_data: float = -9999.0, label_offset: int = 2
) -> tuple[np.ndarray, WatershedGraph, int]:
    """
    Implementation of the Priority-Flood algorithm for a single tile.
    This is algorithm 1 from Parallel Priority-Flood (R. Barnes)
    https://arxiv.org/pdf/1606.06204.pdf

    Args:
        dem (numpy.ndarray): Input Digital Elevation Model (DEM) as a 2D array.
        sides (int): Flags indicating which sides of the DEM are open. See make_sides().
        no_data (float, optional): Value representing no data in the DEM. Defaults to -9999.

    Returns:
        Modifies dem in place
        tuple: A tuple containing:
            - labels (numpy.ndarray): Label for each cell in the DEM.
            - graph WatershedGraph: WatershedGraph associating label pairs with minimum spillover elevation.
            - label_count (int): Number of unique labels in the DEM.
    """
    # Initialize variables
    rows, cols = dem.shape
    labels = np.zeros_like(dem, dtype=np.int64)
    graph = WatershedGraph()
    label_count = label_offset

    # let numba infer type by initalizing with dummy value
    # and immediately popping it
    open_heap = [GridCell(0, 0, dem[0, 0])]
    open_heap.pop()
    pit_queue = Queue([GridCell(0, 0, dem[0, 0])])
    pit_queue.pop()
    minus_inf = float32(-np.inf)

    # Push edge cells onto the open heap
    # Nodata cells are treated as -inf
    for i in range(rows):
        for j in (0, cols - 1):
            if dem[i, j] != no_data:
                heappush(open_heap, GridCell(i, j, dem[i, j]))
            else:
                heappush(open_heap, GridCell(i, j, minus_inf))
    for j in range(1, cols - 1):
        for i in (0, rows - 1):
            if dem[i, j] != no_data:
                heappush(open_heap, GridCell(i, j, dem[i, j]))
            else:
                heappush(open_heap, GridCell(i, j, minus_inf))

    # process cells until both the open heap and pit queue are empty
    while open_heap or pit_queue:
        if pit_queue:
            cell = pit_queue.pop()
            i = cell.row
            j = cell.col
            c = cell.value
        else:
            cell = heappop(open_heap)
            c = cell.value
            i = cell.row
            j = cell.col

        # process the current cell
        if labels[i, j] == 0:
            labeled_neighbors = []
            for n in NEIGHBOR_OFFSETS:
                ni, nj = i + n[0], j + n[1]
                if 0 <= ni < rows and 0 <= nj < cols:
                    if labels[ni, nj] != 0:
                        labeled_neighbors.append((ni, nj))

            for ni, nj in labeled_neighbors:
                if dem[ni, nj] == no_data or dem[ni, nj] < c:
                    labels[i, j] = labels[ni, nj]
                    break
            else:
                labels[i, j] = label_count
                label_count += 1

        # process each neighbor of the current cell
        for n in NEIGHBOR_OFFSETS:
            ni, nj = i + n[0], j + n[1]
            if 0 <= ni < rows and 0 <= nj < cols:
                dem_n = dem[ni, nj] if dem[ni, nj] != no_data else minus_inf
                if labels[ni, nj] != 0:
                    if labels[i, j] == labels[ni, nj]:
                        continue
                    e = float32(max(c, dem_n))
                    label_pair = (
                        labels[i, j],
                        labels[ni, nj],
                    )
                    if label_pair not in graph or e < graph[label_pair]:
                        graph[label_pair] = e
                else:
                    labels[ni, nj] = labels[i, j]
                    if dem_n <= c:
                        dem[ni, nj] = c if c != minus_inf else no_data
                        pit_queue.push(GridCell(ni, nj, c))
                    else:
                        heappush(open_heap, GridCell(ni, nj, dem_n))

    # If this dem is an edge tile, add the edge labels to the graph
    if sides & TOP:
        for j in range(cols):
            label_pair = tuple((EDGE_LABEL, labels[0, j]))
            dem_c = dem[0, j] if dem[0, j] != no_data else minus_inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c
    if sides & RIGHT:
        for i in range(rows):
            label_pair = tuple((EDGE_LABEL, labels[i, cols - 1]))
            dem_c = dem[i, cols - 1] if dem[i, cols - 1] != no_data else minus_inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c
    if sides & BOTTOM:
        for j in range(cols):
            label_pair = tuple((EDGE_LABEL, labels[rows - 1, j]))
            dem_c = dem[rows - 1, j] if dem[rows - 1, j] != no_data else minus_inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c
    if sides & LEFT:
        for i in range(rows):
            label_pair = tuple((EDGE_LABEL, labels[i, 0]))
            dem_c = dem[i, 0] if dem[i, 0] != no_data else minus_inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c

    return labels, graph, label_count


def fill_depressions(dem_file: str, out_file: str) -> None:
    """
    Fill depressions in a DEM using the Priority-Flood algorithm.
    This is an all in RAM implementation.

    Args:
        dem_file (str): Path to the input DEM file.
        out_file (str): Path to the output filled DEM file.

    Returns:
        None
    """
    dem_ds = open_dataset(dem_file)
    dem_band = dem_ds.GetRasterBand(1)
    dem = dem_band.ReadAsArray()
    no_data = dem_band.GetNoDataValue()
    sides = make_sides()
    priority_flood_tile(dem, sides, no_data)
    out_ds = create_dataset(
        out_file,
        no_data,
        dem_band.DataType,
        dem_band.XSize,
        dem_band.YSize,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(dem)
    out_band.FlushCache()
