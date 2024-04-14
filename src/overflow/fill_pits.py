from heapq import heappush, heappop
import numpy as np
from numba import njit
from numba.experimental import jitclass
from osgeo import gdal

gdal.UseExceptions()

DEFAULT_CHUNK_SIZE = 2000

# Define global constant for edge label
EDGE_LABEL = np.int32(1)

# Define neighbor offsets considering all 8 neighbors
NEIGHBOR_OFFSETS = np.array(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    dtype=np.int64,
)

# Define flags for each side
TOP = 0b0001  # Binary representation of 1
RIGHT = 0b0010  # Binary representation of 2
BOTTOM = 0b0100  # Binary representation of 4
LEFT = 0b1000  # Binary representation of 8


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

@jitclass
class GridCell:
    row: int
    col: int
    value: float

    # Constructor
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value

    # add comparisons for values
    def __lt__(self, other):
        return self.value < other.value
    def __le__(self, other):
        return self.value <= other.value
    def __eq__(self, other):
        return self.value == other.value
    def __ne__(self, other):
        return self.value != other.value
    def __gt__(self, other):
        return self.value > other.value
    def __ge__(self, other):
        return self.value >= other.value


@njit
def priority_flood_tile(
    dem: np.ndarray, sides: int, no_data: float = -9999
) -> tuple[np.ndarray, dict[tuple, float]]:
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
            - graph (dict): Dictionary associating label pairs with minimum spillover elevation.
    """
    # Initialize variables
    rows, cols = dem.shape
    labels = np.zeros_like(dem, dtype=np.int32)
    graph = {}
    label_count = 2

    # let numba infer type by initalizing with dummy value
    # and immediately popping it
    open_heap = [GridCell(0, 0, dem[0, 0])]
    pit_queue = [(dem[0, 0], 0, 0)]
    open_heap.pop()
    pit_queue.pop()

    # Push edge cells onto the open heap
    # Nodata cells are treated as -inf
    for i in range(rows):
        for j in (0, cols - 1):
            if dem[i, j] != no_data:
                heappush(open_heap, GridCell(i, j, dem[i, j]))
            else:
                heappush(open_heap, GridCell(i, j, -np.inf))
    for j in range(1, cols - 1):
        for i in (0, rows - 1):
            if dem[i, j] != no_data:
                heappush(open_heap, GridCell(i, j, dem[i, j]))
            else:
                heappush(open_heap, GridCell(i, j, -np.inf))

    # process cells until both the open heap and pit queue are empty
    while open_heap or pit_queue:
        if pit_queue:
            c, i, j = pit_queue.pop(0)
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
                dem_n = dem[ni, nj] if dem[ni, nj] != no_data else -np.inf
                if labels[ni, nj] != 0:
                    if labels[i, j] == labels[ni, nj]:
                        continue
                    e = max(c, dem_n)
                    label_pair = tuple(
                        (
                            min(labels[i, j], labels[ni, nj]),
                            max(labels[i, j], labels[ni, nj])
                        )
                    )
                    if label_pair not in graph or e < graph[label_pair]:
                        graph[label_pair] = e
                else:
                    labels[ni, nj] = labels[i, j]
                    if dem_n <= c:
                        dem[ni, nj] = c
                        pit_queue.append((c, ni, nj))
                    else:
                        heappush(open_heap, GridCell(ni, nj, dem_n))

    # If this dem is an edge tile, add the edge labels to the graph
    if sides & TOP:
        for j in range(cols):
            label_pair = tuple((EDGE_LABEL, labels[0, j]))
            dem_c = dem[0, j] if dem[0, j] != no_data else -np.inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c
    if sides & RIGHT:
        for i in range(rows):
            label_pair = tuple((EDGE_LABEL, labels[i, cols - 1]))
            dem_c = dem[i, cols - 1] if dem[i, cols - 1] != no_data else -np.inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c
    if sides & BOTTOM:
        for j in range(cols):
            label_pair = tuple((EDGE_LABEL, labels[rows - 1, j]))
            dem_c = dem[rows - 1, j] if dem[rows - 1, j] != no_data else -np.inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c
    if sides & LEFT:
        for i in range(rows):
            label_pair = tuple((EDGE_LABEL, labels[i, 0]))
            dem_c = dem[i, 0] if dem[i, 0] != no_data else -np.inf
            if label_pair not in graph or dem_c < graph[label_pair]:
                graph[label_pair] = dem_c

    return labels, graph

@njit
def handle_edge(
    dem_a: np.ndarray,
    labels_a: np.ndarray,
    dem_b: np.ndarray,
    labels_b: np.ndarray,
    graph: dict[tuple[int, int], float],
    no_data: float = -9999
) -> None:
    """
    Combine two tiles by joining their edges.
    Algorithm 2 from Parallel Priority-Flood (R. Barnes)
    https://arxiv.org/pdf/1606.06204.pdf

    Args:
        dem_a (np.ndarray): Vector of cell elevations from tile A adjacent to tile B.
        labels_a (np.ndarray): Vector of cell labels from tile A adjacent to tile B.
        dem_b (np.ndarray): Vector of cell elevations from tile B adjacent to tile A.
        labels_b (np.ndarray): Vector of cell labels from tile B adjacent to tile A.
        graph (dict[tuple[int, int], float]): Master graph containing the
            partially-joined graphs of all tiles.

    Returns:
        None. The function modifies the graph in place.
    """
    # Iterate over all indices in the length of DEM_A
    for i, elev_a in enumerate(dem_a):
        elev_a = elev_a if elev_a != no_data else -np.inf
        # Iterate over all neighboring indices
        for ni in [i - 1, i, i + 1]:
            elev_b = dem_b[ni] if 0 <= ni < dem_b.shape[0] else -np.inf
            # Skip if the neighboring index is out of bounds
            if ni < 0 or ni == dem_a.shape[0]:
                continue
            # Skip if the labels at the current index and the neighboring index are the same
            if labels_a[i] == labels_b[ni]:
                continue
            # Calculate the maximum elevation between the current cell and the neighboring cell
            e = max(elev_a, elev_b)
            label_pair = tuple(
                (
                    min(labels_a[i], labels_b[ni]),
                    max(labels_a[i], labels_b[ni])
                )
            )
            # update the graph
            if label_pair not in graph or e < graph[label_pair]:
                graph[label_pair] = e

@njit
def handle_corner(
    elev_a: float,
    label_a: int,
    elev_b: float,
    label_b: int,
    graph: dict[tuple[int, int], float],
    no_data: float = -9999
) -> None:
    """
    Combine two tiles by joining their corners.
    Algorithm 2 analog from Parallel Priority-Flood (R. Barnes)
    https://arxiv.org/pdf/1606.06204.pdf

    Args:
        elev_a (float): Elevation from tile A corner adjacent to tile B corner.
        label_a (int): Label from tile A corner adjacent to tile B corner.
        elev_b (float): Elevation from tile B corner adjacent to tile A corner.
        label_b (int): Label from tile B corner adjacent to tile A corner.
        graph (dict[tuple[int, int], float]): Master graph containing the
            partially-joined graphs of all tiles.

    Returns:
        None. The function modifies the graph in place.
    """
    # Handle no_data values
    elev_a = elev_a if elev_a != no_data else -np.inf
    elev_b = elev_b if elev_b != no_data else -np.inf

    # Skip if the labels at the corner of tile A and tile B are the same
    if label_a == label_b:
        return

    # Calculate the maximum elevation between the corner cell of tile A and tile B
    e = max(elev_a, elev_b)
    label_pair = tuple(
        (
            min(label_a, label_b),
            max(label_a, label_b)
        )
    )

    # Update the graph
    if label_pair not in graph or e < graph[label_pair]:
        graph[label_pair] = e
