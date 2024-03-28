from enum import Enum
import os
import shutil
import math
import heapq
import tempfile
import numpy as np
from numba import njit, int32
from osgeo import gdal
from numba.experimental import jitclass
from numba.typed import List
from .util.raster import raster_chunker, RasterChunk, neighbor_generator
from .constants import (
    NEIGHBOR_OFFSETS,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTIONS,
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
def flat_edges(dem: np.ndarray, fdr: np.ndarray) -> tuple[list, list]:
    """Algorithm 3 FlatEdges: This function locates flat cells which border on
    higher and lower terrain and places them into queues for further processing,
    as described in §2.2. Upon entry, (1) DEM contains the elevations of every cell
    or a value NoData for cells not part of the DEM. (2) Any cell without a local
    gradient is marked NoFlow in FlowDirs. At exit:
    (1) high_edges contains all the high edge cells (those flat cells adjacent to
    higher terrain) of the DEM, in no particular order.
    (2) low_edges contains all the low edge cells of the DEM, in no particular order.
    https://rbarnes.org/sci/2014_flats.pdf

    Args:
        dem (np.ndarray): The digital elevation model
        fdr (np.ndarray): The flow direction raster

    Returns:
        tuple[list, list]: A tuple containing the high and low edge cell queues
    """
    # FIFO queues for the high and low edge cells
    high_edges = []
    low_edges = []

    for row, col in np.ndindex(fdr.shape):
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, fdr.shape[0], fdr.shape[1]
        ):
            # continue if the neighbor is nodata
            fdr_neighbor = fdr[neighbor_row, neighbor_col]
            if fdr_neighbor == FLOW_DIRECTION_NODATA:
                continue
            fdr_current = fdr[row, col]
            if (
                fdr_current != FLOW_DIRECTION_UNDEFINED
                and fdr_neighbor == FLOW_DIRECTION_UNDEFINED
                and dem[row, col] == dem[neighbor_row, neighbor_col]
            ):
                # cell is a low edge cell since it has a defined flow direction and a neighbor does not
                # because the neighbor is a flat cell
                low_edges.append((row, col))
                break
            if (
                fdr_current == FLOW_DIRECTION_UNDEFINED
                and dem[row, col] < dem[neighbor_row, neighbor_col]
            ):
                # cell is a high edge cell since it has no defined flow direction and a neighbor is higher
                high_edges.append((row, col))
                break
    return high_edges, low_edges


@njit
def label_flats(
    dem: np.ndarray, labels: np.ndarray, new_label: int, flat_row: int, flat_col: int
) -> None:
    """Algorithm 4 LabelFlats: This flood-fill function gives all the cells of a flat a common label,
    as described by §2.2. https://rbarnes.org/sci/2014_flats.pdf
    Upon entry:
    (1) dem contains the elevations of every cell or a value NoData for cells not part of the DEM.
    (2) Labels has the same dimensions as DEM.
    (3) flat_row, flat_col belongs to the flat which is to be labeled.
    (4) new_label is a unique label which has not been previously applied to a flat.
    (5) labels has been initialized to zero prior to the first call to this function.
    (6) labels has values greater than or equal to 1 for each processed cell which is in a flat.
    Each flat's cells bear a label unique to that flat.
    At exit:
    (1) flat_row, flat_col and every cell reachable from flat_row, flat_col by passing over only
    cells of the same elevation as flat_row, flat_col (all the cells in the flat to which c belongs)
    is marked as new_label in Labels.
    (2) labels has been updated to reflect the new labels which have been applied.

    Args:
        dem (np.ndarray): The digital elevation model
        labels (np.ndarray): The labels array
        new_label (int): The new label to apply to the flat
        flat_row (int): The initial row of the flat cell
        flat_col (int): The initial column of the flat cell
    """
    # create FIFO queue for to_be_filled
    to_be_filled = ResizableFIFOQueue([(flat_row, flat_col)])
    elev = dem[flat_row, flat_col]
    while to_be_filled:
        row, col = to_be_filled.pop()
        not_in_bounds = row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]
        if not_in_bounds:
            continue
        if dem[row, col] != elev:
            continue
        if labels[row, col] != 0:
            continue
        labels[row, col] = new_label
        # push all 8 neighbors onto the queue
        for d_row, d_col in NEIGHBOR_OFFSETS:
            neighbor_row = row + d_row
            neighbor_col = col + d_col
            to_be_filled.push((neighbor_row, neighbor_col))


def away_from_higher(
    labels: np.ndarray,
    flat_mask: np.ndarray,
    fdr: np.ndarray,
    high_edges: list,
    flat_height: np.array,
) -> None:
    """Algorithm 5 AwayFromHigher: This procedure builds a gradient away from higher terrain,
    as described in §2.3 and Fig. 1.
    Upon entry:
    (1) Every cell in Labels is marked either 0, indicating that the cell is not part of a flat,
    or a number greater than zero which identifies the flat to which the cell belongs.
    (2) Any cell without a local gradient is marked NoFlow in FlowDirs.
    (3) Every cell in FlatMask is initialized to 0. (4) HighEdges contains, in no particular order,
    all the high edge cells of the DEM which are part of drainable flats.
    At exit:
    (1) flat_height has an entry for each label value of Labels indicating the maximal number of
    increments to be applied to the flat identified by that label.
    (2) flat_mask contains the number of increments to be applied to each cell to form a gradient
    away from higher terrain; cells not in a flat have a value of 0.

    Args:
        labels (np.ndarray): The labels of each flat, same shape as the DEM. 0 means not part of a flat.
        flat_mask (np.ndarray): The flat mask, same shape as the DEM. 0 means not part of a flat.
        fdr (np.ndarray): The flow direction raster
        high_edges (np.ndarray): The high edge cells of the DEM. In no particular order. FIFO queue.
        flat_height (np.array): The flat height array, size of the number of flats
    """
    if len(high_edges) == 0:
        return
    high_edges = ResizableFIFOQueue(high_edges)
    loops = 1
    marker = (-1, -1)
    high_edges.push(marker)
    while len(high_edges) > 1:
        row, col = high_edges.pop()
        if row == marker[0] and col == marker[1]:
            loops += 1
            high_edges.push(marker)
            continue
        if flat_mask[row, col] > 0:
            continue
        flat_mask[row, col] = loops
        flat_height[labels[row, col] - 1] = loops
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, fdr.shape[0], fdr.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                high_edges.push((neighbor_row, neighbor_col))


@njit
def towards_lower(
    labels: np.ndarray,
    flat_mask: np.ndarray,
    fdr: np.ndarray,
    low_edges: list,
    flat_height: np.array,
) -> None:
    """Algorithm 6 TowardsLower:  This procedure builds a gradient towards lower terrain and
    combines it with the gradient away from higher terrain, as described in §2.4 and Fig. 2.
    Upon entry:
    (1) Every cell in Labels is marked either 0, indicating that the cell is not part of a flat,
    or a number greater than zero which identifies the flat to which the cell belongs.
    (2) Any cell without a local gradient is marked NoFlow in FlowDirs.
    (3) Every cell in FlatMask has either a value of 0, indicating that the cell is not part of
    a flat, or a value greater than zero indicating the number of increments which must be
    added to it to form a gradient away from higher terrain.
    (4) FlatHeight has an entry for each label value of Labels indicating the maximal number
    of increments to be applied to the flat identified by that label in order to form the
    gradient away from higher terrain.
    (5) LowEdges contains, in no particular order, all the low edge cells of the DEM.
    At exit:
    (1) FlatMask contains the number of increments to be applied to each cell to form a
    superposition of the gradient away from higher terrain with the gradient towards lower
    terrain; cells not in a flat have a value of 0.

    Args:
        labels (np.ndarray): The labels of each flat, same shape as the DEM. 0 means not part of a flat.
        flat_mask (np.ndarray): The flat mask, same shape as the DEM. 0 means not part of a flat.
        fdr (np.ndarray): The flow direction raster
        low_edges (np.ndarray): The low edge cells of the DEM. In no particular order. FIFO queue.
        flat_height (np.array): The flat height array, size of the number of flats
    """
    # make all entries in flat_mask negative
    flat_mask *= -1
    loops = 1
    marker = (-1, -1)
    low_edges = ResizableFIFOQueue(low_edges)
    low_edges.push(marker)
    while len(low_edges) > 1:
        row, col = low_edges.pop()
        if row == marker[0] and col == marker[1]:
            loops += 1
            low_edges.push(marker)
            continue
        if flat_mask[row, col] > 0:
            continue
        if flat_mask[row, col] < 0:
            flat_mask[row, col] += flat_height[labels[row, col] - 1] + 2 * loops
        else:
            flat_mask[row, col] = 2 * loops
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, fdr.shape[0], fdr.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                low_edges.push((neighbor_row, neighbor_col))


@njit
def resolve_flats(
    dem: np.ndarray, flow_dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1 ResolveFlats: The main body of the algorithm, as described in §2.1.
    Upon entry:
    (1) DEM contains the elevations of every cell or a value NoData for cells not part of the DEM.
    (2) FlowDirs contains the flow direction of every cell; cells without a local gradient are
    marked NoFlow. Algorithm 2 provides an example of how this might be done.
    At exit:
    (1) FlatMask has a value greater than or equal to zero for each cell, indicating its number of
    increments. These can be used be used in conjunction with Labels to determine flow directions
    without altering the DEM, as exemplified by Algorithm 7, or to alter the DEM in subtle ways to
    direct flow, as exemplified by Algorithm 8.
    (2) Labels has values greater than or equal to 1 for each cell which is in a flat. Each flats'
    cells bear a label unique to that flat.

    Args:
        dem (np.ndarray): The digital elevation model
        flow_dirs (np.ndarray): The flow direction raster

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the FlatMask and Labels arrays
    """
    # Initialize FlatMask and Labels arrays
    flat_mask = np.zeros_like(dem, dtype=np.int32)
    labels = np.zeros_like(dem, dtype=np.int32)

    # Find flat edges
    high_edges, low_edges = flat_edges(dem, flow_dirs)

    # Check if there are undrainable flats
    if len(low_edges) == 0:
        if len(high_edges) != 0:
            print("There were undrainable flats")
        else:
            print("There were no flats")
        return flat_mask, labels

    label = 1
    # Label flats from low edges
    for row, col in low_edges:
        if labels[row, col] == 0:
            label_flats(dem, labels, label, row, col)
            label += 1

    # Remove unlabeled cells from high edges
    high_edge_count = len(high_edges)
    high_edges = [(row, col) for row, col in high_edges if labels[row, col] != 0]
    if high_edge_count != len(high_edges):
        print("Not all flats have outlets")
        return flat_mask, labels

    # Initialize FlatHeight array
    flat_height = np.zeros(label, dtype=np.int32)

    # Compute gradient away from higher terrain
    away_from_higher(labels, flat_mask, flow_dirs, high_edges, flat_height)

    # Compute gradient towards lower terrain
    towards_lower(labels, flat_mask, flow_dirs, low_edges, flat_height)

    return flat_mask, labels


@njit
def d8_masked_flow_dirs(
    flat_mask: np.ndarray, fdr: np.ndarray, labels: np.ndarray
) -> None:
    """
    Determine flow directions across flats using the provided flat_mask and labels arrays.
    This is algorithm 7 from the paper https://rbarnes.org/sci/2014_flats.pdf.

    This function iterates over each cell in the fdr array and determines the flow direction
    for cells within flats. If a cell doesn't have a flow direction (i.e., it's not NoData and not
    already assigned a direction), it searches its neighbors within the same flat and selects the
    neighbor with the minimum flat_mask value to determine the direction of flow. Finally, it assigns
    the selected direction to the current cell.

    Args:
        flat_mask (np.ndarray): Array containing the number of increments to be applied to each cell
            to form a gradient which will drain the flat it is a part of.
        fdr (np.ndarray): Array containing flow directions for each cell. Cells without a local
            gradient have a value of NoFlow, while all other cells have defined flow directions.
        labels (np.ndarray): Array indicating which flat each cell is a member of. Cells in a flat
            have a value greater than zero indicating the label of the flat, otherwise, they have a
            value of 0.

    Returns:
        None: The function modifies the fdr array in place.
    """
    for row, col in np.ndindex(fdr.shape):
        if fdr[row, col] == FLOW_DIRECTION_NODATA:
            continue
        if fdr[row, col] != FLOW_DIRECTION_UNDEFINED:
            continue
        nmin = FLOW_DIRECTION_UNDEFINED
        min_slope = np.inf
        for i, (d_row, d_col) in enumerate(NEIGHBOR_OFFSETS):
            neighbor_row = row + d_row
            neighbor_col = col + d_col
            if (
                neighbor_row < 0
                or neighbor_row >= fdr.shape[0]
                or neighbor_col < 0
                or neighbor_col >= fdr.shape[1]
            ):
                continue
            if labels[neighbor_row, neighbor_col] != labels[row, col]:
                continue
            # calculate slope
            dz = float(flat_mask[neighbor_row, neighbor_col]) - flat_mask[row, col]
            slope = dz / (math.sqrt(2) if d_row != 0 and d_col != 0 else 1)
            # update minimum slope
            if slope < min_slope:
                min_slope = slope
                nmin = FLOW_DIRECTIONS[i]
        fdr[row, col] = nmin


@njit
def compute_gradient(
    labels: np.ndarray,
    fdr: np.ndarray,
    edges_cells: list,
) -> np.ndarray:
    """
    Compute the gradient from the given edge cells across the flats.

    This function uses a flood-fill approach to compute the gradient from the given edge cells
    across the flats. It assigns increasing values to cells as it moves away from the edge cells,
    creating a gradient that can be used to determine flow directions within the flats.

    Args:
        labels (np.ndarray): Array indicating which flat each cell is a member of. Cells in a flat
            have a value greater than zero indicating the label of the flat, otherwise, they have a
            value of 0.
        fdr (np.ndarray): Array containing flow directions for each cell. Cells without a local
            gradient have a value of FLOW_DIRECTION_UNDEFINED, while all other cells have defined
            flow directions or FLOW_DIRECTION_NODATA.
        edges_cells (list): List of edge cells from which to compute the gradient. These can be
            either high edge cells or low edge cells.

    Returns:
        np.ndarray: Array containing the computed gradient values. Cells not part of a flat have a
            value of 0.
    """
    graident = np.zeros_like(labels, dtype=np.int32)
    if len(edges_cells) == 0:
        return graident
    # Create a FIFO queue to store cells for processing
    edges_cells = ResizableFIFOQueue(edges_cells)
    loops = 1
    marker = (-1, -1)
    edges_cells.push(marker)
    while len(edges_cells) > 1:
        row, col = edges_cells.pop()
        # Check if the marker is encountered, indicating the end of a loop
        if row == marker[0] and col == marker[1]:
            loops += 1
            edges_cells.push(marker)
            continue
        # Skip cells that have already been assigned a gradient value
        if graident[row, col] > 0:
            continue
        # Assign the current loop value as the gradient value for the cell
        graident[row, col] = loops
        # Add neighboring cells within the same flat to the queue for processing
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, fdr.shape[0], fdr.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                edges_cells.push((neighbor_row, neighbor_col))
    return graident


@njit
def resolve_flats_tile(
    dem: np.ndarray, flow_dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Resolve flats in a tile-based processing pipeline.

    This function is a modified version of the resolve_flats algorithm for use in a tile-based
    processing pipeline. It finds flat edges, labels the flats, and computes gradients away from
    higher and lower terrain within the flats. The computed gradients can be used to determine
    flow directions across the flats.

    Args:
        dem (np.ndarray): Array containing the elevation values for each cell in the tile.
        flow_dirs (np.ndarray): Array containing flow directions for each cell in the tile.
            Cells without a local gradient have a value of FLOW_DIRECTION_UNDEFINED,
            while all other cells have defined flow directions or FLOW_DIRECTION_NODATA.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
            - dist_to_higher: Array containing the gradient values away from higher terrain.
            - dist_to_lower: Array containing the gradient values towards lower terrain.
            - labels: Array indicating which flat each cell is a member of.
            - high_edges: List of high edge cells.
            - low_edges: List of low edge cells.
    """

    # Find flat edges, this will not result in any uncertain edges
    # because we are processing with a 1 cell buffer
    high_edges, low_edges = flat_edges(dem, flow_dirs)

    # if the tile is entirely flat, set labels to 1 and return zero gradients
    if (
        len(low_edges) == 0
        and len(high_edges) == 0
        and flow_dirs[0, 0] == FLOW_DIRECTION_UNDEFINED
    ):
        labels = np.ones_like(dem, dtype=np.int32)
        dist_to_higher = np.zeros_like(dem, dtype=np.int32)
        dist_to_lower = np.zeros_like(dem, dtype=np.int32)
        return dist_to_higher, dist_to_lower, labels, high_edges, low_edges

    # Initialize labels array
    labels = np.zeros_like(dem, dtype=np.int32)

    label = 1
    # Label flats from low edges and high edges
    for row, col in low_edges + high_edges:
        if labels[row, col] == 0:
            label_flats(dem, labels, label, row, col)
            label += 1

    # Compute gradient away from higher terrain
    dist_to_higher = compute_gradient(labels, flow_dirs, high_edges)
    # Compute gradient away from lower terrain
    dist_to_lower = compute_gradient(labels, flow_dirs, low_edges)

    return dist_to_higher, dist_to_lower, labels, high_edges, low_edges


@jitclass(
    [
        ("rows", int32),
        ("cols", int32),
        ("index_offset", int32),
        ("perimeter", int32[:]),
    ]
)
class FlatTileEdgeCells:
    """
    A class representing the cells on the perimeter of a 2D array (tile).

    The perimeter cells are stored in a flattened 1D array in a clockwise order
    starting from the top-left corner.

    Attributes:
        rows (int): The number of rows in the original 2D array.
        cols (int): The number of columns in the original 2D array.
        perimeter (np.ndarray): A 1D contiguous array containing the flattened perimeter cells.
    """

    def __init__(self, array: np.ndarray, tile_index: int):
        self.rows = len(array)
        self.cols = len(array[0])
        self.perimeter = np.empty(2 * (self.rows + self.cols) - 4, dtype=array.dtype)

        # Fill the perimeter array
        self.perimeter[: self.cols] = array[0]  # top
        self.perimeter[self.cols : self.cols + self.rows - 2] = array[1:-1, -1]  # right
        self.perimeter[self.cols + self.rows - 2 : 2 * self.cols + self.rows - 2] = (
            array[-1, ::-1]
        )  # bottom
        self.perimeter[2 * self.cols + self.rows - 2 :] = array[-2:0:-1, 0]  # left
        self.index_offset = tile_index * self.perimeter.size

    def get_index(self, row, col):
        """
        Returns the flattened index of the cell at the given (row, col) coordinates.
        """
        if row == 0:
            return col
        elif row == self.rows - 1:
            return self.cols + self.rows - 2 + self.cols - col - 1
        elif col == self.cols - 1:
            return self.cols + row - 1
        else:
            return 2 * self.cols + self.rows - 2 + self.rows - row - 2

    def get_index_corner(self, corner: Corner):
        """
        Returns the flattened index of the cell at the specified corner.
        """
        if corner == Corner.TOP_LEFT:
            return self.get_index(0, 0)
        elif corner == Corner.TOP_RIGHT:
            return self.get_index(0, self.cols - 1)
        elif corner == Corner.BOTTOM_RIGHT:
            return self.get_index(self.rows - 1, self.cols - 1)
        elif corner == Corner.BOTTOM_LEFT:
            return self.get_index(self.rows - 1, 0)
        else:
            raise ValueError("Invalid corner")

    def get_index_side(self, side: Side, index: int):
        """
        Returns the flattened index of the cell at the specified side and index.
        """
        if side == Side.TOP:
            return self.get_index(0, index)
        elif side == Side.RIGHT:
            return self.get_index(index, self.cols - 1)
        elif side == Side.BOTTOM:
            return self.get_index(self.rows - 1, index)
        elif side == Side.LEFT:
            return self.get_index(index, 0)
        else:
            raise ValueError("Invalid side")

    def get_row_col(self, index):
        """
        Returns the (row, col) coordinates of the cell at the given flattened index.
        """
        if index < self.cols:
            return 0, index
        elif index < self.cols + self.rows - 1:
            return index - self.cols + 1, self.cols - 1
        elif index < 2 * self.cols + self.rows - 2:
            return self.rows - 1, 2 * self.cols + self.rows - index - 3
        else:
            return 2 * self.cols + 2 * self.rows - index - 4, 0

    def size(self):
        """
        Returns the total number of cells in the perimeter.
        """
        return self.perimeter.size

    def get_side(self, side: Side):
        """
        Returns the cells on the specified side of the perimeter.

        Args:
            side (Side): The side of the perimeter.

        Returns:
            np.ndarray: A contiguous array containing the cells on the specified side.

        Raises:
            ValueError: If an invalid side is specified.
        """
        if side == Side.TOP:
            # The enitire top edge (from left to right) includes the corners
            return self.perimeter[: self.cols]
        elif side == Side.RIGHT:
            # The enitire right edge (from top to bottom) includes the corners
            return self.perimeter[self.cols - 1 : self.cols + self.rows - 1]
        elif side == Side.BOTTOM:
            # The enitire bottom edge (from left to right) includes the corners
            return self.perimeter[
                self.cols + self.rows - 2 : 2 * self.cols + self.rows - 2
            ][
                ::-1
            ]  # reversed from internal representation
        elif side == Side.LEFT:
            # The entire left edge (from bottom to top) includes the corners
            indices = np.empty(self.rows, dtype=np.int32)
            indices[0] = 0
            indices[1:] = np.arange(
                self.perimeter.size - 1, self.perimeter.size - self.rows, -1
            )
            return self.perimeter[indices]
        else:
            raise ValueError("Invalid side")

    def get_corner(self, corner: Corner):
        """
        Returns the cell at the specified corner of the perimeter.

        Args:
            corner (Corner): The corner of the perimeter.

        Returns:
            int: The cell value at the specified corner.
        """
        return self.perimeter[self.get_index_corner(corner)]

    def distance(
        self,
        index: int,
        other_index: int,
    ) -> int:
        """
        Returns the Chebyshev distance between two cells specified by their flattened indices.

        The Chebyshev distance is the maximum of the absolute differences between the row and column coordinates.

        Args:
            index (int): The flattened index of the first cell.
            other_index (int): The flattened index of the second cell.

        Returns:
            int: The Chebyshev distance between the two cells.
        """
        from_row, from_col = self.get_row_col(index)
        to_row, to_col = self.get_row_col(other_index)
        return max(abs(from_row - to_row), abs(from_col - to_col))


@njit
def construct_local_edge_graph(
    labels: FlatTileEdgeCells,
    to_higher: FlatTileEdgeCells,
    to_lower: FlatTileEdgeCells,
) -> tuple[list, list, list]:
    # create a list of lists to represent the graph
    # there are at most labels.size() vertices in the graph
    graph = List()
    # initalize with empty lists
    for _ in range(labels.size()):
        # let numba infer the type of the list, is there a better way?
        new_list = []
        new_list.append((-1, -1))
        new_list.pop()
        graph.append(new_list)
    index_offset = labels.index_offset
    for i, label in enumerate(labels.perimeter):
        if label == 0:
            # this cell is not part of a flat
            continue
        # walk around the perimeter and connect neighboring cells in the same group
        # a neighboring cell is defined as the next or previous cell with the same label
        # as seen by continuing clockwise around the perimeter, not necessarily the
        # cell directly adjacent in the grid
        for j in range(i + 1, labels.size() + 1):
            k = j % labels.size()
            if k != i and labels.perimeter[k] == label:
                dist = labels.distance(i, k)
                graph[i].append((k + index_offset, dist))
                graph[k].append((i + index_offset, dist))
                break
        # connect non-neighboring cells to each other if they are part of the same group
        # we can skip any edges where there exist other paths that have the shortest
        # distances. For example, if the difference in the distances of vertex i to two
        # vertices (j and k) equals the distance of the
        # two end vertices to each other, only the closest vertex should be connected
        # since the path containing the vertices in between is among the shortest
        # paths between the two end vertices.
        # Theoretically, a fully connected graph would produce identical results.
        # However, this method is used to reduce the number of unnecessary edges in the graph.
        for j in range(i + 2, labels.size()):
            dist_ij = labels.distance(i, j)
            if labels.perimeter[j] != label:
                # this cell is not part of the same flat
                continue
            for k in range(i + 1, j):
                dist_ik = labels.distance(i, k)
                dist_kj = labels.distance(k, j)
                if dist_ij == dist_ik + dist_kj:
                    break
            else:
                graph[i].append((j + index_offset, dist_ij))
                graph[j].append((i + index_offset, dist_ij))

    # connect each cell to the high/low terrain node
    high_edges = List()
    low_edges = List()

    for i, distance in enumerate(to_higher.perimeter):
        if distance == 0:
            continue
        high_edges.append((i + index_offset, distance))

    for i, distance in enumerate(to_lower.perimeter):
        if distance == 0:
            continue
        low_edges.append((i + index_offset, distance))

    return graph, high_edges, low_edges


@njit
def join_neighbor(
    global_index_a: int,
    global_index_b: int,
    global_graph: list,
):
    # connect the two cells in the global graph
    global_graph[global_index_a].append((global_index_b, 1))
    global_graph[global_index_b].append((global_index_a, 1))


@njit
def handle_edge(
    labels_a: FlatTileEdgeCells,
    labels_b: FlatTileEdgeCells,
    side_a: Side,
    side_b: Side,
    global_graph: list,
):
    # iterate over the side of tile_a and tile_b
    # and connect the neighboring cells of the same label
    # neighboring cells are (-1, 0, 1) away
    labels_a_side = labels_a.get_side(side_a)
    labels_b_side = labels_b.get_side(side_b)
    for i, label_i in enumerate(labels_a_side):
        # if the cell is not part of a flat, skip it
        if label_i == 0:
            continue
        label_j = labels_b_side[i]
        global_index_a = labels_a.get_index_side(side_a, i) + labels_a.index_offset
        # if the neighboring cell directly adjacent is part of a flat, connect this cell to it
        if label_j != 0:
            global_index_b = labels_b.get_index_side(side_b, i) + labels_b.index_offset
            join_neighbor(global_index_a, global_index_b, global_graph)
        # if the neighboring cell diagonally is part of a flat, connect this cell to it
        if i != 0 and labels_b_side[i - 1] != 0:
            global_index_b = (
                labels_b.get_index_side(side_b, i - 1) + labels_b.index_offset
            )
            join_neighbor(global_index_a, global_index_b, global_graph)
        if i != labels_a_side.size - 1 and labels_b_side[i + 1] != 0:
            global_index_b = (
                labels_b.get_index_side(side_b, i + 1) + labels_b.index_offset
            )
            join_neighbor(global_index_a, global_index_b, global_graph)


@njit
def handle_corner(
    labels_a: FlatTileEdgeCells,
    labels_b: FlatTileEdgeCells,
    corner_a: Corner,
    corner_b: Corner,
    global_graph: list,
):
    label_a = labels_a.get_corner(corner_a)
    label_b = labels_b.get_corner(corner_b)
    if label_a != 0 and label_b != 0:
        global_index_a = labels_a.get_index_corner(corner_a) + labels_a.index_offset
        global_index_b = labels_b.get_index_corner(corner_b) + labels_b.index_offset
        join_neighbor(global_index_a, global_index_b, global_graph)


@njit
def join_adjacent_tiles(
    labels_a: FlatTileEdgeCells,
    labels_b: FlatTileEdgeCells,
    labels_c: FlatTileEdgeCells,
    labels_d: FlatTileEdgeCells,
    global_graph: list,
):
    # + - - + - - +
    # |  A  |  B  |
    # + - - * - - +
    # |  C  |  D  |
    # + - - + - - +
    # connect edge A-B
    handle_edge(
        labels_a,
        labels_b,
        Side.RIGHT,
        Side.LEFT,
        global_graph,
    )
    # connect edge B-D
    handle_edge(
        labels_b,
        labels_d,
        Side.BOTTOM,
        Side.TOP,
        global_graph,
    )
    # connect edge D-C
    handle_edge(
        labels_d,
        labels_c,
        Side.LEFT,
        Side.RIGHT,
        global_graph,
    )
    # connect edge C-A
    handle_edge(
        labels_c,
        labels_a,
        Side.TOP,
        Side.BOTTOM,
        global_graph,
    )
    # connect corner A-D
    handle_corner(
        labels_a,
        labels_d,
        Corner.BOTTOM_RIGHT,
        Corner.TOP_LEFT,
        global_graph,
    )
    # connect corner B-C
    handle_corner(
        labels_b,
        labels_c,
        Corner.BOTTOM_LEFT,
        Corner.TOP_RIGHT,
        global_graph,
    )


@njit
def solve_graph(global_graph: list, terrain_edges: list):
    min_dist = {}
    # using djikstra's algorithm, find the minimum distance to all cells on the perimeter
    # from terrain_edges which is a single node connecting all flat edge cells to the high terrain node
    # The result in min_dist will be the minimum distance to all cells in the graph
    # from the terrain node
    pq = []
    pq.append((0, 0))
    pq.pop()
    # populate pq with the terrain node and its neighbors
    for neighbor, weight in terrain_edges:
        min_dist[neighbor] = weight
        heapq.heappush(pq, (weight, neighbor))
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > min_dist[node]:
            continue
        for neighbor, weight in global_graph[node]:
            if neighbor not in min_dist:
                min_dist[neighbor] = dist + weight
                heapq.heappush(pq, (dist + weight, neighbor))
            elif dist + weight < min_dist[neighbor]:
                min_dist[neighbor] = dist + weight
                heapq.heappush(pq, (dist + weight, neighbor))
    return min_dist


def open_dataset(filepath, band_number=1, access=gdal.GA_ReadOnly):
    dataset = gdal.Open(filepath, access)
    if dataset is None:
        raise ValueError(f"Unable to open dataset: {filepath}")
    band = dataset.GetRasterBand(band_number)
    return dataset, band


def create_dataset(
    filepath,
    nodata_value,
    data_type,
    x_size,
    y_size,
    geotransform=None,
    projection=None,
    num_bands=1,
):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        filepath,
        x_size,
        y_size,
        num_bands,
        data_type,
        options=["COMPRESS=LZW"],
    )
    if geotransform:
        dataset.SetGeoTransform(geotransform)
    if projection:
        dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(nodata_value)
    return dataset, band


def setup_datasets(
    dem_filepath: str,
    fdr_filepath: str,
    working_dir: str,
    fixed_fdr_filepath: str,
):
    dem_ds, dem_band = open_dataset(dem_filepath)
    fdr_ds, fdr_band = open_dataset(fdr_filepath)
    x_size = dem_band.XSize
    y_size = dem_band.YSize
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()
    fixed_fdr_ds, fixed_fdr_band = create_dataset(
        fixed_fdr_filepath,
        FLOW_DIRECTION_NODATA,
        gdal.GDT_Byte,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    return (
        dem_ds,
        dem_band,
        fdr_ds,
        fdr_band,
        fixed_fdr_ds,
        fixed_fdr_band,
    )


def read_tile(filepath: str):
    dataset, band = open_dataset(filepath)
    return band.ReadAsArray()


def mosaic_tiles(
    prefix,
    rows,
    cols,
    chunk_size,
    output_filepath,
    working_dir,
    nodata_value,
    data_type,
    x_size,
    y_size,
    geotransform,
    projection,
):
    out_ds, out_band = create_dataset(
        output_filepath,
        nodata_value,
        data_type,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    for row in range(rows):
        for col in range(cols):
            tile_filepath = os.path.join(working_dir, f"{prefix}_{row}_{col}.tif")
            tile = read_tile(tile_filepath)
            chunk = RasterChunk(row, col, chunk_size, 0)
            chunk.from_numpy(tile)
            chunk.write(out_band)
    out_band.FlushCache()
    out_ds.FlushCache()
    out_ds = None
    out_band = None


def process_tiles(dem_ds, dem_band, fdr_band, chunk_size, working_directory):
    tile_labels = {}
    # setup the numba typed lists, is there a better way?
    global_graph = List()
    init_list = [(-1, -1)]
    global_graph.append(init_list)
    global_graph.pop()
    global_perimeter_high_edges = List()
    init_edge = (-1, -1)
    global_perimeter_high_edges.append(init_edge)
    global_perimeter_high_edges.pop()
    global_perimeter_low_edges = List()
    global_perimeter_low_edges.append(init_edge)
    global_perimeter_low_edges.pop()
    global_interior_high_edges = []
    global_interior_low_edges = []
    tile_index = 0
    for dem_tile in raster_chunker(dem_band, chunk_size, 1):
        # Here we use a 1 cell buffer, this differs from the paper.
        # This is done so that there are no 'uncertain' edges, only low and high edges.
        # This removes the need to handle uncertain edges and we don't need to save the
        # perimeter DEM elevation values since we no don't need to identify them later
        fdr_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
        fdr_tile.read(fdr_band)
        labels_tile_filepath = os.path.join(
            working_directory, f"labels_{dem_tile.row}_{dem_tile.col}.tif"
        )
        labels_tile_ds, labels_tile_band = create_dataset(
            labels_tile_filepath, 0, gdal.GDT_Int32, chunk_size, chunk_size
        )
        labels_tile = RasterChunk(0, 0, chunk_size, 1)
        # resolve_flats_tile removes the buffer region from the dem and flow_dirs when
        # creating the flat mask and labels
        to_higher, to_lower, labels, high_edges, low_edges = resolve_flats_tile(
            dem_tile.data, fdr_tile.data
        )
        labels_tile.from_numpy(labels)
        labels_tile.write(labels_tile_band)
        labels_tile_band.FlushCache()
        labels_tile_ds.FlushCache()
        labels_tile_ds = None
        labels_tile_band = None
        # remove buffer and create the tile edge data
        to_higher = to_higher[1:-1, 1:-1]
        to_lower = to_lower[1:-1, 1:-1]
        labels = labels[1:-1, 1:-1]
        # remove high and low edges from the perimeter in the buffer
        high_edges = [
            (row - 1, col - 1)
            for row, col in high_edges
            if row != 0 and row != chunk_size + 1 and col != 0 and col != chunk_size + 1
        ]
        low_edges = [
            (row - 1, col - 1)
            for row, col in low_edges
            if row != 0 and row != chunk_size + 1 and col != 0 and col != chunk_size + 1
        ]
        tile_labels[(dem_tile.row, dem_tile.col)] = FlatTileEdgeCells(
            labels, tile_index
        )
        graph, perimeter_high_edges, perimeter_low_edges = construct_local_edge_graph(
            tile_labels[(dem_tile.row, dem_tile.col)],
            FlatTileEdgeCells(to_higher, tile_index),
            FlatTileEdgeCells(to_lower, tile_index),
        )
        tile_index += 1
        global_graph += graph
        global_perimeter_high_edges += perimeter_high_edges
        global_perimeter_low_edges += perimeter_low_edges
        global_interior_high_edges.append(high_edges)
        global_interior_low_edges.append(low_edges)
    mosaic_tiles(
        "labels",
        math.ceil(dem_band.YSize / chunk_size),
        math.ceil(dem_band.XSize / chunk_size),
        chunk_size,
        os.path.join(working_directory, "labels.tif"),
        working_directory,
        0,
        gdal.GDT_Int32,
        dem_band.XSize,
        dem_band.YSize,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )
    return (
        tile_labels,
        global_graph,
        global_perimeter_high_edges,
        global_perimeter_low_edges,
        global_interior_high_edges,
        global_interior_low_edges,
    )


def fix_flats(
    dem_filepath: str,
    fdr_filepath: str,
    output_filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    working_dir: str = None,
):
    # create a temporary directory if one is not provided
    cleanup_working_dir = False
    if working_dir is None:
        # if we're creating a temporary directory, we should clean it up
        # otherwise, we leave it to the caller to clean up
        cleanup_working_dir = True
        working_dir = tempfile.mkdtemp()
    (
        dem_ds,
        dem_band,
        fdr_ds,
        fdr_band,
        fixed_fdr_ds,
        fixed_fdr_band,
    ) = setup_datasets(
        dem_filepath,
        fdr_filepath,
        working_dir,
        output_filepath,
    )
    (
        tile_labels,
        global_graph,
        global_perimeter_high_edges,
        global_perimeter_low_edges,
        global_interior_high_edges,
        global_interior_low_edges,
    ) = process_tiles(dem_ds, dem_band, fdr_band, chunk_size, working_dir)

    num_tile_rows = math.ceil(dem_band.YSize / chunk_size)
    num_tile_cols = math.ceil(dem_band.XSize / chunk_size)
    for i in range(num_tile_rows - 1):
        for j in range(num_tile_cols - 1):
            labels_a = tile_labels[(i, j)]
            labels_b = tile_labels[(i, j + 1)]
            labels_c = tile_labels[(i + 1, j)]
            labels_d = tile_labels[(i + 1, j + 1)]
            join_adjacent_tiles(
                labels_a,
                labels_b,
                labels_c,
                labels_d,
                global_graph,
            )

    min_dist_high = solve_graph(global_graph, global_perimeter_high_edges)
    min_dist_low = solve_graph(global_graph, global_perimeter_low_edges)
    # remove low edges from min_dist_high
    # if min_dist_low == 1, the cell is part of a low edge
    for i, dist in min_dist_low.items():
        if dist == 1 and i in min_dist_high:
            del min_dist_high[i]
    tile_index = 0
    for tile_row in range(num_tile_rows):
        for tile_col in range(num_tile_cols):
            fdr_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            fdr_tile.read(fdr_band)
            flat_mask_tile = RasterChunk(0, 0, chunk_size, 0)
            fdr = fdr_tile.data
            labels = read_tile(
                os.path.join(working_dir, f"labels_{tile_row}_{tile_col}.tif")
            )
            flat_mask = np.zeros_like(labels, dtype=np.int32)
            edge_labels = tile_labels[(tile_row, tile_col)]
            # add all interior_high_edges to adjusted_high_edges with value 1
            adjusted_high_edges = global_interior_high_edges[tile_index]
            # add all min_dist_high from this tile
            tile_min_dist_high = {
                k - edge_labels.index_offset: v
                for k, v in min_dist_high.items()
                if k >= edge_labels.index_offset
                and k < edge_labels.index_offset + edge_labels.size()
            }
            # key is loops, value is list of (row, col) tuples
            adjusted_high_edges_dict = {}
            max_high_loop = 0
            for i, dist in tile_min_dist_high.items():
                if dist == 1:
                    continue
                row, col = edge_labels.get_row_col(i)
                if dist not in adjusted_high_edges_dict:
                    max_high_loop = max(max_high_loop, dist)
                    adjusted_high_edges_dict[dist] = []
                adjusted_high_edges_dict[dist].append((row, col))
            # away from higher analog
            marker = (-1, -1)
            adjusted_high_edges.append(marker)
            adjusted_high_edges = ResizableFIFOQueue(adjusted_high_edges)
            loops = 1
            while len(adjusted_high_edges) > 1 or loops < max_high_loop:
                row, col = adjusted_high_edges.pop()
                if row == marker[0] and col == marker[1]:
                    loops += 1
                    if loops in adjusted_high_edges_dict:
                        for row, col in adjusted_high_edges_dict[loops]:
                            adjusted_high_edges.push((row, col))
                    adjusted_high_edges.push(marker)
                    continue
                if flat_mask[row, col] > 0:
                    continue
                flat_mask[row, col] = loops
                for neighbor_row, neighbor_col in neighbor_generator(
                    row, col, labels.shape[0], labels.shape[1]
                ):
                    if (
                        labels[neighbor_row, neighbor_col] == labels[row, col]
                        and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
                    ):
                        adjusted_high_edges.push((neighbor_row, neighbor_col))
            # towards lower analog
            adjusted_low_edges = global_interior_low_edges[tile_index]
            tile_min_dist_low = {
                k - edge_labels.index_offset: v
                for k, v in min_dist_low.items()
                if k >= edge_labels.index_offset
                and k < edge_labels.index_offset + edge_labels.size()
            }
            adjusted_low_edges_dict = {}
            max_low_loop = 0
            for i, dist in tile_min_dist_low.items():
                if dist == 1:
                    continue
                row, col = edge_labels.get_row_col(i)
                if dist not in adjusted_low_edges_dict:
                    max_low_loop = max(max_low_loop, dist)
                    adjusted_low_edges_dict[dist] = []
                adjusted_low_edges_dict[dist].append((row, col))
            adjusted_low_edges.append(marker)
            adjusted_low_edges = ResizableFIFOQueue(adjusted_low_edges)
            loops = 1
            flat_mask *= -1
            # LARGE_INT = (2**31 - 1) // 2
            LARGE_INT = 3
            while len(adjusted_low_edges) > 1 or loops < max_low_loop:
                row, col = adjusted_low_edges.pop()
                if row == marker[0] and col == marker[1]:
                    loops += 1
                    if loops in adjusted_low_edges_dict:
                        for row, col in adjusted_low_edges_dict[loops]:
                            adjusted_low_edges.push((row, col))
                    adjusted_low_edges.push(marker)
                    continue
                if flat_mask[row, col] > 0:
                    continue
                if flat_mask[row, col] < 0:
                    flat_mask[row, col] += LARGE_INT + 2 * loops
                else:
                    flat_mask[row, col] = 2 * loops
                for neighbor_row, neighbor_col in neighbor_generator(
                    row, col, labels.shape[0], labels.shape[1]
                ):
                    if (
                        labels[neighbor_row, neighbor_col] == labels[row, col]
                        and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
                    ):
                        adjusted_low_edges.push((neighbor_row, neighbor_col))
            flat_mask_filepath = os.path.join(
                working_dir, f"flat_mask_{tile_row}_{tile_col}.tif"
            )
            flat_mask_ds, flat_mask_band = create_dataset(
                flat_mask_filepath, 0, gdal.GDT_Int32, chunk_size, chunk_size
            )
            flat_mask_tile = RasterChunk(0, 0, chunk_size, 0)
            flat_mask_tile.from_numpy(flat_mask)
            flat_mask_tile.write(flat_mask_band)
            tile_index += 1
            flat_mask_band.FlushCache()
            flat_mask_ds.FlushCache()
            flat_mask_band = None
            flat_mask_ds = None
    mosaic_tiles(
        "flat_mask",
        math.ceil(dem_band.YSize / chunk_size),
        math.ceil(dem_band.XSize / chunk_size),
        chunk_size,
        os.path.join(working_dir, "flat_mask.tif"),
        working_dir,
        0,
        gdal.GDT_Int32,
        dem_band.XSize,
        dem_band.YSize,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )
    labels_ds, labels_band = open_dataset(
        os.path.join(working_dir, "labels.tif"), 1, gdal.GA_Update
    )
    flat_mask_ds, flat_mask_band = open_dataset(
        os.path.join(working_dir, "flat_mask.tif"), 1, gdal.GA_ReadOnly
    )
    # update fdr using d8_masked_flow_dirs
    for fdr_tile in raster_chunker(fdr_band, chunk_size, 1):
        labels_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        labels_tile.read(labels_band)
        flat_mask_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        flat_mask_tile.read(flat_mask_band)
        d8_masked_flow_dirs(flat_mask_tile.data, fdr_tile.data, labels_tile.data)
        fdr_tile.write(fixed_fdr_band)
    fixed_fdr_band.FlushCache()
    fixed_fdr_ds.FlushCache()
    # cleanup
    dem_band = None
    fdr_band = None
    labels_band = None
    flat_mask_band = None
    fixed_fdr_band = None
    dem_ds = None
    fdr_ds = None
    labels_ds = None
    flat_mask_ds = None
    fixed_fdr_ds = None
    if cleanup_working_dir:
        shutil.rmtree(working_dir)
