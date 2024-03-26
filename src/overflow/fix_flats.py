from enum import Enum
import math
import numpy as np
from numba import njit
from osgeo import gdal
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


def resolve_flats_tile(
    dem: np.ndarray, flow_dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1 ResolveFlats modified for use in the tile-based processing pipeline."""

    # Find flat edges, this will not result in any uncertain edges
    # because we are processing with a 1 cell buffer
    high_edges, low_edges = flat_edges(dem, flow_dirs)

    # Remove the buffer region from the edges, dem, and flow_dirs
    # this is necessary because the buffer region is not part of the tile
    # and we don't want to process it since it was only used to define the uncertain edges
    low_edges = [
        (row - 1, col - 1)
        for row, col in low_edges
        if row > 0 and col > 0 and row < dem.shape[0] - 1 and col < dem.shape[1] - 1
    ]
    high_edges = [
        (row - 1, col - 1)
        for row, col in high_edges
        if row > 0 and col > 0 and row < dem.shape[0] - 1 and col < dem.shape[1] - 1
    ]
    # remove buffer region from dem and flow_dirs
    dem = dem[1:-1, 1:-1]
    flow_dirs = flow_dirs[1:-1, 1:-1]

    # Initialize distance and labels arrays
    dist_to_higher = np.zeros_like(dem, dtype=np.int32)
    dist_to_lower = np.zeros_like(dem, dtype=np.int32)
    labels = np.zeros_like(dem, dtype=np.int32)

    label = 1
    # Label flats from low edges and high edges
    for row, col in low_edges + high_edges:
        if labels[row, col] == 0:
            label_flats(dem, labels, label, row, col)
            label += 1

    # Initialize max_dist arrays
    max_dist_to_higher = np.zeros(label, dtype=np.int32)
    max_dist_to_lower = np.zeros(label, dtype=np.int32)

    # Compute gradient away from higher terrain
    away_from_higher(labels, dist_to_higher, flow_dirs, high_edges, max_dist_to_higher)
    # Compute gradient towards lower terrain (same logic as away_from_higher so we reuse the function)
    away_from_higher(labels, dist_to_lower, flow_dirs, low_edges, max_dist_to_lower)

    return dist_to_higher, dist_to_lower, labels


import heapq


class FlatTileEdgeCells:
    def __init__(self, array):
        self.rows = len(array)
        self.cols = len(array[0])
        self.perimeter = []
        # self.perimeter contains one element for each cell in the tile perimeter
        # counting clockwise from the top left corner. This is the flattened
        # index of the cell in the local graph as well
        #
        # 0 | 1 | 2
        # 7 | - | 3
        # 6 | 5 | 4
        #
        self.perimeter.extend(array[0])  # Top edge
        self.perimeter.extend(
            row[-1] for row in array[1:-1]
        )  # Right edge (not including corners)
        self.perimeter.extend(array[-1][::-1])  # Bottom edge (reversed)
        self.perimeter.extend(
            row[0] for row in array[-2:0:-1]
        )  # Left edge (reversed, not including corners)
        self.perimeter = np.ascontiguousarray(self.perimeter)

    def get_flattened_index(self, row, col):
        #
        # 0 | 1 | 2
        # 7 | - | 3
        # 6 | 5 | 4
        #
        # check if row and col are on the perimeter
        if 0 < row < self.rows - 1 and 0 < col < self.cols - 1:
            raise IndexError("Row and col must be on the perimeter")
        # check bounds
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise IndexError("Row and col must be within bounds")
        if row == 0:  # Top edge
            return col
        elif row == self.rows - 1:  # Bottom edge
            return self.cols + self.rows - 2 + (self.cols - col - 1)
        elif col == self.cols - 1:  # Right edge
            return self.cols + row - 1
        else:  # Left edge
            return 2 * self.cols + self.rows - 2 + (self.rows - row - 2)

    def get_flattened_index_corner(self, corner: Corner):
        if corner == Corner.TOP_LEFT:
            return self.get_flattened_index(0, 0)
        elif corner == Corner.TOP_RIGHT:
            return self.get_flattened_index(0, self.cols - 1)
        elif corner == Corner.BOTTOM_RIGHT:
            return self.get_flattened_index(self.rows - 1, self.cols - 1)
        elif corner == Corner.BOTTOM_LEFT:
            return self.get_flattened_index(self.rows - 1, 0)
        else:
            raise ValueError("Invalid corner")

    def get_flattened_index_side(self, side: Side, index: int):
        if side == Side.TOP:
            return self.get_flattened_index(0, index)
        elif side == Side.RIGHT:
            return self.get_flattened_index(index, self.cols - 1)
        elif side == Side.BOTTOM:
            return self.get_flattened_index(self.rows - 1, index)
        elif side == Side.LEFT:
            return self.get_flattened_index(index, 0)
        else:
            raise ValueError("Invalid side")

    def get_row_col(self, index):
        #
        # 0 | 1 | 2
        # 7 | - | 3
        # 6 | 5 | 4
        #
        # check bounds
        if index < 0 or index >= self.perimeter.size:
            raise IndexError("Index out of bounds")
        if index < self.cols:  # Top edge
            return 0, index
        elif index < self.cols + self.rows - 1:  # Right edge
            return index - self.cols + 1, self.cols - 1
        elif index < 2 * self.cols + self.rows - 2:  # Bottom edge
            return self.rows - 1, self.cols - (index - self.cols - self.rows + 3)
        else:  # Left edge
            return self.rows - (index - 2 * self.cols - self.rows + 4), 0

    def size(self):
        return self.perimeter.size

    def get_side(self, side: Side):
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
            return np.concatenate(
                (self.perimeter[-self.rows + 1 :], [self.perimeter[0]])
            )[::-1]
        else:
            raise ValueError("Invalid side")

    def get_corner(self, corner: Corner):
        return self.perimeter[self.get_flattened_index_corner(corner)]

    def distance(
        self,
        index: int,
        other_index: int,
    ) -> int:
        from_row, from_col = self.get_row_col(index)
        to_row, to_col = self.get_row_col(other_index)
        return max(abs(from_row - to_row), abs(from_col - to_col))


class FlatTileEdgeData:
    def __init__(self, row, col, tile_index, labels, to_higher, to_lower):
        self.to_higher = FlatTileEdgeCells(to_higher)
        self.to_lower = FlatTileEdgeCells(to_lower)
        self.labels = FlatTileEdgeCells(labels)
        self.row = row
        self.col = col
        # the offset of the starting index in the global graph for this tile
        self.index_offset = tile_index * self.labels.size()


def construct_local_edge_graph(
    tile_data: FlatTileEdgeData,
) -> tuple[list, list, list, list]:
    # connect every cell to every other cell of the same label
    # to make a fully connected graph
    # TODO: optimize this. We only need to connect some of these cells
    graph = [[] for _ in range(tile_data.labels.size())]
    index_offset = tile_data.index_offset
    labels = tile_data.labels.perimeter
    for i, label_i in enumerate(labels):
        if label_i == 0:
            continue
        label_i = labels[i]
        for j in range(i + 1, len(labels)):
            if label_i == labels[j]:
                dist = tile_data.labels.distance(i, j)
                graph[i].append((j + index_offset, dist))
                graph[j].append((i + index_offset, dist))

    # connect each cell to the high/low terrain node
    high_edges = []
    low_edges = []
    to_higher = tile_data.to_higher.perimeter
    to_lower = tile_data.to_lower.perimeter

    for i, distance in enumerate(to_higher):
        if distance == 0:
            continue
        high_edges.append((i + index_offset, distance))

    for i, distance in enumerate(to_lower):
        if distance == 0:
            continue
        low_edges.append((i + index_offset, distance))

    return graph, high_edges, low_edges


def join_neighbor(
    global_index_a: int,
    global_index_b: int,
    global_graph: list,
):
    # connect the two cells in the global graph
    global_graph[global_index_a].append((global_index_b, 1))
    global_graph[global_index_b].append((global_index_a, 1))


def handle_edge(
    tile_a: FlatTileEdgeData,
    tile_b: FlatTileEdgeData,
    side_a: Side,
    side_b: Side,
    global_graph: list,
):
    # iterate over the side of tile_a and tile_b
    # and connect the neighboring cells of the same label
    # neighboring cells are (-1, 0, 1) away
    tile_a_side = tile_a.labels.get_side(side_a)
    tile_b_side = tile_b.labels.get_side(side_b)
    for i, label_i in enumerate(tile_a_side):
        # if the cell is not part of a flat, skip it
        if label_i == 0:
            continue
        label_j = tile_b_side[i]
        global_index_a = (
            tile_a.labels.get_flattened_index_side(side_a, i) + tile_a.index_offset
        )
        global_index_b = (
            tile_b.labels.get_flattened_index_side(side_b, i) + tile_b.index_offset
        )
        # if the neighboring cell directly adjacent is part of a flat, connect this cell to it
        if label_j != 0:
            join_neighbor(global_index_a, global_index_b, global_graph)
        # if the neighboring cell diagonally is part of a flat, connect this cell to it
        if i != 0 and tile_b_side[i - 1] != 0:
            join_neighbor(global_index_a, global_index_b - 1, global_graph)
        if i != tile_a_side.size - 1 and tile_b_side[i + 1] != 0:
            join_neighbor(global_index_a, global_index_b + 1, global_graph)


def handle_corner(
    tile_a: FlatTileEdgeData,
    tile_b: FlatTileEdgeData,
    corner_a: Corner,
    corner_b: Corner,
    global_graph: list,
):
    label_a = tile_a.labels.get_corner(corner_a)
    label_b = tile_b.labels.get_corner(corner_b)
    if label_a != 0 and label_b != 0:
        global_index_a = (
            tile_a.labels.get_flattened_index_corner(corner_a) + tile_a.index_offset
        )
        global_index_b = (
            tile_b.labels.get_flattened_index_corner(corner_b) + tile_b.index_offset
        )
        join_neighbor(global_index_a, global_index_b, global_graph)


def join_adjacent_tiles(
    tile_a: FlatTileEdgeData,
    tile_b: FlatTileEdgeData,
    tile_c: FlatTileEdgeData,
    tile_d: FlatTileEdgeData,
    global_graph: list,
):
    # + - - + - - +
    # |  A  |  B  |
    # + - - * - - +
    # |  C  |  D  |
    # + - - + - - +
    # connect edge A-B
    handle_edge(tile_a, tile_b, Side.RIGHT, Side.LEFT, global_graph)
    # connect edge B-D
    handle_edge(tile_b, tile_d, Side.BOTTOM, Side.TOP, global_graph)
    # connect edge D-C
    handle_edge(tile_d, tile_c, Side.LEFT, Side.RIGHT, global_graph)
    # connect edge C-A
    handle_edge(tile_c, tile_a, Side.TOP, Side.BOTTOM, global_graph)
    # connect corner A-D
    handle_corner(tile_a, tile_d, Corner.BOTTOM_RIGHT, Corner.TOP_LEFT, global_graph)
    # connect corner B-C
    handle_corner(tile_b, tile_c, Corner.BOTTOM_LEFT, Corner.TOP_RIGHT, global_graph)


def fix_flats(
    dem_filepath: str, fdr_filepath: str, chunk_size: int = DEFAULT_CHUNK_SIZE
):
    dem_ds = gdal.Open(dem_filepath)
    dem_band = dem_ds.GetRasterBand(1)
    fdr_ds = gdal.Open(fdr_filepath)
    fdr_band = fdr_ds.GetRasterBand(1)

    tile_edge_data = {}
    # here we use a 1 cell buffer so that there
    # are no uncertain edges, only low and high edges
    # this removes the need to handle uncertain edges
    # and the need to save the perimeter DEM elevation values
    # for all tiles
    global_graph = []
    global_high_edges = []
    global_low_edges = []
    tile_index = 0
    for dem_tile in raster_chunker(dem_band, chunk_size, 1):
        fdr_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
        fdr_tile.read(fdr_band)
        # resolve_flats_tile removes the buffer region from the dem and flow_dirs when
        # creating the flat mask and labels
        to_higher, to_lower, labels = resolve_flats_tile(dem_tile.data, fdr_tile.data)
        tile_edge_data[(dem_tile.row, dem_tile.col)] = FlatTileEdgeData(
            dem_tile.row, dem_tile.col, tile_index, labels, to_higher, to_lower
        )
        tile_index += 1
        local_graph, local_high_edges, local_low_edges = construct_local_edge_graph(
            tile_edge_data[(dem_tile.row, dem_tile.col)]
        )
        global_graph += local_graph
        global_high_edges += local_high_edges
        global_low_edges += local_low_edges

    # for all 2x2 tiles, connect edges and corners
    # + - - + - - +
    # |  A  |  B  |
    # + - - * - - +
    # |  C  |  D  |
    # + - - + - - +
    num_tile_rows = math.ceil(dem_band.YSize / chunk_size)
    num_tile_cols = math.ceil(dem_band.XSize / chunk_size)
    for i in range(num_tile_rows - 1):
        for j in range(num_tile_cols - 1):
            A = tile_edge_data[(i, j)]
            B = tile_edge_data[(i, j + 1)]
            C = tile_edge_data[(i + 1, j)]
            D = tile_edge_data[(i + 1, j + 1)]
            join_adjacent_tiles(A, B, C, D, global_graph)

    min_dist_high = []
    # using djikstra's algorithm, find the minimum distance to all cells on the perimeter
    # from global_high_terrain_edges which is a single node connecting all edge cells to the high terrain node
    # The result in min_dist_high will be the minimum distance to all cells in the high graph
    # from the high terrain node
    for i in range(len(global_graph)):
        min_dist_high.append(float("inf"))
    pq = []
    # populate pq with the high terrain node and its neighbors
    for neighbor, weight in global_high_edges:
        if weight < min_dist_high[neighbor]:
            min_dist_high[neighbor] = weight
            heapq.heappush(pq, (weight, neighbor))
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > min_dist_high[node]:
            continue
        for neighbor, weight in global_graph[node]:
            if dist + weight < min_dist_high[neighbor]:
                min_dist_high[neighbor] = dist + weight
                heapq.heappush(pq, (dist + weight, neighbor))
    # update the flow directions
    print("Updating flow directions")
