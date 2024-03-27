from enum import Enum
import os
import shutil
import math
import heapq
import tempfile
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


def away_from_higher_tile(
    labels: np.ndarray,
    flat_mask: np.ndarray,
    fdr: np.ndarray,
    high_edges: list,
    flat_height: np.array,
) -> None:
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


def towards_lower_tile(
    labels: np.ndarray,
    flat_mask: np.ndarray,
    fdr: np.ndarray,
    low_edges: list,
    flat_height: np.array,
) -> None:
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


def compute_gradient(
    labels: np.ndarray,
    fdr: np.ndarray,
    high_edges: list,
) -> np.ndarray:
    graident = np.zeros_like(labels, dtype=np.int32)
    if len(high_edges) == 0:
        return graident
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
        if graident[row, col] > 0:
            continue
        graident[row, col] = loops
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, fdr.shape[0], fdr.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                high_edges.push((neighbor_row, neighbor_col))
    return graident


def resolve_flats_tile(
    dem: np.ndarray, flow_dirs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algorithm 1 ResolveFlats modified for use in the tile-based processing pipeline."""

    # Find flat edges, this will not result in any uncertain edges
    # because we are processing with a 1 cell buffer
    high_edges, low_edges = flat_edges(dem, flow_dirs)

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
    # Compute gradient towards lower terrain
    dist_to_lower = compute_gradient(labels, flow_dirs, low_edges)

    return dist_to_higher, dist_to_lower, labels, high_edges, low_edges


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
) -> tuple[list, list, list]:

    graph = [[] for _ in range(tile_data.labels.size())]
    index_offset = tile_data.index_offset
    labels = tile_data.labels.perimeter
    # get all groups of cells with the same label as lists with indices
    label_groups = {}
    for i, label in enumerate(labels):
        if label == 0:
            continue
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(i)

    # walk around the perimeter and connect neighboring cells in the same group
    # a neighboring cell is defined as the next or previous cell with the same label
    # as seen by continuing clockwise around the perimeter, not necessarily the
    # cell directly adjacent in the grid
    for group in label_groups.values():
        for i, group_i in enumerate(group):
            # connect to the next cell with the same label
            j = (i + 1) % len(group)
            dist = tile_data.labels.distance(group_i, group[j])
            graph[group_i].append((group[j] + index_offset, dist))
            graph[group[j]].append((group_i + index_offset, dist))

        # connect non-neighboring cells to each other if they are part of the same group
        # we can skip any edges where there exist other paths that have the shortest
        # distances. For example, if the difference in the distances of vertex i to two
        # vertices (j and k) equals the distance of the
        # two end vertices to each other, only the closest vertex should be connected
        # since the path containing the vertices in between is among the shortest
        # paths between the two end vertices.
        # Theoretically, a fully connected graph would produce identical results.
        # However, this method is used to reduce the number of edges in the graph.
        for i, group_i in enumerate(group):
            for j in range(i + 2, len(group)):
                dist_ij = tile_data.labels.distance(group_i, group[j])
                for k in range(i + 1, j):
                    dist_ik = tile_data.labels.distance(group_i, group[k])
                    dist_kj = tile_data.labels.distance(group[k], group[j])
                    if dist_ij == dist_ik + dist_kj:
                        break
                else:
                    graph[group_i].append((group[j] + index_offset, dist_ij))
                    graph[group[j]].append((group_i + index_offset, dist_ij))

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
        # if the neighboring cell directly adjacent is part of a flat, connect this cell to it
        if label_j != 0:
            global_index_b = (
                tile_b.labels.get_flattened_index_side(side_b, i) + tile_b.index_offset
            )
            join_neighbor(global_index_a, global_index_b, global_graph)
        # if the neighboring cell diagonally is part of a flat, connect this cell to it
        if i != 0 and tile_b_side[i - 1] != 0:
            global_index_b = (
                tile_b.labels.get_flattened_index_side(side_b, i - 1)
                + tile_b.index_offset
            )
            join_neighbor(global_index_a, global_index_b, global_graph)
        if i != tile_a_side.size - 1 and tile_b_side[i + 1] != 0:
            global_index_b = (
                tile_b.labels.get_flattened_index_side(side_b, i + 1)
                + tile_b.index_offset
            )
            join_neighbor(global_index_a, global_index_b, global_graph)


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


def compute_dist(global_graph: list, terrain_edges: list):
    min_dist = []
    # using djikstra's algorithm, find the minimum distance to all cells on the perimeter
    # from terrain_edges which is a single node connecting all flat edge cells to the high terrain node
    # The result in min_dist will be the minimum distance to all cells in the graph
    # from the terrain node
    for i in range(len(global_graph)):
        min_dist.append(float("inf"))
    pq = []
    # populate pq with the terrain node and its neighbors
    for neighbor, weight in terrain_edges:
        if weight < min_dist[neighbor]:
            min_dist[neighbor] = weight
            heapq.heappush(pq, (weight, neighbor))
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > min_dist[node]:
            continue
        for neighbor, weight in global_graph[node]:
            if dist + weight < min_dist[neighbor]:
                min_dist[neighbor] = dist + weight
                heapq.heappush(pq, (dist + weight, neighbor))
    return min_dist


def fix_flats(
    dem_filepath: str,
    fdr_filepath: str,
    output_filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    working_dir: str = None,
):
    dem_ds = gdal.Open(dem_filepath)
    dem_band = dem_ds.GetRasterBand(1)
    fdr_ds = gdal.Open(fdr_filepath)
    fdr_band = fdr_ds.GetRasterBand(1)
    # create a temporary directory if one is not provided
    cleanup_working_dir = False
    if working_dir is None:
        cleanup_working_dir = True
        working_dir = tempfile.mkdtemp()
    labels_ds = gdal.GetDriverByName("GTiff").Create(
        os.path.join(working_dir, "labels.tif"),
        dem_band.XSize,
        dem_band.YSize,
        1,
        gdal.GDT_Int32,
    )
    labels_ds.SetGeoTransform(dem_ds.GetGeoTransform())
    labels_ds.SetProjection(dem_ds.GetProjection())
    labels_band = labels_ds.GetRasterBand(1)
    labels_band.SetNoDataValue(0)

    flat_mask_ds = gdal.GetDriverByName("GTiff").Create(
        os.path.join(working_dir, "flat_mask.tif"),
        dem_band.XSize,
        dem_band.YSize,
        1,
        gdal.GDT_Int32,
    )
    flat_mask_ds.SetGeoTransform(dem_ds.GetGeoTransform())
    flat_mask_ds.SetProjection(dem_ds.GetProjection())
    flat_mask_band = flat_mask_ds.GetRasterBand(1)
    flat_mask_band.SetNoDataValue(0)

    fixed_fdr_ds = gdal.GetDriverByName("GTiff").Create(
        output_filepath,
        fdr_band.XSize,
        fdr_band.YSize,
        1,
        gdal.GDT_Byte,
    )
    fixed_fdr_ds.SetGeoTransform(fdr_ds.GetGeoTransform())
    fixed_fdr_ds.SetProjection(fdr_ds.GetProjection())
    fixed_fdr_band = fixed_fdr_ds.GetRasterBand(1)
    fixed_fdr_band.SetNoDataValue(FLOW_DIRECTION_NODATA)

    tile_edge_data = {}
    # here we use a 1 cell buffer so that there
    # are no uncertain edges, only low and high edges
    # this removes the need to handle uncertain edges
    # and the need to save the perimeter DEM elevation values
    # for all tiles
    global_graph = []
    global_high_edges = []
    global_low_edges = []
    interior_high_edges = []
    interior_low_edges = []
    tile_index = 0
    for dem_tile in raster_chunker(dem_band, chunk_size, 1):
        fdr_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
        fdr_tile.read(fdr_band)
        labels_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
        # resolve_flats_tile removes the buffer region from the dem and flow_dirs when
        # creating the flat mask and labels
        to_higher, to_lower, labels, high_edges, low_edges = resolve_flats_tile(
            dem_tile.data, fdr_tile.data
        )
        labels_tile.from_numpy(labels)
        labels_tile.write(labels_band)
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
        interior_high_edges.append(high_edges)
        interior_low_edges.append(low_edges)
    # flush cached data
    labels_band.FlushCache()
    labels_ds.FlushCache()

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

    min_dist_high = compute_dist(global_graph, global_high_edges)
    min_dist_low = compute_dist(global_graph, global_low_edges)
    # remove low edges from min_dist_high
    # if min_dist_low == 1, the cell is part of a low edge
    for i, dist in enumerate(min_dist_low):
        if dist == 1:
            min_dist_high[i] = float("inf")
    tile_index = 0
    for labels_tile in raster_chunker(labels_band, chunk_size, 0):
        fdr_tile = RasterChunk(labels_tile.row, labels_tile.col, chunk_size, 0)
        fdr_tile.read(fdr_band)
        flat_mask_tile = RasterChunk(labels_tile.row, labels_tile.col, chunk_size, 0)

        fdr = fdr_tile.data
        labels = labels_tile.data
        flat_mask = np.zeros_like(labels, dtype=np.int32)
        edge_data = tile_edge_data[(labels_tile.row, labels_tile.col)]
        flat_height = np.zeros(labels.max(), dtype=np.int32)
        # add all interior_high_edges to adjusted_high_edges with value 1
        adjusted_high_edges = interior_high_edges[tile_index]
        # add all min_dist_high from this tile
        tile_min_dist_high = min_dist_high[
            edge_data.index_offset : edge_data.index_offset + edge_data.labels.size()
        ]
        # key is loops, value is list of (row, col) tuples
        adjusted_high_edges_dict = {}
        max_high_loop = 0
        for i, dist in enumerate(tile_min_dist_high):
            if dist == float("inf") or dist == 1:
                continue
            row, col = edge_data.labels.get_row_col(i)
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
            flat_height[labels[row, col] - 1] = loops
            for neighbor_row, neighbor_col in neighbor_generator(
                row, col, labels.shape[0], labels.shape[1]
            ):
                if (
                    labels[neighbor_row, neighbor_col] == labels[row, col]
                    and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
                ):
                    adjusted_high_edges.push((neighbor_row, neighbor_col))
        # towards lower analog
        adjusted_low_edges = interior_low_edges[tile_index]
        tile_min_dist_low = min_dist_low[
            edge_data.index_offset : edge_data.index_offset + edge_data.labels.size()
        ]
        adjusted_low_edges_dict = {}
        max_low_loop = 0
        for i, dist in enumerate(tile_min_dist_low):
            if dist == float("inf") or dist == 1:
                continue
            row, col = edge_data.labels.get_row_col(i)
            if dist not in adjusted_low_edges_dict:
                max_low_loop = max(max_low_loop, dist)
                adjusted_low_edges_dict[dist] = []
            adjusted_low_edges_dict[dist].append((row, col))
        adjusted_low_edges.append(marker)
        adjusted_low_edges = ResizableFIFOQueue(adjusted_low_edges)
        loops = 1
        flat_mask *= -1
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
                flat_mask[row, col] += flat_height[labels[row, col] - 1] + 2 * loops
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
        flat_mask_tile.from_numpy(flat_mask)
        flat_mask_tile.write(flat_mask_band)
        tile_index += 1
    flat_mask_band.FlushCache()
    flat_mask_ds.FlushCache()
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
