import math
import numpy as np
from numba import njit
from .util.raster import raster_chunker, RasterChunk
from osgeo import gdal
from .constants import (
    NEIGHBOR_OFFSETS,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTIONS,
    DEFAULT_CHUNK_SIZE,
    TERRAIN_ID,
)
from .util.raster import neighbor_generator
from .util.numba_datastructures import ResizableFIFOQueue
from enum import Enum


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


def get_index_row_col(
    index: int, num_top_labels: int, num_left_labels: int
) -> tuple[int, int]:
    row, col = -1, -1
    num_perimeter_cells = 2 * num_top_labels + 2 * num_left_labels
    # if i in top_edge, set row to 0
    if index < num_top_labels:
        row = 0
    # if i in bottom_edge, set row to len(left_edge) + 1
    if index >= num_top_labels and index < num_top_labels + num_top_labels:
        row = num_left_labels + 1
    # if i in left_edge, set col to 0
    if (
        index == 0
        or index == num_top_labels
        or (
            index >= num_top_labels + num_top_labels
            and index < num_top_labels + num_top_labels + num_left_labels
        )
    ):
        col = 0
    # if i in right_edge, set col to num_top_labels - 1
    if (
        index == num_top_labels - 1
        or index == num_top_labels + num_top_labels - 1
        or (
            index >= num_top_labels + num_top_labels + num_left_labels
            and index < num_perimeter_cells
        )
    ):
        col = num_top_labels - 1
    return row, col


def handle_edge(
    global_labels: list,
    edge_a_start_index,
    edge_b_start_index,
    edge_length,
    global_graph,
):
    for k in range(edge_length):
        if global_labels[edge_a_start_index + k] != 0:
            if k == 0:
                if global_labels[edge_b_start_index] != 0:
                    global_graph[edge_a_start_index + k].append((edge_b_start_index, 1))
                    global_graph[edge_b_start_index].append((edge_a_start_index + k, 1))
                if global_labels[edge_b_start_index + 1] != 0:
                    global_graph[edge_a_start_index + k].append(
                        (edge_b_start_index + 1, 1)
                    )
                    global_graph[edge_b_start_index + 1].append(
                        (edge_a_start_index + k, 1)
                    )
            if k == edge_length - 1:
                if global_labels[edge_b_start_index + edge_length - 1] != 0:
                    global_graph[edge_a_start_index + k].append(
                        (edge_b_start_index + edge_length - 1, 1)
                    )
                    global_graph[edge_b_start_index + edge_length - 1].append(
                        (edge_a_start_index + k, 1)
                    )
                if global_labels[edge_b_start_index + edge_length - 2] != 0:
                    global_graph[edge_a_start_index + k].append(
                        (edge_b_start_index + edge_length - 2, 1)
                    )
                    global_graph[edge_b_start_index + edge_length - 2].append(
                        (edge_a_start_index + k, 1)
                    )
            else:
                if global_labels[edge_b_start_index + k] != 0:
                    global_graph[edge_a_start_index + k].append(
                        (edge_b_start_index + k, 1)
                    )
                    global_graph[edge_b_start_index + k].append(
                        (edge_a_start_index + k, 1)
                    )
                if global_labels[edge_b_start_index + k + 1] != 0:
                    global_graph[edge_a_start_index + k].append(
                        (edge_b_start_index + k + 1, 1)
                    )
                    global_graph[edge_b_start_index + k + 1].append(
                        (edge_a_start_index + k, 1)
                    )
                if global_labels[edge_b_start_index + k - 1] != 0:
                    global_graph[edge_a_start_index + k].append(
                        (edge_b_start_index + k - 1, 1)
                    )
                    global_graph[edge_b_start_index + k - 1].append(
                        (edge_a_start_index + k, 1)
                    )


def handle_corner(
    global_labels: list,
    corner_a_index,
    corner_b_index,
    corner_c_index,
    corner_d_index,
    corner_e_index,
    corner_f_index,
    corner_g_index,
    corner_h_index,
    corner_i_index,
    corner_j_index,
    corner_k_index,
    corner_l_index,
    global_graph,
):
    #   a  b
    # c d  e f
    #
    # g h  i j
    #   k  l
    #
    # Define the connections for each corner
    connections = {
        corner_d_index: [
            corner_b_index,
            corner_e_index,
            corner_i_index,
            corner_h_index,
            corner_g_index,
        ],
        corner_e_index: [
            corner_a_index,
            corner_d_index,
            corner_h_index,
            corner_i_index,
            corner_j_index,
        ],
        corner_h_index: [
            corner_c_index,
            corner_d_index,
            corner_e_index,
            corner_i_index,
            corner_l_index,
        ],
        corner_i_index: [
            corner_f_index,
            corner_e_index,
            corner_d_index,
            corner_h_index,
            corner_k_index,
        ],
        corner_a_index: [corner_e_index],
        corner_b_index: [corner_d_index],
        corner_f_index: [corner_i_index],
        corner_j_index: [corner_e_index],
        corner_k_index: [corner_i_index],
        corner_l_index: [corner_h_index],
        corner_g_index: [corner_d_index],
        corner_c_index: [corner_h_index],
    }

    # Iterate over the connections and create them
    for corner, connects in connections.items():
        if global_labels[corner] != 0:
            for connect in connects:
                if global_labels[connect] != 0:
                    global_graph[corner].append((connect, 1))


import heapq


class TileEdgeCells:
    def __init__(self, array):
        (
            self.top,
            self.bottom,
            self.left,
            self.right,
            self.top_left_corner,
            self.top_right_corner,
            self.bottom_left_corner,
            self.bottom_right_corner,
        ) = self._get_perimeter_cells(array)

    def _get_perimeter_cells(
        self,
        array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the perimeter cells of the array.

        Args:
            array (np.ndarray): The array to get the perimeter cells of

        Returns:
            np.ndarray, np.ndarray, np.ndarray, np.ndarray: The perimeter cells of the array (top, bottom, left, right)
        """

        top_edge = np.ascontiguousarray(array[0, 1:-1])  # top row (excluding corners)
        bottom_edge = np.ascontiguousarray(
            array[-1, 1:-1]
        )  # bottom row (excluding corners)
        left_edge = np.ascontiguousarray(
            array[1:-1, 0]
        )  # left column (excluding corners)
        right_edge = np.ascontiguousarray(
            array[1:-1, -1]
        )  # right column (excluding corners)
        top_left_corner = array[0, 0]
        top_right_corner = array[0, -1]
        bottom_left_corner = array[-1, 0]
        bottom_right_corner = array[-1, -1]
        return (
            top_edge,
            bottom_edge,
            left_edge,
            right_edge,
            top_left_corner,
            top_right_corner,
            bottom_left_corner,
            bottom_right_corner,
        )


class TileEdgeData:
    def __init__(self, row, col, labels, to_higher, to_lower):
        self.to_higher = TileEdgeCells(to_higher)
        self.to_lower = TileEdgeCells(to_lower)
        self.labels = TileEdgeCells(labels)
        self.row = row
        self.col = col
        self.index_offset = # TODO, then use a map to producer to choose adjacent tiles when joining edges and corners


def construct_local_edge_graph(
    tile_data: TileEdgeData,
) -> list:
    # connect every cell to every other cell of the same label
    # to make a fully connected graph
    # TODO, optimize this. We only need to connect some of these cells
    index_offset = tile_index * len(labels_perimeter)
    graph = [[] for _ in range(len(labels_perimeter))]
    for i, label in enumerate(labels_perimeter):
        if label == 0:
            continue
        row_i, col_i = get_index_row_col(i, len(top_labels), len(left_labels))

        for j, other_label in enumerate(labels_perimeter):
            if other_label == label and i != j:
                row_j, col_j = get_index_row_col(j, len(top_labels), len(left_labels))
                distance = max(abs(row_i - row_j), abs(col_i - col_j))
                graph[i].append((j + index_offset, distance))
                graph[j].append((i + index_offset, distance))
    # connect each cell to the high/low terrain node
    terrain_edges = []
    for i, distance in enumerate(distance_perimeter):
        if labels_perimeter[i] == 0 or distance == 0:
            continue
        # graph[i].append((TERRAIN_ID, distance))
        terrain_edges.append((i + index_offset, distance))

    return graph, terrain_edges


def fix_flats(
    dem_filepath: str, fdr_filepath: str, chunk_size: int = DEFAULT_CHUNK_SIZE
):
    dem_ds = gdal.Open(dem_filepath)
    dem_band = dem_ds.GetRasterBand(1)
    fdr_ds = gdal.Open(fdr_filepath)
    fdr_band = fdr_ds.GetRasterBand(1)

    num_rows = dem_band.YSize
    num_cols = dem_band.XSize
    num_tile_rows = math.ceil(num_rows / chunk_size)
    num_tile_cols = math.ceil(num_cols / chunk_size)

    tile_edge_data = []
    # here we use a 1 cell buffer so that there
    # are no uncertain edges, only low and high edges
    # this removes the need to handle uncertain edges
    # and the need to save the perimeter DEM elevation values
    # for all tiles
    for dem_tile in raster_chunker(dem_band, chunk_size, 1):
        fdr_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
        fdr_tile.read(fdr_band)
        # resolve_flats_tile removes the buffer region from the dem and flow_dirs when
        # creating the flat mask and labels
        to_higher, to_lower, labels = resolve_flats_tile(dem_tile.data, fdr_tile.data)
        tile_edge_data.append(
            TileEdgeData(dem_tile.row, dem_tile.col, labels, to_higher, to_lower)
        )

    for dem_tile in raster_chunker(dem_band, chunk_size, 1):
        dem_tiles.append(dem_tile.data)
    for fdr_tile in raster_chunker(fdr_band, chunk_size, 1):
        fdr_tiles.append(fdr_tile.data)
    tile_index = 0
    global_high_graph = []
    global_low_graph = []
    global_high_terrain_edges = []
    global_low_terrain_edges = []
    global_labels = []
    for dem_tile, fdr_tile in zip(dem_tiles, fdr_tiles):
        dist_to_higher, dist_to_lower, labels = resolve_flats_tile(dem_tile, fdr_tile)
        # remove buffer region from dem and flow_dirs
        dem_tile = dem_tile[1:-1, 1:-1]
        fdr_tile = fdr_tile[1:-1, 1:-1]
        # construct high graph and low graph
        dem_perimeter = get_perimeter_cells(dem_tile)
        labels_perimeter = get_perimeter_cells(labels)
        dist_to_higher_perimeter = get_perimeter_cells(dist_to_higher)
        dist_to_lower_perimeter = get_perimeter_cells(dist_to_lower)
        high_graph, high_terrain_edges = construct_local_edge_graph(
            labels_perimeter, dist_to_higher_perimeter, tile_index
        )
        global_high_graph += high_graph
        global_high_terrain_edges += high_terrain_edges
        low_graph, low_terrain_edges = construct_local_edge_graph(
            labels_perimeter, dist_to_lower_perimeter, tile_index
        )
        global_low_graph += low_graph
        global_low_terrain_edges += low_terrain_edges
        global_labels = np.concatenate(
            (
                global_labels,
                labels_perimeter[0],
                labels_perimeter[1],
                labels_perimeter[2],
                labels_perimeter[3],
            )
        )
        tile_index += 1

    # for all 2x2 tiles, connect edges and corners
    # + - - + - - +
    # |  A  |  B  |
    # + - - * - - +
    # |  C  |  D  |
    # + - - + - - +
    for i in range(num_tile_rows - 1):
        for j in range(num_tile_cols - 1):
            tile_index = i * num_tile_cols + j
            # connect A-B edge (not including corners)
            perimeter_cells_per_tile = 2 * (chunk_size - 2) + 2 * chunk_size
            tile_a_index_offset = tile_index * perimeter_cells_per_tile
            right_edge_start_index = (
                tile_a_index_offset + 2 * chunk_size + (chunk_size - 2)
            )
            tile_b_index_offset = (tile_index + 1) * perimeter_cells_per_tile
            left_edge_start_index = tile_b_index_offset + 2 * chunk_size
            edge_length = chunk_size - 2
            handle_edge(
                global_labels,
                right_edge_start_index,
                left_edge_start_index,
                edge_length,
                global_high_graph,
            )
            handle_edge(
                global_labels,
                right_edge_start_index,
                left_edge_start_index,
                edge_length,
                global_low_graph,
            )
            # connect A-C edge (not including corners)
            bottom_edge_start_index = tile_a_index_offset + chunk_size + 1
            tile_c_index_offset = (
                tile_index + num_tile_cols
            ) * perimeter_cells_per_tile
            top_edge_start_index = tile_c_index_offset + 1
            handle_edge(
                global_labels,
                bottom_edge_start_index,
                top_edge_start_index,
                edge_length,
                global_high_graph,
            )
            handle_edge(
                global_labels,
                bottom_edge_start_index,
                top_edge_start_index,
                edge_length,
                global_low_graph,
            )
            # connect D-B edge (not including corners)
            tile_d_index_offset = (
                tile_index + num_tile_cols + 1
            ) * perimeter_cells_per_tile
            top_edge_start_index = tile_d_index_offset + 1
            bottom_edge_start_index = tile_b_index_offset + chunk_size + 1
            handle_edge(
                global_labels,
                top_edge_start_index,
                bottom_edge_start_index,
                edge_length,
                global_high_graph,
            )
            handle_edge(
                global_labels,
                top_edge_start_index,
                bottom_edge_start_index,
                edge_length,
                global_low_graph,
            )
            # connect D-C edge (not including corners)
            right_edge_start_index = (
                tile_c_index_offset + 2 * chunk_size + (chunk_size - 2)
            )
            left_edge_start_index = tile_d_index_offset + 2 * chunk_size
            handle_edge(
                global_labels,
                right_edge_start_index,
                left_edge_start_index,
                edge_length,
                global_high_graph,
            )
            handle_edge(
                global_labels,
                right_edge_start_index,
                left_edge_start_index,
                edge_length,
                global_low_graph,
            )

            # connect top left tile bottom right corner with top right tile top left corner
            corner_a_index = (
                tile_a_index_offset + 2 * chunk_size + 2 * (chunk_size - 2) - 1
            )
            corner_b_index = tile_b_index_offset + 2 * chunk_size + (chunk_size - 2) - 1
            corner_c_index = tile_a_index_offset + 2 * chunk_size - 2
            corner_d_index = tile_a_index_offset + 2 * chunk_size - 1
            corner_e_index = tile_b_index_offset + chunk_size
            corner_f_index = tile_b_index_offset + chunk_size + 1
            corner_g_index = tile_c_index_offset + chunk_size - 2
            corner_h_index = tile_c_index_offset + chunk_size - 1
            corner_i_index = tile_d_index_offset
            corner_j_index = tile_d_index_offset + 1
            corner_k_index = tile_c_index_offset + 2 * chunk_size + (chunk_size - 2)
            corner_l_index = tile_c_index_offset + 2 * chunk_size
            handle_corner(
                global_labels,
                corner_a_index,
                corner_b_index,
                corner_c_index,
                corner_d_index,
                corner_e_index,
                corner_f_index,
                corner_g_index,
                corner_h_index,
                corner_i_index,
                corner_j_index,
                corner_k_index,
                corner_l_index,
                global_high_graph,
            )
            handle_corner(
                global_labels,
                corner_a_index,
                corner_b_index,
                corner_c_index,
                corner_d_index,
                corner_e_index,
                corner_f_index,
                corner_g_index,
                corner_h_index,
                corner_i_index,
                corner_j_index,
                corner_k_index,
                corner_l_index,
                global_low_graph,
            )
    min_dist_high = []
    # using djikstra's algorithm, find the minimum distance to all cells on the perimeter
    # from global_high_terrain_edges which is a single node connecting all edge cells to the high terrain node
    # The result in min_dist_high will be the minimum distance to all cells in the high graph
    # from the high terrain node
    for i in range(len(global_high_graph)):
        min_dist_high.append(float("inf"))
    pq = []
    # populate pq with the high terrain node and its neighbors
    for neighbor, weight in global_high_terrain_edges:
        if weight < min_dist_high[neighbor]:
            min_dist_high[neighbor] = weight
            heapq.heappush(pq, (weight, neighbor))
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > min_dist_high[node]:
            continue
        for neighbor, weight in global_high_graph[node]:
            if dist + weight < min_dist_high[neighbor]:
                min_dist_high[neighbor] = dist + weight
                heapq.heappush(pq, (dist + weight, neighbor))

    # update the flow directions
    print("Updating flow directions")
