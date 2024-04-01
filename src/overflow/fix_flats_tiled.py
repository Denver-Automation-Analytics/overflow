import os
import shutil
import math
import heapq
import tempfile
import numpy as np
from numba import njit, int64, prange, int32
from osgeo import gdal
import concurrent.futures
from numba.typed import List, Dict
from numba.types import UniTuple, Tuple, Array
from .util.raster import (
    raster_chunker,
    RasterChunk,
    neighbor_generator,
    Side,
    Corner,
    TileEdgeCells,
    open_dataset,
    create_dataset,
    read_tile,
    mosaic_tiles,
)
from .fix_flats import flat_edges, label_flats, d8_masked_flow_dirs
from .constants import (
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_UNDEFINED,
    DEFAULT_CHUNK_SIZE,
)
from .util.numba_datastructures import Queue


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
    edges_cells = Queue(edges_cells)
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
def filter_edges(edges, dem_shape):
    filtered_edges = List()
    for row, col in edges:
        if (
            row != 0
            and row != dem_shape[0] + 1
            and col != 0
            and col != dem_shape[1] + 1
        ):
            filtered_edges.append((row - 1, col - 1))
    return filtered_edges


@njit
def flat_edges_tile(dem: np.ndarray, fdr: np.ndarray) -> tuple[list, list]:
    """TODO"""
    # FIFO queues for the high and low edge cells
    high_edges = List()
    low_edges = List()

    for row in range(1, dem.shape[0] - 1):
        for col in range(1, dem.shape[1] - 1):
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
                    low_edges.append((row - 1, col - 1))
                    break
                if (
                    fdr_current == FLOW_DIRECTION_UNDEFINED
                    and dem[row, col] < dem[neighbor_row, neighbor_col]
                ):
                    # cell is a high edge cell since it has no defined flow direction and a neighbor is higher
                    high_edges.append((row - 1, col - 1))
                    break
    return high_edges, low_edges


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
    high_edges, low_edges = flat_edges_tile(dem, flow_dirs)

    # remove buffer from dem and flow_dirs since we now have the edges
    dem = dem[1:-1, 1:-1]
    flow_dirs = flow_dirs[1:-1, 1:-1]

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
    for row, col in low_edges:
        if labels[row, col] == 0:
            label_flats(dem, labels, label, row, col)
            label += 1
    for row, col in high_edges:
        if labels[row, col] == 0:
            label_flats(dem, labels, label, row, col)
            label += 1

    # Compute gradient away from higher terrain
    dist_to_higher = compute_gradient(labels, flow_dirs, high_edges)
    # Compute gradient away from lower terrain
    dist_to_lower = compute_gradient(labels, flow_dirs, low_edges)

    return dist_to_higher, dist_to_lower, labels, high_edges, low_edges


@njit
def get_tile_exterior_edges(min_dist, edge_labels):
    # get all min_dist elements from this tile
    tile_min_dist = {
        k - edge_labels.index_offset: v
        for k, v in min_dist.items()
        if k >= edge_labels.index_offset
        and k < edge_labels.index_offset + edge_labels.size()
    }
    # The dictionary is used to group cells by distance from the flat edge
    # Each loop is a list of cells that are the same distance from the flat edge
    exterior_edges = List()
    max_loop = 0
    loop_index_map = {}
    index = 0
    for i, dist in tile_min_dist.items():
        if dist == 1:
            continue
        row, col = edge_labels.get_row_col(i)
        if dist not in loop_index_map:
            max_loop = max(max_loop, dist)
            loop_index_map[dist] = index
            index += 1
            new_edge = List()
            new_edge.append((row, col))
            exterior_edges.append(new_edge)
        else:
            exterior_edges[loop_index_map[dist]].append((row, col))
    return exterior_edges, max_loop, loop_index_map


@njit
def away_from_higher_tile(
    interior_high_edges,
    exterior_high_edges,
    loop_index_map,
    max_high_loop,
    labels,
    fdr,
    flat_mask,
):
    marker = (-1, -1)
    interior_high_edges.append(marker)
    interior_high_edges = Queue(interior_high_edges)
    loops = 1
    while len(interior_high_edges) > 1 or loops < max_high_loop:
        row, col = interior_high_edges.pop()
        if row == marker[0] and col == marker[1]:
            loops += 1
            if loops in loop_index_map:
                for row, col in exterior_high_edges[loop_index_map[loops]]:
                    interior_high_edges.push((row, col))
            if len(interior_high_edges) > 1:
                interior_high_edges.push(marker)
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
                interior_high_edges.push((neighbor_row, neighbor_col))


@njit
def towards_lower_tile(
    interior_low_edges,
    exterior_low_edges,
    loop_index_map,
    max_low_loop,
    labels,
    fdr,
    flat_mask,
):
    marker = (-1, -1)
    interior_low_edges.append(marker)
    interior_low_edges = Queue(interior_low_edges)
    loops = 1
    flat_mask *= -1
    # max_flat_height can be any arbitrary large number
    # as long as is is greater than the maximum distance
    # to any high edge cell in the flat
    max_flat_height = (2**31 - 1) // 2
    while len(interior_low_edges) > 1 or loops < max_low_loop:
        row, col = interior_low_edges.pop()
        if row == marker[0] and col == marker[1]:
            loops += 1
            if loops in loop_index_map:
                for row, col in exterior_low_edges[loop_index_map[loops]]:
                    interior_low_edges.push((row, col))
            if len(interior_low_edges) > 1:
                interior_low_edges.push(marker)
            continue
        if flat_mask[row, col] > 0:
            continue
        if flat_mask[row, col] < 0:
            flat_mask[row, col] += max_flat_height + 2 * loops
        else:
            flat_mask[row, col] = 2 * loops
        for neighbor_row, neighbor_col in neighbor_generator(
            row, col, labels.shape[0], labels.shape[1]
        ):
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                interior_low_edges.push((neighbor_row, neighbor_col))


@njit(parallel=True)
def compute_min_dist(labels_array, labels_edge_cells):
    perimeter = labels_edge_cells.perimeter
    min_dist = np.zeros((perimeter.size, perimeter.size), dtype=np.int32)
    for i in prange(perimeter.size):
        label = perimeter[i]
        if label == 0:
            continue
        visited = set()
        dist = 0
        from_row, from_col = labels_edge_cells.get_row_col(i)
        marker = (-1, -1)
        queue = Queue([(from_row, from_col)])
        queue.push(marker)
        while len(queue) > 1:
            row, col = queue.pop()
            if (row, col) == marker:
                dist += 1
                queue.push(marker)
                continue
            if (row, col) in visited:
                continue
            visited.add((row, col))
            if (
                row == 0
                or row == labels_array.shape[0] - 1
                or col == 0
                or col == labels_array.shape[1] - 1
            ):
                j = labels_edge_cells.get_index(row, col)
                if min_dist[i][j] > 0:
                    continue
                min_dist[i][j] = dist
            for neighbor_row, neighbor_col in neighbor_generator(
                row, col, labels_array.shape[0], labels_array.shape[1]
            ):
                if labels_array[neighbor_row, neighbor_col] == label:
                    queue.push((neighbor_row, neighbor_col))

    return min_dist


@njit
def construct_local_edge_graph(
    labels: TileEdgeCells,
    to_higher: TileEdgeCells,
    to_lower: TileEdgeCells,
    labels_array: np.ndarray,
) -> tuple[list, list, list]:
    # create a list of lists to represent the graph
    # there are at most labels.size() vertices in the graph
    graph = List()
    # initalize with empty lists
    for _ in range(labels.size()):
        # let numba infer the type of the list, is there a better way?
        new_list = List()
        new_list.append((-1, -1))
        new_list.pop()
        graph.append(new_list)
    index_offset = labels.index_offset

    min_dists = compute_min_dist(labels_array, labels)
    # for i, dists in min_dists.items():
    #     for j, dist in enumerate(dists):
    #         if dist != -1:
    #             graph[i].append((j + index_offset, dist))
    #             graph[j].append((i + index_offset, dist))

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
                dist = min_dists[i][k]
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
            if labels.perimeter[j] != label:
                # this cell is not part of the same flat
                continue
            dist_ij = min_dists[i][j]
            for k in range(i + 1, j):
                if labels.perimeter[k] != label:
                    continue
                dist_ik = min_dists[i][k]
                dist_kj = min_dists[k][j]
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
        high_edges.append((i + index_offset, int64(distance)))

    for i, distance in enumerate(to_lower.perimeter):
        if distance == 0:
            continue
        low_edges.append((i + index_offset, int64(distance)))

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
    labels_a: TileEdgeCells,
    labels_b: TileEdgeCells,
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
    labels_a: TileEdgeCells,
    labels_b: TileEdgeCells,
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
    labels_a: TileEdgeCells,
    labels_b: TileEdgeCells,
    labels_c: TileEdgeCells,
    labels_d: TileEdgeCells,
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
def solve_low_graph(global_graph: list, terrain_edges: list):
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


@njit
def solve_high_graph(global_graph: list, terrain_edges: list, min_dist_low: dict):
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
        # ignore low edge cells
        if neighbor in min_dist_low and min_dist_low[neighbor] == 1:
            continue
        min_dist[neighbor] = weight
        heapq.heappush(pq, (weight, neighbor))
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > min_dist[node]:
            continue
        for neighbor, weight in global_graph[node]:
            # ignore low edge cells
            if neighbor in min_dist_low and min_dist_low[neighbor] == 1:
                continue
            if neighbor not in min_dist:
                min_dist[neighbor] = dist + weight
                heapq.heappush(pq, (dist + weight, neighbor))
            elif dist + weight < min_dist[neighbor]:
                min_dist[neighbor] = dist + weight
                heapq.heappush(pq, (dist + weight, neighbor))
    return min_dist


def setup_datasets(
    dem_filepath: str,
    fdr_filepath: str,
    fixed_fdr_filepath: str,
):
    dem_ds = open_dataset(dem_filepath)
    fdr_ds = open_dataset(fdr_filepath)
    dem_band = dem_ds.GetRasterBand(1)
    x_size = dem_band.XSize
    y_size = dem_band.YSize
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()
    fixed_fdr_ds = create_dataset(
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
        fdr_ds,
        fixed_fdr_ds,
    )


@njit
def create_labels_and_graph_process_tile(
    dem,
    fdr,
    tile_index,
    perimeter_cells_per_tile,
    tile_index_map,
    tile_labels,
    global_graph,
    global_perimeter_high_edges,
    global_perimeter_low_edges,
    global_interior_high_edges,
    global_interior_low_edges,
):
    # resolve_flats_tile removes the buffer region from the dem and flow_dirs when
    # creating the flat mask and labels
    to_higher, to_lower, labels, high_edges, low_edges = resolve_flats_tile(dem, fdr)
    edge_labels = TileEdgeCells(labels, tile_index)
    tile_row, tile_col = tile_index_map[tile_index]
    graph, perimeter_high_edges, perimeter_low_edges = construct_local_edge_graph(
        edge_labels,
        TileEdgeCells(to_higher, tile_index),
        TileEdgeCells(to_lower, tile_index),
        labels,
    )
    tile_labels[(tile_row, tile_col)] = edge_labels
    for j, edge in enumerate(graph):
        global_index = tile_index * perimeter_cells_per_tile + j
        global_graph[global_index] = edge
    for edge in perimeter_high_edges:
        global_perimeter_high_edges.append(edge)
    for edge in perimeter_low_edges:
        global_perimeter_low_edges.append(edge)
    if len(high_edges) > 0:
        global_interior_high_edges[tile_index] = high_edges
    if len(low_edges) > 0:
        global_interior_low_edges[tile_index] = low_edges
    return labels


@njit
def initalize_global_graph(num_tiles, perimeter_cells_per_tile):
    global_graph = List()
    empty_list_of_tuples = List()
    empty_list_of_tuples.append((-1, -1))
    empty_list_of_tuples.pop()
    for _ in range(num_tiles * perimeter_cells_per_tile):
        global_graph.append(empty_list_of_tuples)
    global_perimeter_high_edges = empty_list_of_tuples.copy()
    global_perimeter_low_edges = empty_list_of_tuples.copy()
    global_interior_high_edges = List()
    global_interior_low_edges = List()
    for _ in range(num_tiles):
        global_interior_high_edges.append(empty_list_of_tuples)
        global_interior_low_edges.append(empty_list_of_tuples)
    return (
        global_graph,
        global_perimeter_high_edges,
        global_perimeter_low_edges,
        global_interior_high_edges,
        global_interior_low_edges,
    )


def create_labels_and_graph(
    dem_band,
    fdr_band,
    chunk_size,
    working_directory,
    geotransform,
    projection,
    batch_size=8,
):
    # setup the numba typed lists, is there a better way?
    num_tiles = math.ceil(fdr_band.YSize / chunk_size) * math.ceil(
        fdr_band.XSize / chunk_size
    )
    perimeter_cells_per_tile = 4 * (chunk_size - 2) + 4

    (
        global_graph,
        global_perimeter_high_edges,
        global_perimeter_low_edges,
        global_interior_high_edges,
        global_interior_low_edges,
    ) = initalize_global_graph(num_tiles, perimeter_cells_per_tile)

    tile_index = 0

    num_tiles = math.ceil(fdr_band.YSize / chunk_size) * math.ceil(
        fdr_band.XSize / chunk_size
    )
    # map of tile index to tile row, col
    tile_index_map = Dict.empty(key_type=int64, value_type=Tuple((int64, int64)))
    dummy_tile_edge_cells = TileEdgeCells(np.zeros((1, 1), dtype=np.int32), 0)
    tile_labels = Dict()
    tile_labels[(0, 0)] = dummy_tile_edge_cells
    del tile_labels[(0, 0)]

    labels_tile_filepath = os.path.join(working_directory, "labels.tif")
    labels_tile_ds = create_dataset(
        labels_tile_filepath,
        0,
        gdal.GDT_Int32,
        dem_band.XSize,
        dem_band.YSize,
        geotransform,
        projection,
    )
    labels_tile_band = labels_tile_ds.GetRasterBand(1)

    for dem_tile in raster_chunker(dem_band, chunk_size, 1):
        # Here we use a 1 cell buffer, this differs from the paper.
        # This is done so that there are no 'uncertain' edges, only low and high edges.
        # This removes the need to handle uncertain edges and we don't need to save the
        # perimeter DEM elevation values since we no don't need to identify them later
        fdr_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
        fdr_tile.read(fdr_band)
        tile_index_map[tile_index] = (dem_tile.row, dem_tile.col)
        labels = create_labels_and_graph_process_tile(
            dem_tile.data,
            fdr_tile.data,
            tile_index,
            perimeter_cells_per_tile,
            tile_index_map,
            tile_labels,
            global_graph,
            global_perimeter_high_edges,
            global_perimeter_low_edges,
            global_interior_high_edges,
            global_interior_low_edges,
        )
        labels_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 0)
        labels_tile.from_numpy(labels)
        labels_tile.write(labels_tile_band)
        tile_index += 1

    labels_tile_band.FlushCache()
    labels_tile_ds.FlushCache()
    labels_tile_ds = None
    labels_tile_band = None
    return (
        tile_labels,
        global_graph,
        global_perimeter_high_edges,
        global_perimeter_low_edges,
        global_interior_high_edges,
        global_interior_low_edges,
    )


@njit
def complete_graph(tile_labels, rows, cols, global_graph):
    for i in range(rows - 1):
        for j in range(cols - 1):
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


@njit(parallel=True)
def create_flat_mask_process_batch(
    tile_indices,
    chunk_size,
    tile_index_map,
    tile_labels,
    global_interior_high_edges,
    global_interior_low_edges,
    min_dist_high,
    min_dist_low,
    fdr_batch,
    labels_list,
):
    batch_flat_masks = np.empty(
        (len(tile_indices), chunk_size, chunk_size), dtype=np.int32
    )
    for i in prange(len(labels_list)):
        labels = labels_list[i]
        fdr = fdr_batch[i]
        flat_mask = np.zeros_like(labels, dtype=np.int32)
        tile_index = tile_indices[i]
        tile_row, tile_col = tile_index_map[tile_index]
        edge_labels = tile_labels[(tile_row, tile_col)]
        interior_high_edges = global_interior_high_edges[tile_index]
        exterior_edges, max_high_loop, loop_index_map = get_tile_exterior_edges(
            min_dist_high, edge_labels
        )
        away_from_higher_tile(
            interior_high_edges,
            exterior_edges,
            loop_index_map,
            max_high_loop,
            labels,
            fdr,
            flat_mask,
        )
        interior_low_edges = global_interior_low_edges[tile_index]
        exterior_edges, max_low_loop, loop_index_map = get_tile_exterior_edges(
            min_dist_low, edge_labels
        )
        towards_lower_tile(
            interior_low_edges,
            exterior_edges,
            loop_index_map,
            max_low_loop,
            labels,
            fdr,
            flat_mask,
        )
        batch_flat_masks[i] = flat_mask
    return batch_flat_masks


def create_flat_mask(
    rows,
    cols,
    chunk_size,
    working_dir,
    dem_band,
    dem_ds,
    fdr_band,
    tile_labels,
    global_interior_high_edges,
    global_interior_low_edges,
    min_dist_high,
    min_dist_low,
    batch_size=9,
):
    labels_dataset = open_dataset(os.path.join(working_dir, "labels.tif"))
    labels_band = labels_dataset.GetRasterBand(1)
    tile_index = 0
    tile_index_map = Dict.empty(key_type=int64, value_type=Tuple((int64, int64)))
    labels_batch = []
    fdr_batch = []
    batch_tile_indices = []
    num_tiles = math.ceil(fdr_band.YSize / chunk_size) * math.ceil(
        fdr_band.XSize / chunk_size
    )
    flat_mask_dataset = create_dataset(
        os.path.join(working_dir, "flat_mask.tif"),
        0,
        gdal.GDT_Int32,
        dem_band.XSize,
        dem_band.YSize,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )
    flat_mask_band = flat_mask_dataset.GetRasterBand(1)
    for tile_row in range(rows):
        for tile_col in range(cols):
            fdr_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            fdr_tile.read(fdr_band)
            flat_mask_tile = RasterChunk(0, 0, chunk_size, 0)
            fdr_batch.append(fdr_tile.data)
            labels_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
            labels_tile.read(labels_band)
            labels = labels_tile.data
            batch_tile_indices.append(tile_index)
            tile_index_map[tile_index] = (tile_row, tile_col)
            tile_index += 1
            labels_batch.append(labels)
            if len(labels_batch) == batch_size or tile_index == num_tiles:
                batch_flat_masks = create_flat_mask_process_batch(
                    batch_tile_indices,
                    chunk_size,
                    tile_index_map,
                    tile_labels,
                    global_interior_high_edges,
                    global_interior_low_edges,
                    min_dist_high,
                    min_dist_low,
                    fdr_batch,
                    labels_batch,
                )
                for flat_mask, result_index in zip(
                    batch_flat_masks, batch_tile_indices
                ):
                    tile_row, tile_col = tile_index_map[result_index]
                    flat_mask_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
                    flat_mask_tile.from_numpy(flat_mask)
                    flat_mask_tile.write(flat_mask_band)
                labels_batch = []
                batch_tile_indices = []
                fdr_batch = []

    flat_mask_band.FlushCache()
    flat_mask_dataset.FlushCache()
    flat_mask_band = None
    flat_mask_dataset = None


def update_fdr(dem_band, fdr_band, fixed_fdr_band, working_dir, chunk_size):
    flat_mask_ds = open_dataset(os.path.join(working_dir, "flat_mask.tif"))
    flat_mask_band = flat_mask_ds.GetRasterBand(1)
    # update fdr using d8_masked_flow_dirs
    for fdr_tile in raster_chunker(fdr_band, chunk_size, 1):
        dem_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        dem_tile.read(dem_band)
        flat_mask_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        flat_mask_tile.read(flat_mask_band)
        d8_masked_flow_dirs(dem_tile.data, flat_mask_tile.data, fdr_tile.data)
        fdr_tile.write(fixed_fdr_band)
    fixed_fdr_band.FlushCache()


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
    # setup the input and output datasets
    (
        dem_ds,
        fdr_ds,
        fixed_fdr_ds,
    ) = setup_datasets(
        dem_filepath,
        fdr_filepath,
        output_filepath,
    )
    dem_band = dem_ds.GetRasterBand(1)
    fdr_band = fdr_ds.GetRasterBand(1)
    fixed_fdr_band = fixed_fdr_ds.GetRasterBand(1)
    num_tile_rows = math.ceil(dem_band.YSize / chunk_size)
    num_tile_cols = math.ceil(dem_band.XSize / chunk_size)

    # create labels and graph for each tile
    (
        tile_labels,
        global_graph,
        global_perimeter_high_edges,
        global_perimeter_low_edges,
        global_interior_high_edges,
        global_interior_low_edges,
    ) = create_labels_and_graph(
        dem_band,
        fdr_band,
        chunk_size,
        working_dir,
        dem_ds.GetGeoTransform(),
        dem_ds.GetProjection(),
    )

    # complete the graph by connecting the neighboring tiles
    complete_graph(tile_labels, num_tile_rows, num_tile_cols, global_graph)

    # solve the graph for the low and high terrain nodes
    min_dist_low = solve_low_graph(global_graph, global_perimeter_low_edges)
    min_dist_high = solve_high_graph(
        global_graph, global_perimeter_high_edges, min_dist_low
    )

    # create the flat mask for each tile
    create_flat_mask(
        num_tile_rows,
        num_tile_cols,
        chunk_size,
        working_dir,
        dem_band,
        dem_ds,
        fdr_band,
        tile_labels,
        global_interior_high_edges,
        global_interior_low_edges,
        min_dist_high,
        min_dist_low,
    )

    # update the flow direction raster
    update_fdr(dem_band, fdr_band, fixed_fdr_band, working_dir, chunk_size)

    if cleanup_working_dir:
        shutil.rmtree(working_dir)
    fixed_fdr_ds = None
    fdr_ds = None
    dem_ds = None
