import tempfile
import os
import shutil
import math
from heapq import heappop, heappush
import numpy as np
from numba import njit
from numba.typed import Dict  # pylint: disable=no-name-in-module
from numba.types import DictType, int64, float32, UniTuple
from numba.experimental import jitclass
from osgeo import gdal
from overflow.util.raster import open_dataset, create_dataset
from overflow.fill_depressions.core import priority_flood_tile, make_sides
from overflow.util.raster import Side, Corner, RasterChunk, raster_chunker
from overflow.util.perimeter import (
    get_tile_perimeter,
    Int64Perimeter,
    Float32Perimeter,
)


@jitclass([("edges", DictType(UniTuple(int64, 2), float32))])
class GlobalGraph:
    edges: dict

    def __init__(self):
        self.edges = Dict.empty((int64, int64), float32)

    def update(self, local_graph: dict[tuple[int, int], float]) -> None:
        self.edges.update(local_graph)


@jitclass(
    [
        ("label_perimeters", int64[:, :]),
        ("elevation_perimeters", float32[:, :]),
    ]
)
class GlobalState:
    graph: GlobalGraph
    label_perimeters: np.ndarray
    elevation_perimeters: np.ndarray
    num_rows: int
    num_cols: int
    chunk_size: int
    no_data: float

    def __init__(self, num_rows: int, num_cols: int, chunk_size: int, no_data: float):
        self.graph = GlobalGraph()
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.chunk_size = chunk_size
        self.no_data = no_data
        tile_count = num_rows * num_cols
        perimeter_cell_count = 4 * chunk_size - 4

        self.label_perimeters = np.empty(
            (tile_count, perimeter_cell_count), dtype=np.int64
        )
        self.elevation_perimeters = np.empty(
            (tile_count, perimeter_cell_count), dtype=np.float32
        )

    def update_graph(self, local_graph: dict[tuple[int, int], float]) -> None:
        self.graph.update(local_graph)

    def row_col_to_tile_index(self, row: int, col: int) -> int:
        return row * self.num_cols + col


@njit
def handle_edge(
    dem_a: np.ndarray,
    labels_a: np.ndarray,
    dem_b: np.ndarray,
    labels_b: np.ndarray,
    graph: dict[tuple[int, int], float],
    no_data: float = -9999,
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
                (min(labels_a[i], labels_b[ni]), max(labels_a[i], labels_b[ni]))
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
    no_data: float = -9999,
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
    label_pair = tuple((min(label_a, label_b), max(label_a, label_b)))

    # Update the graph
    if label_pair not in graph or e < graph[label_pair]:
        graph[label_pair] = e


class PerimeterData:
    labels: Int64Perimeter
    elevations: Float32Perimeter

    def __init__(self, global_state: GlobalState, tile_index: int):
        self.labels = Int64Perimeter(
            global_state.label_perimeters[tile_index],
            global_state.chunk_size,
            global_state.chunk_size,
            tile_index,
        )
        self.elevations = Float32Perimeter(
            global_state.elevation_perimeters[tile_index],
            global_state.chunk_size,
            global_state.chunk_size,
            tile_index,
        )


LABELS_FILENAME = "labels.tif"


def setup_datasets(input_path, output_path, working_dir):
    dem_ds = open_dataset(input_path)
    input_band = dem_ds.GetRasterBand(1)
    no_data_value = input_band.GetNoDataValue()
    if no_data_value is None:
        raise ValueError("Input raster must have a no data value")
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()
    output_ds = create_dataset(
        output_path,
        no_data_value,
        input_band.DataType,
        input_band.XSize,
        input_band.YSize,
        geotransform,
        projection,
    )
    labels_ds = create_dataset(
        os.path.join(working_dir, LABELS_FILENAME),
        0,
        gdal.GDT_Int64,
        input_band.XSize,
        input_band.YSize,
        geotransform,
        projection,
    )
    return dem_ds, output_ds, labels_ds, no_data_value


def fill_depressions_tiled(input_path, output_path, chunk_size, working_dir):
    """
    Fill depressions in a digital elevation model (DEM).
    Algorithm 3 from Parallel Priority-Flood (R. Barnes, 2016)
    """
    cleanup_working_dir = False
    if working_dir is None:
        working_dir = tempfile.mkdtemp()
        cleanup_working_dir = True
    dem_ds, output_ds, labels_ds, no_data_value = setup_datasets(
        input_path, output_path, working_dir
    )
    input_band = dem_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)
    labels_band = labels_ds.GetRasterBand(1)
    output_band = output_ds.GetRasterBand(1)
    n_chunks_row = math.ceil(input_band.YSize / chunk_size)
    n_chunks_col = math.ceil(input_band.XSize / chunk_size)
    global_state = GlobalState(n_chunks_row, n_chunks_col, chunk_size, no_data_value)
    label_offset = 0
    for tile_index, tile_dem in enumerate(raster_chunker(input_band, chunk_size)):
        sides = create_tile_side_flags(tile_dem, n_chunks_row, n_chunks_col)
        labels, graph, label_offset = priority_flood_tile(
            tile_dem.data, sides, no_data_value, label_offset
        )
        global_state.update_graph(graph)
        labels_tile = RasterChunk(tile_dem.row, tile_dem.col, chunk_size, 0)
        labels_tile.from_numpy(labels)
        labels_tile.write(labels_band)
        labels_band.FlushCache()
        tile_dem.write(output_band)
        output_band.FlushCache()
        global_state.label_perimeters[tile_index] = get_tile_perimeter(labels)
        global_state.elevation_perimeters[tile_index] = get_tile_perimeter(
            tile_dem.data
        )

    labels_band.FlushCache()
    labels_ds.FlushCache()
    output_band.FlushCache()
    output_ds.FlushCache()
    connect_tile_edges_and_corners(global_state)
    graph = build_graph(global_state, label_offset)
    elevation_graph = priority_flood_graph(graph)
    for tile_index, tile_dem in enumerate(raster_chunker(output_band, chunk_size)):
        tile_labels = RasterChunk(tile_dem.row, tile_dem.col, chunk_size, 0)
        tile_labels.read(labels_band)
        raise_elevation(output_band, tile_labels.data, tile_dem, elevation_graph)
    if cleanup_working_dir:
        shutil.rmtree(working_dir)
    output_band.FlushCache()
    output_ds.FlushCache()
    output_ds = None
    labels_ds = None


def connect_tile_edges_and_corners(global_state):
    """Combine the spillover graphs of all tiles into a single global graph."""
    for row_index in range(global_state.num_rows - 1):
        for col_index in range(global_state.num_cols - 1):
            # + - - + - - +
            # |  A  |  B  |
            # + - - * - - +
            # |  C  |  D  |
            # + - - + - - +
            tile_index_a = global_state.row_col_to_tile_index(row_index, col_index)
            tile_index_b = global_state.row_col_to_tile_index(row_index, col_index + 1)
            tile_index_c = global_state.row_col_to_tile_index(row_index + 1, col_index)
            tile_index_d = global_state.row_col_to_tile_index(
                row_index + 1, col_index + 1
            )
            perimeter_data_a = PerimeterData(global_state, tile_index_a)
            perimeter_data_b = PerimeterData(global_state, tile_index_b)
            perimeter_data_c = PerimeterData(global_state, tile_index_c)
            perimeter_data_d = PerimeterData(global_state, tile_index_d)

            # Connect A-B, A-C, D-B, D-C edges
            combine_edge(
                perimeter_data_a, Side.RIGHT, perimeter_data_b, Side.LEFT, global_state
            )
            combine_edge(
                perimeter_data_a, Side.BOTTOM, perimeter_data_c, Side.TOP, global_state
            )
            combine_edge(
                perimeter_data_d, Side.TOP, perimeter_data_b, Side.BOTTOM, global_state
            )
            combine_edge(
                perimeter_data_d, Side.LEFT, perimeter_data_c, Side.RIGHT, global_state
            )
            # Connect A-D and B-C corners
            combine_corner(
                perimeter_data_a,
                Corner.BOTTOM_RIGHT,
                perimeter_data_d,
                Corner.TOP_LEFT,
                global_state,
            )
            combine_corner(
                perimeter_data_b,
                Corner.BOTTOM_LEFT,
                perimeter_data_c,
                Corner.TOP_RIGHT,
                global_state,
            )


def combine_edge(
    perimeter_data_a: PerimeterData,
    side_a: Side,
    perimeter_data_b: PerimeterData,
    side_b: Side,
    global_state: GlobalState,
):
    """Combine two tiles by joining their edges."""
    dem_a = perimeter_data_a.elevations.get_side(side_a)
    labels_a = perimeter_data_a.labels.get_side(side_a)
    dem_b = perimeter_data_b.elevations.get_side(side_b)
    labels_b = perimeter_data_b.labels.get_side(side_b)
    handle_edge(dem_a, labels_a, dem_b, labels_b, global_state.graph.edges)


def combine_corner(
    perimeter_data_a: PerimeterData,
    corner_a: Corner,
    perimeter_data_b: PerimeterData,
    corner_b: Corner,
    global_state: GlobalState,
):
    """Combine two tiles by joining their corners."""
    dem_a = perimeter_data_a.elevations.get_corner(corner_a)
    labels_a = perimeter_data_a.labels.get_corner(corner_a)
    dem_b = perimeter_data_b.elevations.get_corner(corner_b)
    labels_b = perimeter_data_b.labels.get_corner(corner_b)
    handle_corner(dem_a, labels_a, dem_b, labels_b, global_state.graph.edges)


def create_tile_side_flags(tile: RasterChunk, n_chunks_row: int, n_chunks_col: int):
    top = tile.row == 0
    right = tile.col == n_chunks_col - 1
    bottom = tile.row == n_chunks_row - 1
    left = tile.col == 0
    return make_sides(top, right, bottom, left)


def build_graph(global_state: GlobalState, max_label) -> list:
    graph = [[] for _ in range(max_label)]
    for label_pair in global_state.graph.edges:
        graph[label_pair[0]].append(
            (label_pair[1], global_state.graph.edges[label_pair])
        )
        graph[label_pair[1]].append(
            (label_pair[0], global_state.graph.edges[label_pair])
        )
    return graph


def priority_flood_graph(graph: list[list[tuple[int, float]]]):
    """
    Modified priority flood algorithm for graph

    modifies mastergraph and elevation_graph in place

    Algorithm 2 from priority-flood (r. barnes, 2015)
    https://arxiv.org/pdf/1511.04463.pdf
    """
    open_heap = [(float("-inf"), 1)]
    closed = np.zeros(len(graph), dtype=bool)
    graph_elev = np.zeros(len(graph), dtype=float)

    while open_heap:
        elev, node = heappop(open_heap)
        if closed[node]:
            continue
        closed[node] = True
        graph_elev[node] = elev
        for neighbor, neighbor_elev in graph[node]:
            if not closed[neighbor]:
                max_height = max(elev, neighbor_elev)
                new_node = (max_height, neighbor)
                heappush(open_heap, new_node)

    return graph_elev

    # open_heap: list[tuple[float, int]] = [(float("-inf"), 1)]
    # pit_queue: list[tuple[float, int]] = []
    # closed = np.zeros(len(graph), dtype=bool)
    # graph_elev = np.zeros(len(graph), dtype=float)


#
# while open_heap or pit_queue:
#     if pit_queue:
#         elev, node = pit_queue.pop(0)
#     else:
#         elev, node = heappop(open_heap)
#     if closed[node]:
#         continue
#     closed[node] = True
#     graph_elev[node] = elev
#     for neighbor, neighbor_elev in graph[node]:
#         if not closed[neighbor]:
#             if neighbor_elev <= elev:
#                 heappush(open_heap, (elev, neighbor))
#             else:
#                 pit_queue.append((neighbor_elev, neighbor))
#
# return graph_elev


def raise_elevation(
    output_band: gdal.Band,
    tile_labels: np.ndarray,
    tile_dem: RasterChunk,
    global_labels,
):
    """Raise elevation of dem to match global labels"""
    n_row, n_col = tile_dem.data.shape
    for row in range(n_row):
        for col in range(n_col):
            height = tile_dem.data[row, col]
            label = tile_labels[row, col]
            tile_dem.data[row, col] = max(global_labels[label], height)
    tile_dem.write(output_band)
