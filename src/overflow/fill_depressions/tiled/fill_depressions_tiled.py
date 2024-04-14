import tempfile
import os
import shutil
import numpy as np
from numba import njit
from numba.experimental import jitclass
from osgeo import gdal
from overflow.util.raster import open_dataset, create_dataset
from overflow.fill_depressions.core import priority_flood_tile, make_sides
from overflow.util.raster import Side, Corner, RasterChunk, raster_chunker


@jitclass
class GlobalGraph:
    edges


@jitclass
class GlobalState:
    graph: GlobalGraph


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


class Coordinate:
    row: np.int32
    col: np.int32

    def __init__(self, row, col):
        self.row = row
        self.col = col


class DemLabelArr:
    labels: np.ndarray
    dem: np.ndarray


class DemLabel:
    label: int
    dem: float


LABELS_FILENAME = "labels.tif"


def setup_datasets(input_path, output_path, working_dir):
    dem_ds = open_dataset(input_path)
    input_band = dem_ds.GetRasterBand(1)
    no_data_value = input_band.GetNoDataValue()
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


def depression_fill(input_path, output_path, chunk_size, working_dir):
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
    n_chunks_row, n_chunks_col = calculate_chunks(input_band, chunk_size)
    global_spillover_graph = {}
    label_offset = 0
    for tile_index, tile_dem in enumerate(raster_chunker(input_band, chunk_size)):
        sides = create_tile_side_flags(tile_dem, n_chunks_row, n_chunks_col)
        labels, graph, label_offset = priority_flood_tile(
            tile_dem, sides, no_data_value, label_offset
        )
        global_spillover_graph.update(graph)
        labels_tile = RasterChunk(tile_dem.row, tile_dem.col, chunk_size, 0)
        labels_tile.from_numpy(labels)
        labels_tile.write(labels_band)
    global_spillover_graph = connect_tile_edges_and_corners(
        n_chunks_row, n_chunks_col, input_band, chunk_size, global_spillover_graph
    )
    elevation_graph = priority_flood_graph(global_spillover_graph)
    for tile_index, tile_dem in enumerate(raster_chunker(input_band, chunk_size)):
        tile_name = get_tile_labels_file_name(tile_index)
        tile_labels = np.fromfile(tile_name, dtype=int)
        raise_elevation(output_band, tile_labels, tile_dem, elevation_graph)
    if cleanup_working_dir:
        shutil.rmtree(working_dir)


def connect_tile_edges_and_corners(
    n_tiles_row, n_tiles_col, input_band, chunk_size, init_graph
):
    """Combine the spillover graphs of all tiles into a single global graph."""
    spillover_graph = init_graph
    tile_indexs_iter = walk_adjacent_tiles(n_tiles_row, n_tiles_col)
    for top_l_i, top_r_i, bot_l_i, bot_r_i in tile_indexs_iter:
        # + - - + - - +
        # |  A  |  B  |
        # + - - * - - +
        # |  C  |  D  |
        # + - - + - - +
        tile_a = get_tile_labels_and_dem(top_l_i, n_tiles_col, input_band, chunk_size)
        tile_b = get_tile_labels_and_dem(top_r_i, n_tiles_col, input_band, chunk_size)
        tile_c = get_tile_labels_and_dem(bot_l_i, n_tiles_col, input_band, chunk_size)
        tile_d = get_tile_labels_and_dem(bot_r_i, n_tiles_col, input_band, chunk_size)
        # Connect A-B, A-C, D-B, D-C edges
        combine_edge(tile_a, Side.RIGHT, tile_b, Side.LEFT, spillover_graph)
        combine_edge(tile_a, Side.BOTTOM, tile_c, Side.TOP, spillover_graph)
        combine_edge(tile_d, Side.TOP, tile_b, Side.BOTTOM, spillover_graph)
        combine_edge(tile_d, Side.LEFT, tile_c, Side.RIGHT, spillover_graph)
        # Connect A-D and B-C corners
        combine_corner(
            tile_a, Corner.BOTTOM_RIGHT, tile_d, Corner.TOP_LEFT, spillover_graph
        )
        combine_corner(
            tile_b, Corner.BOTTOM_LEFT, tile_c, Corner.TOP_RIGHT, spillover_graph
        )
    return spillover_graph


def walk_adjacent_tiles(n_rows, n_cols):
    """Returns (top_left, top_right, bottom_left, bottom_right) adjacent grid indicies"""
    for row_index in range(n_rows - 1):
        for col_index in range(n_cols - 1):
            yield (
                (row_index, col_index),
                (row_index, col_index + 1),
                (row_index + 1, col_index),
                (row_index + 1, col_index + 1),
            )


def combine_edge(
    tile_a: DemLabelArr,
    side_a: Side,
    tile_b: DemLabelArr,
    side_b: Side,
    spillover_graph,
):
    """Combine two tiles by joining their edges."""
    edge_a = get_array_edge(tile_a.dem, side_a)
    labels_a = get_array_edge(tile_a, side_a)
    edge_b = get_array_edge(tile_b, side_b)
    labels_b = get_array_edge(tile_b, side_b)
    handle_edge(edge_a, labels_a, edge_b, labels_b, spillover_graph)


def combine_corner(
    tile_a: DemLabelArr,
    corner_a: Corner,
    tile_b: DemLabelArr,
    corner_b: Corner,
    spillover_graph,
):
    """Combine two tiles by joining their corners."""
    a = get_array_corner(tile_a, corner_a)
    b = get_array_corner(tile_b, corner_b)
    handle_corner(a.dem, a.label, b.dem, b.label, spillover_graph)


def get_tile_labels_and_dem(tile_location, n_cols, band, chunk_size):
    chunk_index = get_chunk_index_from_location(tile_location, n_cols)
    labels = np.fromfile(get_tile_labels_file_name(chunk_index), dtype=float)
    dem = read_tile(band, chunk_size, tile_location[0], tile_location[1])
    return DemLabelArr(labels, dem)


def get_array_corner(arr: DemLabelArr, corner: Corner):
    """Returns the value of the corner of the array specified by the corner enum"""
    if corner == Corner.TOP_LEFT:
        return DemLabel(arr.dem[0, 0], arr.labels[0, 0])
    if corner == Corner.TOP_RIGHT:
        return DemLabel(arr.dem[0, -1], arr.labels[0, -1])
    if corner == Corner.BOTTOM_RIGHT:
        return DemLabel(arr.dem[-1, -1], arr.labels[-1, -1])
    if corner == Corner.BOTTOM_LEFT:
        return DemLabel(arr.dem[-1, 0], arr.labels[-1, 0])


def get_array_edge(arr: np.ndarray, side: Side):
    """Returns the edge vector of the array specified by the side enum"""
    if side == Side.TOP:
        return arr[0, :]
    if side == Side.RIGHT:
        return arr[:, -1]
    if side == Side.BOTTOM:
        return arr[-1, :]
    if side == Side.LEFT:
        return arr[:, 0]


def get_tile_labels_file_name(tile_index):
    return f"tile-labels-{tile_index}.dat"


def get_chunk_index_from_location(location, n_cols):
    row_index, col_index = location
    return row_index * n_cols + col_index


def get_vertical_edges(n_rows, n_cols):
    """Returns (left, right) pairs of tile locations for vertical edges"""
    for row_index in range(n_rows):
        for col_index in range(n_cols - 1):
            yield ((row_index, col_index), (row_index, col_index + 1))


def get_horizontal_edges(n_rows, n_cols):
    """Returns (top, bottom) pairs of tile locations for horizontal edges"""
    for row_index in range(n_rows - 1):
        for col_index in range(n_cols):
            yield ((row_index, col_index), (row_index + 1, col_index))


def calculate_chunks(band: gdal.Band, chunk_size: int):
    n_chunks_row = (band.YSize + chunk_size - 1) // chunk_size
    n_chunks_col = (band.XSize + chunk_size - 1) // chunk_size
    return n_chunks_row, n_chunks_col


def create_tile_side_flags(tile: RasterChunk, n_chunks_row: int, n_chunks_col: int):
    top = tile.row == 0
    right = tile.col == n_chunks_col - 1
    bottom = tile.row == n_chunks_row - 1
    left = tile.col == 0
    return make_sides(top, right, bottom, left)


def priority_flood_graph(mastergraph: list[dict[int, float]]):
    """
    Modified priority flood algorithm for graph

    modifies mastergraph and elevation_graph in place

    Algorithm 2 from priority-flood (r. barnes, 2015)
    https://arxiv.org/pdf/1511.04463.pdf
    """
    open_heap: list[tuple[float, int]] = [(float("-inf"), 1)]
    seen: set[int] = set()
    elevation_graph: dict[int, float] = {}

    while open_heap:
        cell_elevation, vertex_label = heappop(open_heap)
        if vertex_label in seen:
            continue
        seen.add(vertex_label)
        elevation_graph[vertex_label] = cell_elevation

        for node_label, node_elevation in mastergraph[vertex_label].items():
            if node_label in seen:
                continue
            max_height = max(cell_elevation, node_elevation)
            new_node = (max_height, node_label)
            heappush(open_heap, new_node)
    return elevation_graph


def raise_elevation(
    output_band: gdal.Band,
    tile_labels: np.ndarray,
    tile_dem: RasterChunk,
    global_labels,
):
    """Raise elevation of dem to match global labels"""
    n_row, n_col = tile_dem.shape
    for row in range(n_row):
        for col in range(n_col):
            height = tile_dem[row, col]
            label = tile_labels[row, col]
            tile_dem[row, col] = max(global_labels[label], height)
    tile_dem.write(output_band)
