import numpy as np
import math
from osgeo import gdal
from numba import njit, prange, int64, uint8
from numba.typed import Dict
from numba.types import DictType
from numba.experimental import jitclass
from overflow.util.raster import (
    Side,
    Corner,
    raster_chunker,
    create_dataset,
    RasterChunk,
)
from overflow.util.perimeter import get_tile_perimeter, UInt8Perimeter, Int64Perimeter
from overflow.constants import (
    NEIGHBOR_OFFSETS,
    FLOW_DIRECTION_UNDEFINED,
    FLOW_DIRECTION_NODATA,
    FLOW_DIRECTION_SOUTH_EAST,
    FLOW_DIRECTION_NORTH_WEST,
    FLOW_DIRECTION_SOUTH_WEST,
    FLOW_DIRECTION_NORTH_EAST,
    FLOW_DIRECTION_EAST,
    FLOW_DIRECTION_NORTH,
    FLOW_DIRECTION_WEST,
    FLOW_DIRECTION_SOUTH,
    DEFAULT_CHUNK_SIZE,
)
from overflow.util.queue import Queue


@njit
def upstream_neighbor_generator(fdr: np.ndarray, row: int, col: int):
    rows, cols = fdr.shape
    for i, (d_row, d_col) in enumerate(NEIGHBOR_OFFSETS):
        n_row, n_col = row + d_row, col + d_col
        if 0 <= n_row < rows and 0 <= n_col < cols:
            n_direction = fdr[n_row, n_col]
            if (i + 4) % 8 == n_direction:
                yield n_row, n_col


@njit
def is_outlet(fdr: np.ndarray, row: int, col: int):
    if fdr[row, col] in [FLOW_DIRECTION_UNDEFINED, FLOW_DIRECTION_NODATA]:
        return False
    d_row, d_col = NEIGHBOR_OFFSETS[fdr[row, col]]
    n_row, n_col = row + d_row, col + d_col
    # it is an outlet if the next cell is outside the domain, is nodata, or undefined
    return (
        n_row < 0
        or n_row >= fdr.shape[0]
        or n_col < 0
        or n_col >= fdr.shape[1]
        or fdr[n_row, n_col]
        in [
            FLOW_DIRECTION_UNDEFINED,
            FLOW_DIRECTION_NODATA,
        ]
    )


@njit(parallel=True)
def label_watersheds(
    fdr: np.ndarray,
    drainage_points: dict,
    id_offset: int = 0,
    row_offset: int = 0,
    col_offset: int = 0,
) -> tuple[np.ndarray, int]:
    watersheds = np.zeros_like(fdr, dtype=np.int64)
    rows, cols = fdr.shape
    global_ids = np.zeros(rows * cols, dtype=np.int64)
    for row in prange(rows):  # pylint: disable=not-an-iterable
        for col in range(cols):
            if is_outlet(fdr, row, col):
                watershed_id = row * cols + col + 1
                queue = Queue([(row, col, watershed_id)])
                watersheds[row, col] = watershed_id
                global_ids[row * cols + col] = watershed_id

                while queue:
                    r, c, watershed_id = queue.pop()
                    if (r + row_offset, c + col_offset) in drainage_points:
                        # TODO, in the case of multiple drainage points,
                        # need to keep track of the downstream watershed id
                        # for each drainage point
                        watershed_id = r * c + c + 1
                        watersheds[r, c] = watershed_id
                        global_ids[r * c + c] = watershed_id
                    for n_row, n_col in upstream_neighbor_generator(fdr, r, c):
                        watersheds[n_row, n_col] = watershed_id
                        queue.push((n_row, n_col, watershed_id))

    global_ids = global_ids[global_ids > 0]
    unique_ids = np.unique(global_ids)
    id_mapping = dict(
        zip(unique_ids, range(1 + id_offset, len(unique_ids) + 1 + id_offset))
    )
    global_id = id_offset + len(unique_ids) + 1

    for i in prange(rows * cols):  # pylint: disable=not-an-iterable
        if watersheds.flat[i] > 0:
            watersheds.flat[i] = id_mapping[watersheds.flat[i]]

    for row, col in drainage_points:
        if 0 <= row - row_offset < rows and 0 <= col - col_offset < cols:
            drainage_points[(row, col)] = watersheds[row - row_offset, col - col_offset]

    return watersheds, global_id


@jitclass
class TileEdgeData:
    flow_direction: UInt8Perimeter
    watershed: Int64Perimeter

    def __init__(self, flow_direction: UInt8Perimeter, watershed: Int64Perimeter):
        self.flow_direction = flow_direction
        self.watershed = watershed


@jitclass(
    [
        ("flow_directions", uint8[:, :]),
        ("watersheds", int64[:, :]),
        ("graph", DictType(int64, int64)),
    ]
)
class GlobalState:
    flow_directions: np.ndarray
    watersheds: np.ndarray
    tile_rows: int
    tile_cols: int
    chunk_size: int
    graph: dict

    def __init__(self, tile_rows: int, tile_cols: int, chunk_size: int):
        perimeter_cells_per_tile = 2 * (chunk_size + chunk_size) - 4
        tile_count = tile_rows * tile_cols
        self.flow_directions = np.empty(
            (tile_count, perimeter_cells_per_tile), dtype=np.uint8
        )
        self.watersheds = np.empty(
            (tile_count, perimeter_cells_per_tile), dtype=np.int64
        )
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.chunk_size = chunk_size
        self.graph = Dict.empty(int64, int64)

    def _get_tile_index(self, row: int, col: int) -> int:
        return row * self.tile_cols + col

    def _get_directions_perimeter(self, tile_row: int, tile_col: int) -> UInt8Perimeter:
        tile_index = self._get_tile_index(tile_row, tile_col)
        return UInt8Perimeter(
            self.flow_directions[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )

    def _get_watersheds_perimeter(self, tile_row: int, tile_col: int) -> Int64Perimeter:
        tile_index = self._get_tile_index(tile_row, tile_col)
        return Int64Perimeter(
            self.watersheds[tile_index],
            self.chunk_size,
            self.chunk_size,
            tile_index,
        )

    def _get_tile_edge_data(self, tile_row: int, tile_col: int) -> TileEdgeData:
        return TileEdgeData(
            self._get_directions_perimeter(tile_row, tile_col),
            self._get_watersheds_perimeter(tile_row, tile_col),
        )

    def _handle_corner(
        self,
        tile_a: TileEdgeData,
        tile_b: TileEdgeData,
        corner_a: Corner,
    ):
        if corner_a == Corner.TOP_LEFT:
            direction_a = FLOW_DIRECTION_NORTH_WEST
            corner_b = Corner.BOTTOM_RIGHT
        if corner_a == Corner.TOP_RIGHT:
            direction_a = FLOW_DIRECTION_NORTH_EAST
            corner_b = Corner.BOTTOM_LEFT
        if corner_a == Corner.BOTTOM_LEFT:
            direction_a = FLOW_DIRECTION_SOUTH_WEST
            corner_b = Corner.TOP_RIGHT
        if corner_a == Corner.BOTTOM_RIGHT:
            direction_a = FLOW_DIRECTION_SOUTH_EAST
            corner_b = Corner.TOP_LEFT

        if tile_a.flow_direction.get_corner(corner_a) == direction_a:
            self.graph[tile_a.watershed.get_corner(corner_a)] = (
                tile_b.watershed.get_corner(corner_b)
            )

    def _handle_edge(
        self,
        tile_a: TileEdgeData,
        tile_b: TileEdgeData,
        side_a: Side,
    ):
        if side_a == Side.RIGHT:
            flow_across = FLOW_DIRECTION_EAST
            flow_up = FLOW_DIRECTION_NORTH_EAST
            flow_down = FLOW_DIRECTION_SOUTH_EAST
            side_b = Side.LEFT
        elif side_a == Side.BOTTOM:
            flow_across = FLOW_DIRECTION_SOUTH
            flow_up = FLOW_DIRECTION_SOUTH_WEST
            flow_down = FLOW_DIRECTION_SOUTH_EAST
            side_b = Side.TOP
        elif side_a == Side.LEFT:
            flow_across = FLOW_DIRECTION_WEST
            flow_up = FLOW_DIRECTION_NORTH_WEST
            flow_down = FLOW_DIRECTION_SOUTH_WEST
            side_b = Side.RIGHT
        elif side_a == Side.TOP:
            flow_across = FLOW_DIRECTION_NORTH
            flow_up = FLOW_DIRECTION_NORTH_WEST
            flow_down = FLOW_DIRECTION_NORTH_EAST
            side_b = Side.BOTTOM
        # check if tile a flows into tile b
        tile_a_flow_directions_right = tile_a.flow_direction.get_side(side_a)
        for i, direction in enumerate(tile_a_flow_directions_right):
            if direction == flow_across:
                self.graph[tile_a.watershed.get_side(side_a)[i]] = (
                    tile_b.watershed.get_side(side_b)[i]
                )
            if i < len(tile_a_flow_directions_right) - 1:
                if direction == flow_down:
                    self.graph[tile_a.watershed.get_side(side_a)[i]] = (
                        tile_b.watershed.get_side(side_b)[i + 1]
                    )
            if i > 0:
                if direction == flow_up:
                    self.graph[tile_a.watershed.get_side(side_a)[i]] = (
                        tile_b.watershed.get_side(side_b)[i - 1]
                    )

    def _join_adjacent_tiles(
        self,
        tile_a: TileEdgeData,
        tile_b: TileEdgeData,
        tile_c: TileEdgeData,
        tile_d: TileEdgeData,
    ):
        self._handle_corner(tile_a, tile_d, Corner.BOTTOM_RIGHT)
        self._handle_corner(tile_d, tile_a, Corner.TOP_LEFT)
        self._handle_corner(tile_b, tile_c, Corner.BOTTOM_LEFT)
        self._handle_corner(tile_c, tile_b, Corner.TOP_RIGHT)
        self._handle_edge(tile_a, tile_b, Side.RIGHT)
        self._handle_edge(tile_b, tile_a, Side.LEFT)
        self._handle_edge(tile_c, tile_d, Side.RIGHT)
        self._handle_edge(tile_d, tile_c, Side.LEFT)
        self._handle_edge(tile_a, tile_c, Side.BOTTOM)
        self._handle_edge(tile_c, tile_a, Side.TOP)
        self._handle_edge(tile_b, tile_d, Side.BOTTOM)
        self._handle_edge(tile_d, tile_b, Side.TOP)

    def complete_graph(self):
        for tile_row in range(self.tile_rows - 1):
            for tile_col in range(self.tile_cols - 1):
                tile_a = self._get_tile_edge_data(tile_row, tile_col)
                tile_b = self._get_tile_edge_data(tile_row, tile_col + 1)
                tile_c = self._get_tile_edge_data(tile_row + 1, tile_col)
                tile_d = self._get_tile_edge_data(tile_row + 1, tile_col + 1)
                self._join_adjacent_tiles(tile_a, tile_b, tile_c, tile_d)


@njit
def apply_watershed_map(labels: np.ndarray, global_graph: dict, drainage_points: dict):
    # dp labels are the set of values in the drainage points
    drainage_point_labels = set(drainage_points.values())

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            current_label = labels[row, col]
            if current_label not in drainage_point_labels:
                # walk down the graph until we find a node that is in the drainage points
                # or we reach the end of the graph (i.e. the node is not in the graph)
                while (
                    current_label in global_graph
                    and current_label not in drainage_point_labels
                ):
                    current_label = global_graph[current_label]
                labels[row, col] = current_label


def label_watersheds_tiled(
    fdr_filepath: str,
    drainage_points: dict,
    output_filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    fdr_ds = gdal.Open(fdr_filepath)
    fdr_band = fdr_ds.GetRasterBand(1)

    labels_ds = create_dataset(
        output_filepath,
        0,
        gdal.GDT_Int64,
        fdr_band.XSize,
        fdr_band.YSize,
        fdr_ds.GetGeoTransform(),
        fdr_ds.GetProjection(),
    )
    labels_band = labels_ds.GetRasterBand(1)

    global_id = 0
    tile_rows = math.ceil(fdr_band.YSize / chunk_size)
    tile_cols = math.ceil(fdr_band.XSize / chunk_size)
    global_state = GlobalState(tile_rows, tile_cols, chunk_size)
    tile_index = 0
    for fdr_tile in raster_chunker(fdr_band, chunk_size, 0):
        tile_row = fdr_tile.row
        tile_col = fdr_tile.col
        row_offset = tile_row * chunk_size
        col_offset = tile_col * chunk_size
        watersheds, next_id = label_watersheds(
            fdr_tile.data, drainage_points, global_id, row_offset, col_offset
        )
        global_id = next_id
        labels_tile = RasterChunk(tile_row, tile_col, chunk_size, 0)
        labels_tile.from_numpy(watersheds)
        labels_tile.write(labels_band)
        fdr_perimeter = get_tile_perimeter(fdr_tile.data)
        watersheds_perimeter = get_tile_perimeter(watersheds)
        global_state.flow_directions[tile_index] = fdr_perimeter
        global_state.watersheds[tile_index] = watersheds_perimeter
        tile_index += 1

    labels_band.FlushCache()
    labels_ds.FlushCache()
    fdr_ds = None

    global_state.complete_graph()

    # iterate over labels now using map to assign basin to drainage points or outlets
    for labels_tile in raster_chunker(labels_band, chunk_size, 0):
        apply_watershed_map(labels_tile.data, global_state.graph, drainage_points)
        labels_tile.write(labels_band)
    labels_band.FlushCache()
    labels_ds = None
