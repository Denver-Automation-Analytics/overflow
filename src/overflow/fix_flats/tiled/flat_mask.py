from numba.typed import List
from numba import njit
import numpy as np
from overflow.util.raster import raster_chunker, RasterChunk
from overflow.fix_flats.tiled.state_manager import GlobalState, Int64Perimeter
from overflow.util.numba_datastructures import Queue
from overflow.constants import FLOW_DIRECTION_UNDEFINED
from overflow.util.raster import neighbor_generator


def get_edge_exterior_distances(dist_to_edge_perimeter: Int64Perimeter):
    exterior_edges = List()
    max_loop = 0
    loop_index_map = {}
    index = 0
    for i, dist in enumerate(dist_to_edge_perimeter.data):
        if dist <= 1:
            continue
        row, col = dist_to_edge_perimeter.get_row_col(i)
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
    dist_to_high_edge_perimeter,
    labels,
    fdr,
    flat_mask,
):
    exterior_high_edges, max_high_loop, loop_index_map = get_edge_exterior_distances(
        dist_to_high_edge_perimeter
    )
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
    dist_to_low_edge_perimeter,
    labels,
    fdr,
    flat_mask,
):
    exterior_low_edges, max_low_loop, loop_index_map = get_edge_exterior_distances(
        dist_to_low_edge_perimeter
    )
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


def create_flat_mask(
    chunk_size,
    flat_mask_band,
    labels_band,
    fdr_band,
    global_state,
    dist_to_high_edge_tiles,
    dist_to_low_edge_tiles,
):
    tile_index = 0
    for fdr_tile in raster_chunker(fdr_band, chunk_size, 0):
        labels_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 0)
        labels_tile.read(labels_band)
        flat_mask = np.zeros_like(fdr_tile.data, dtype=np.int64)

        # Thread Start
        dist_to_high_edge_perimeter = Int64Perimeter(
            dist_to_high_edge_tiles[tile_index],
            global_state.tile_rows,
            global_state.tile_cols,
            tile_index,
        )
        dist_to_low_edge_perimeter = Int64Perimeter(
            dist_to_low_edge_tiles[tile_index],
            global_state.tile_rows,
            global_state.tile_cols,
            tile_index,
        )
        away_from_higher_tile(
            global_state.interior_high_edges[tile_index],
            dist_to_high_edge_perimeter,
            labels_tile.data,
            fdr_tile.data,
            flat_mask,
        )
        towards_lower_tile(
            global_state.interior_low_edges[tile_index],
            dist_to_low_edge_perimeter,
            labels_tile.data,
            fdr_tile.data,
            flat_mask,
        )
        # Thread End

        flat_mask_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 0)
        flat_mask_tile.from_numpy(flat_mask)
        flat_mask_tile.write(flat_mask_band)
        tile_index += 1
    flat_mask_band.FlushCache()
