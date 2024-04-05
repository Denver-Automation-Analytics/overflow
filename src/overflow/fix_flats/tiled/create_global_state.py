import math
import os
from osgeo import gdal
from overflow.util.raster import raster_chunker, RasterChunk, create_dataset
from overflow.fix_flats.tiled.state_manager import (
    GlobalState,
    Float32Perimeter,
    get_tile_perimeter,
)
from overflow.fix_flats.tiled.resolve_flats import compute_labels_and_gradients
from overflow.fix_flats.tiled.distance import compute_min_dist
from overflow.fix_flats.tiled.graphs import LocalGraph


def create_global_state(
    dem_band: gdal.Band,
    fdr_band: gdal.Band,
    labels_band: gdal.Band,
    chunk_size: int,
):
    tile_rows = math.ceil(dem_band.YSize / chunk_size)
    tile_cols = math.ceil(dem_band.XSize / chunk_size)
    perimeter_cells_per_tile = 4 * (chunk_size - 2) + 4
    global_state = GlobalState(tile_rows, tile_cols, perimeter_cells_per_tile)
    tile_index = 0

    for dem_tile in raster_chunker(dem_band, chunk_size, 1):
        fdr_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 1)
        fdr_tile.read(fdr_band)
        perimeter_elevations = get_tile_perimeter(dem_tile.data)
        global_state.elevations[tile_index] = Float32Perimeter(
            perimeter_elevations, tile_rows, tile_cols, tile_index
        )

        ## Thread Start
        to_higher, to_lower, labels, high_edges, low_edges = (
            compute_labels_and_gradients(dem_tile.data, fdr_tile.data)
        )
        min_dists = compute_min_dist(labels)
        to_higher_perimeter = get_tile_perimeter(to_higher)
        to_lower_perimeter = get_tile_perimeter(to_lower)
        labels_perimeter = get_tile_perimeter(labels)
        local_graph = LocalGraph(
            labels_perimeter,
            to_higher_perimeter,
            to_lower_perimeter,
            min_dists,
            tile_index,
        )
        ## Thread End

        global_state.graph.add(local_graph)
        global_state.interior_low_edges[tile_index] = low_edges
        global_state.interior_high_edges[tile_index] = high_edges
        tile_index += 1
        labels_tile = RasterChunk(dem_tile.row, dem_tile.col, chunk_size, 0)
        labels_tile.from_numpy(labels)
        labels_tile.write(labels_band)
    global_state.graph.join()
    labels_band.FlushCache()
    return global_state
