from numba import njit, prange
import numpy as np
from overflow.util.numba_datastructures import Queue
from overflow.util.raster import neighbor_generator
from overflow.fix_flats.tiled.state_manager import Int64Perimeter, get_tile_perimeter


@njit(parallel=True)
def compute_min_dist(labels_array):
    labels_perimeter = Int64Perimeter(
        get_tile_perimeter(labels_array),
        labels_array.shape[0],
        labels_array.shape[1],
        0,
    )
    perimeter_count = labels_perimeter.size()
    min_dist = np.zeros((perimeter_count, perimeter_count), dtype=np.int64)
    for i in prange(perimeter_count):
        label = labels_perimeter.data[i]
        if label == 0:
            continue
        visited = set()
        dist = 0
        from_row, from_col = labels_perimeter.get_row_col(i)
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
                j = labels_perimeter.get_index(row, col)
                if min_dist[i][j] > 0:
                    continue
                min_dist[i][j] = dist
            for neighbor_row, neighbor_col in neighbor_generator(
                row, col, labels_array.shape[0], labels_array.shape[1]
            ):
                if labels_array[neighbor_row, neighbor_col] == label:
                    queue.push((neighbor_row, neighbor_col))

    return min_dist
