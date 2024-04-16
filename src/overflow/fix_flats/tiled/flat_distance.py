from numba import njit, prange
import numpy as np
from overflow.util.queue import RowColQueue as Queue
from overflow.util.raster import neighbor_generator
from overflow.util.perimeter import Int64Perimeter, get_tile_perimeter


@njit(parallel=True)
def compute_min_dist(labels_array: np.ndarray) -> np.ndarray:
    """
    Compute the minimum distances between perimeter cells with the same label.

    This function takes a 2D array of labels and computes the minimum distances between perimeter
    cells that have the same label. It uses a breadth-first search (BFS) algorithm to traverse
    the cells with the same label and calculate the distances.

    Args:
        labels_array (np.ndarray): A 2D array of labels, where each cell is assigned a label value.
            Cells with the same label belong to the same flat region.

    Returns:
        np.ndarray: A 2D array of shape (perimeter_count, perimeter_count) representing the
            minimum distances between perimeter cells with the same label. The value at index
            (i, j) represents the minimum distance between perimeter cell i and perimeter cell j.
    """
    labels_perimeter = Int64Perimeter(
        get_tile_perimeter(labels_array),
        labels_array.shape[0],
        labels_array.shape[1],
        0,
    )
    perimeter_count = labels_perimeter.size()
    min_dist = np.zeros((perimeter_count, perimeter_count), dtype=np.int64)
    # Disabling pylint warning, see https://github.com/PyCQA/pylint/issues/2910
    for i in prange(perimeter_count):  # pylint: disable=not-an-iterable
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
            record = queue.pop()
            row = record["row"]
            col = record["col"]
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
