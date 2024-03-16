import numpy as np
from .constants import NEIGHBOR_OFFSETS, FLOW_DIRECTION_NODATA, FLOW_DIRECTION_UNDEFINED


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
        for d_row, d_col in NEIGHBOR_OFFSETS:
            neighbor_row = row + d_row
            neighbor_col = col + d_col
            # Check if the neighbor is not within the bounds of the DEM
            not_in_bounds = (
                neighbor_row < 0
                or neighbor_row >= fdr.shape[0]
                or neighbor_col < 0
                or neighbor_col >= fdr.shape[1]
            )
            if not_in_bounds:
                continue
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
