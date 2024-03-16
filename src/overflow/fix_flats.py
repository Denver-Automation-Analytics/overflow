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


def label_flats(
    dem: np.ndarray, labels: np.ndarray, new_label: int, flat_row: int, flat_col: int
) -> None:
    """Algorithm 4 LabelFlats: This flood-fill function gives all the cells of a flat a common label,
    as described by §2.2. https://rbarnes.org/sci/2014_flats.pdf
    Upon entry:
    (1) DEM contains the elevations of every cell or a value NoData for cells not part of the DEM.
    (2) Labels has the same dimensions as DEM.
    (3) c belongs to the flat which is to be labeled.
    (4) L is a unique label which has not been previously applied to a flat.
    (5) Labels has been initialized to zero prior to the first call to this function.
    (6) Labels has values greater than or equal to 1 for each processed cell which is in a flat.
    Each flat's cells bear a label unique to that flat.
    At exit:
    (1) c and every cell reachable from c by passing over only cells of the same elevation as c
    (all the cells in the flat to which c belongs) is marked as L in Labels.
    (2) Labels has been updated to reflect the new labels which have been applied.

    Args:
        dem (np.ndarray): The digital elevation model
        labels (np.ndarray): The labels array
        new_label (int): The new label to apply to the flat
        flat_row (int): The initial row of the flat cell
        flat_col (int): The initial column of the flat cell
    """
    # create FIFO queue for to_be_filled
    to_be_filled = [(flat_row, flat_col)]
    elev = dem[flat_row, flat_col]
    while to_be_filled:
        row, col = to_be_filled.pop(0)
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
            to_be_filled.append((neighbor_row, neighbor_col))
