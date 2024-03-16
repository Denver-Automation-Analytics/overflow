import numpy as np
from numba import njit
from .constants import NEIGHBOR_OFFSETS, FLOW_DIRECTION_NODATA, FLOW_DIRECTION_UNDEFINED


@njit
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


@njit
def label_flats(
    dem: np.ndarray, labels: np.ndarray, new_label: int, flat_row: int, flat_col: int
) -> None:
    """Algorithm 4 LabelFlats: This flood-fill function gives all the cells of a flat a common label,
    as described by §2.2. https://rbarnes.org/sci/2014_flats.pdf
    Upon entry:
    (1) dem contains the elevations of every cell or a value NoData for cells not part of the DEM.
    (2) Labels has the same dimensions as DEM.
    (3) flat_row, flat_col belongs to the flat which is to be labeled.
    (4) new_label is a unique label which has not been previously applied to a flat.
    (5) labels has been initialized to zero prior to the first call to this function.
    (6) labels has values greater than or equal to 1 for each processed cell which is in a flat.
    Each flat's cells bear a label unique to that flat.
    At exit:
    (1) flat_row, flat_col and every cell reachable from flat_row, flat_col by passing over only
    cells of the same elevation as flat_row, flat_col (all the cells in the flat to which c belongs)
    is marked as new_label in Labels.
    (2) labels has been updated to reflect the new labels which have been applied.

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


@njit
def away_from_higher(
    labels: np.ndarray,
    flat_mask: np.ndarray,
    fdr: np.ndarray,
    high_edges: list,
    flat_height: np.array,
) -> None:
    """Algorithm 5 AwayFromHigher: This procedure builds a gradient away from higher terrain,
    as described in §2.3 and Fig. 1.
    Upon entry:
    (1) Every cell in Labels is marked either 0, indicating that the cell is not part of a flat,
    or a number greater than zero which identifies the flat to which the cell belongs.
    (2) Any cell without a local gradient is marked NoFlow in FlowDirs.
    (3) Every cell in FlatMask is initialized to 0. (4) HighEdges contains, in no particular order,
    all the high edge cells of the DEM which are part of drainable flats.
    At exit:
    (1) flat_height has an entry for each label value of Labels indicating the maximal number of
    increments to be applied to the flat identified by that label.
    (2) flat_mask contains the number of increments to be applied to each cell to form a gradient
    away from higher terrain; cells not in a flat have a value of 0.

    Args:
        labels (np.ndarray): The labels of each flat, same shape as the DEM. 0 means not part of a flat.
        flat_mask (np.ndarray): The flat mask, same shape as the DEM. 0 means not part of a flat.
        fdr (np.ndarray): The flow direction raster
        high_edges (np.ndarray): The high edge cells of the DEM. In no particular order. FIFO queue.
        flat_height (np.array): The flat height array, size of the number of flats
    """
    loops = 1
    marker = (-1, -1)
    high_edges.append(marker)
    while len(high_edges) > 1:
        row, col = high_edges.pop(0)
        if row == marker[0] and col == marker[1]:
            loops += 1
            high_edges.append(marker)
            continue
        if flat_mask[row, col] > 0:
            continue
        flat_mask[row, col] = loops
        flat_height[labels[row, col] - 1] = loops
        for d_row, d_col in NEIGHBOR_OFFSETS:
            neighbor_row = row + d_row
            neighbor_col = col + d_col
            if (
                neighbor_row < 0
                or neighbor_row >= labels.shape[0]
                or neighbor_col < 0
                or neighbor_col >= labels.shape[1]
            ):
                continue
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                high_edges.append((neighbor_row, neighbor_col))


@njit
def towards_lower(
    labels: np.ndarray,
    flat_mask: np.ndarray,
    fdr: np.ndarray,
    low_edges: list,
    flat_height: np.array,
) -> None:
    """Algorithm 6 TowardsLower:  This procedure builds a gradient towards lower terrain and
    combines it with the gradient away from higher terrain, as described in §2.4 and Fig. 2.
    Upon entry:
    (1) Every cell in Labels is marked either 0, indicating that the cell is not part of a flat,
    or a number greater than zero which identifies the flat to which the cell belongs.
    (2) Any cell without a local gradient is marked NoFlow in FlowDirs.
    (3) Every cell in FlatMask has either a value of 0, indicating that the cell is not part of
    a flat, or a value greater than zero indicating the number of increments which must be
    added to it to form a gradient away from higher terrain.
    (4) FlatHeight has an entry for each label value of Labels indicating the maximal number
    of increments to be applied to the flat identified by that label in order to form the
    gradient away from higher terrain.
    (5) LowEdges contains, in no particular order, all the low edge cells of the DEM.
    At exit:
    (1) FlatMask contains the number of increments to be applied to each cell to form a
    superposition of the gradient away from higher terrain with the gradient towards lower
    terrain; cells not in a flat have a value of 0.

    Args:
        labels (np.ndarray): The labels of each flat, same shape as the DEM. 0 means not part of a flat.
        flat_mask (np.ndarray): The flat mask, same shape as the DEM. 0 means not part of a flat.
        fdr (np.ndarray): The flow direction raster
        low_edges (np.ndarray): The low edge cells of the DEM. In no particular order. FIFO queue.
        flat_height (np.array): The flat height array, size of the number of flats
    """
    # make all entries in flat_mask negative
    flat_mask *= -1
    loops = 1
    marker = (-1, -1)
    low_edges.append(marker)
    while len(low_edges) > 1:
        row, col = low_edges.pop(0)
        if row == marker[0] and col == marker[1]:
            loops += 1
            low_edges.append(marker)
            continue
        if flat_mask[row, col] > 0:
            continue
        if flat_mask[row, col] < 0:
            flat_mask[row, col] += flat_height[labels[row, col] - 1] + 2 * loops
        else:
            flat_mask[row, col] = 2 * loops
        for d_row, d_col in NEIGHBOR_OFFSETS:
            neighbor_row = row + d_row
            neighbor_col = col + d_col
            if (
                neighbor_row < 0
                or neighbor_row >= labels.shape[0]
                or neighbor_col < 0
                or neighbor_col >= labels.shape[1]
            ):
                continue
            if (
                labels[neighbor_row, neighbor_col] == labels[row, col]
                and fdr[neighbor_row, neighbor_col] == FLOW_DIRECTION_UNDEFINED
            ):
                low_edges.append((neighbor_row, neighbor_col))
