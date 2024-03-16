import numpy as np

# constants used in the overflow module
DEFAULT_SEARCH_RADIUS = 200
DEFAULT_MAX_PITS = 24
UNVISITED_INDEX = -1
EPSILON_GRADIENT = 1e-5  # small value to apply to gradient of breaching to nodata cells
DEFAULT_CHUNK_SIZE = 2000
# The offsets for neighbors of each cell in a raster
NEIGHBORS = np.array(
    [
        (0, 1),  # Right
        (1, 0),  # Down
        (0, -1),  # Left
        (-1, 0),  # Up
        (-1, -1),  # Upper Left
        (-1, 1),  # Upper Right
        (1, -1),  # Lower Left
        (1, 1),  # Lower Right
    ],
    dtype=np.int64,
)
