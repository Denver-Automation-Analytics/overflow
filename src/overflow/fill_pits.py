import time
import heapq
from osgeo import gdal
from numba import njit, int32,int64, float32, typed
from numba.experimental import jitclass
import numpy as np

EDGE_LABEL = int64(1)
spec = [
    ("row", int64),
    ("column", int64),
    ("cost", float32),
]
@jitclass(spec)
class GridCell:
    """A class to represent a cell in the grid. Used with heapq to prioritize cells by cost."""

    def __init__(self, row, column, cost):
        self.row = row
        self.column = column
        self.cost = cost

    # Define comparison methods based on the cost attribute so this can be used in a heapq
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost

@njit()
def process_tile(dem, is_edge_tile=False):
    labels = np.zeros_like(dem, dtype=np.int64)
    graph = dict()
    label_counter = 2

    rows, cols = dem.shape
    # TODO we push and pop dummy value to get numba to compile
    open_cells = [GridCell(0, 0, dem[0, 0])]
    heapq.heapify(open_cells)
    pit_cells = [GridCell(0, 0, dem[0, 0])]
    heapq.heapify(pit_cells)
    heapq.heappop(pit_cells)
    heapq.heappop(open_cells)

    def push_to_open_or_pit(cell):
        grid_cell = GridCell(cell[1], cell[2], cell[0])
        if cell[1] == 0 or cell[1] == rows - 1 or cell[2] == 0 or cell[2] == cols - 1:
            heapq.heappush(pit_cells, grid_cell)
        else:
            heapq.heappush(open_cells, grid_cell)

    # Push cells on the edges of dem onto open with priority dem value
    for i in range(rows):
        push_to_open_or_pit((dem[i, 0], i, 0))
        push_to_open_or_pit((dem[i, cols - 1], i, cols - 1))
    for j in range(1, cols - 1):
        push_to_open_or_pit((dem[0, j], 0, j))
        push_to_open_or_pit((dem[rows - 1, j], rows - 1, j))

    while open_cells or pit_cells:
        if pit_cells:
            current_cell = heapq.heappop(pit_cells)
        else:
            current_cell = heapq.heappop(open_cells)

        if labels[current_cell.row, current_cell.column] == 0:
            # Check neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_x, next_y = current_cell.row + dx, current_cell.column + dy
                if 0 <= next_x < rows and 0 <= next_y < cols:
                    if (
                        labels[next_x, next_y] != 0
                        and dem[next_x, next_y] <= dem[current_cell.row, current_cell.column]
                    ):
                        labels[current_cell.row, current_cell.column] = labels[
                            next_x, next_y
                        ]
                        break
            else:
                labels[current_cell.row, current_cell.column] = label_counter
                label_counter += 1

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_x, next_y = current_cell.row + dx, current_cell.column + dy
            if 0 <= next_x < rows and 0 <= next_y < cols:
                if labels[next_x, next_y] != 0:
                    if (
                        labels[current_cell.row, current_cell.column]
                        == labels[next_x, next_y]
                    ):
                        continue
                    edge = max(
                        dem[current_cell.row, current_cell.column], dem[next_x, next_y]
                    )
                    key = (
                        labels[current_cell.row, current_cell.column],
                        labels[next_x, next_y],
                    )
                    if key not in graph or edge < graph[key]:
                        graph[key] = edge
                else:
                    labels[next_x, next_y] = labels[current_cell.row, current_cell.column]
                    if dem[next_x, next_y] <= current_cell.cost:
                        dem[next_x, next_y] = current_cell.cost
                        heapq.heappush(pit_cells, GridCell(next_x, next_y, dem[next_x, next_y]))
                    else:
                        heapq.heappush(
                            open_cells, GridCell(next_x, next_y, dem[next_x, next_y])
                        )

    # Process edge tile
    if is_edge_tile:
        for i in range(rows):
            for j in [0, cols - 1]:
                if (labels[i, j], EDGE_LABEL) not in graph or dem[i, j] < graph[(labels[i, j], EDGE_LABEL)]:
                    graph[(labels[i, j], EDGE_LABEL)] = dem[i, j]
        for j in range(1, cols - 1):
            for i in [0, rows - 1]:
                if (labels[i, j], EDGE_LABEL) not in graph or dem[i, j] < graph[(labels[i, j], EDGE_LABEL)]:
                    graph[(labels[i, j], EDGE_LABEL)] = dem[i, j]

    return labels, graph


# Create Tile object
np.random.seed(32)
dem = np.random.randint(10, 50, size=(5, 5)).astype(np.float32)
print(dem)
dem_copy = dem.copy()
# Process tile
start_time = time.time()
labels, graph = process_tile(dem, True)
end_time = time.time()
print(f'Filling the pits took {end_time - start_time} seconds')
start_time = time.time()
labels, graph = process_tile(dem_copy, True)
end_time = time.time()
print(f'Filling the pits took {end_time - start_time} seconds')
print(dem)
print(labels)
print(graph)


# ds = gdal.Open("data/clip.tif")
# dem = ds.GetRasterBand(1).ReadAsArray()
#
# start_time = time.time()
# Tile = Tile(dem, np.zeros_like(dem), {}, True)
# processed_tile = process_tile(Tile)
# end_time = time.time()
# print(f"Filling the pits took {end_time - start_time} seconds")
#
# # create new raster dataset for DEM
# driver = gdal.GetDriverByName("GTiff")
# out_ds = driver.Create(
#     "data/filled_pits.tif", dem.shape[1], dem.shape[0], 1, gdal.GDT_Float32
# )
# out_ds.SetProjection(ds.GetProjection())
# out_ds.SetGeoTransform(ds.GetGeoTransform())
# out_band = out_ds.GetRasterBand(1)
# out_band.SetNoDataValue(ds.GetRasterBand(1).GetNoDataValue())
# out_band.WriteArray(processed_tile.dem)
#
# # create new raster dataset for Labels
# out_ds = driver.Create("data/labels.tif", dem.shape[1], dem.shape[0], 1, gdal.GDT_Int32)
# out_ds.SetProjection(ds.GetProjection())
# out_ds.SetGeoTransform(ds.GetGeoTransform())
# out_band = out_ds.GetRasterBand(1)
# out_band.SetNoDataValue(ds.GetRasterBand(1).GetNoDataValue())
# out_band.WriteArray(processed_tile.labels)
#
# ds = None
# out_ds = None
#
