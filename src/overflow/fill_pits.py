import time
import heapq
from osgeo import gdal
import numpy as np


class Tile:
    def __init__(self, dem, labels, graph, is_edge_tile=False):
        self.dem = dem
        self.labels = labels
        self.graph = graph
        self.is_edge_tile = is_edge_tile


def unique_label():
    unique_label.counter += 1
    return unique_label.counter


unique_label.counter = 0


def process_tile(tile):
    dem = tile.dem
    labels = tile.labels
    graph = tile.graph

    rows, cols = dem.shape
    open_cells = []
    pit_cells = []

    # Initialize labels to 0
    labels.fill(0)

    def push_to_open_or_pit(cell):
        if cell[0] == 0 or cell[0] == rows - 1 or cell[1] == 0 or cell[1] == cols - 1:
            heapq.heappush(pit_cells, cell)
        else:
            heapq.heappush(open_cells, cell)

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

        if labels[current_cell[1], current_cell[2]] == 0:
            # Check neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_x, next_y = current_cell[1] + dx, current_cell[2] + dy
                if 0 <= next_x < rows and 0 <= next_y < cols:
                    if (
                        labels[next_x, next_y] != 0
                        and dem[next_x, next_y] <= dem[current_cell[1], current_cell[2]]
                    ):
                        labels[current_cell[1], current_cell[2]] = labels[
                            next_x, next_y
                        ]
                        break
            else:
                labels[current_cell[1], current_cell[2]] = unique_label()

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_x, next_y = current_cell[1] + dx, current_cell[2] + dy
            if 0 <= next_x < rows and 0 <= next_y < cols:
                if labels[next_x, next_y] != 0:
                    if (
                        labels[current_cell[1], current_cell[2]]
                        == labels[next_x, next_y]
                    ):
                        continue
                    edge = max(
                        dem[current_cell[1], current_cell[2]], dem[next_x, next_y]
                    )
                    other_edge = graph.get(
                        (
                            labels[current_cell[1], current_cell[2]],
                            labels[next_x, next_y],
                        ),
                        None,
                    )
                    if other_edge is None or edge < other_edge:
                        graph[
                            (
                                labels[current_cell[1], current_cell[2]],
                                labels[next_x, next_y],
                            )
                        ] = edge
                else:
                    labels[next_x, next_y] = labels[current_cell[1], current_cell[2]]
                    if dem[next_x, next_y] <= current_cell[0]:
                        dem[next_x, next_y] = current_cell[0]
                        heapq.heappush(pit_cells, (dem[next_x, next_y], next_x, next_y))
                    else:
                        heapq.heappush(
                            open_cells, (dem[next_x, next_y], next_x, next_y)
                        )

    # Process edge tile
    if tile.is_edge_tile:
        for i in range(rows):
            for j in [0, cols - 1]:
                other_edge = graph.get((labels[i, j], 1), None)
                if other_edge is None or dem[i, j] < other_edge:
                    graph[(labels[i, j], 1)] = dem[i, j]
        for j in range(1, cols - 1):
            for i in [0, rows - 1]:
                other_edge = graph.get((labels[i, j], 1), None)
                if other_edge is None or dem[i, j] < other_edge:
                    graph[(labels[i, j], 1)] = dem[i, j]

    return tile


# Create Tile object
np.random.seed(32)
dem = np.random.randint(10, 50, size=(5, 5))
print(dem)
Tile = Tile(dem, np.zeros_like(dem), {})
# Process tile
processed_tile = process_tile(Tile)
# Access processed tile attributes
processed_dEM = processed_tile.dem
processed_labels = processed_tile.labels
processed_graph = processed_tile.graph
print(processed_dEM)
print(processed_labels)
print(processed_graph)


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