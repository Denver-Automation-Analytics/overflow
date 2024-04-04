from numba.experimental import jitclass
from overflow.numba_types import Int64PairListList, Int64PairList, Int64Pair
from numba.typed import List
import numpy as np
import heapq


@jitclass
class LocalGraph:
    flat_edges: Int64PairListList
    high_edges: Int64PairList
    low_edges: Int64PairList
    tile_index: int

    def __init__(
        self,
        labels_perimeter: np.ndarray,
        to_higher_perimeter: np.ndarray,
        to_lower_perimeter: np.ndarray,
        min_dists: np.ndarray,
        tile_index: int,
    ):
        self.tile_index = tile_index
        cell_count = labels_perimeter.size
        self.flat_edges = List.empty_list(Int64PairList)
        for _ in range(cell_count):
            self.flat_edges.append(List.empty_list(Int64Pair))
        index_offset = self.tile_index * cell_count

        for i, label in enumerate(labels_perimeter):
            if label == 0:
                # this cell is not part of a flat
                continue
            # walk around the perimeter and connect neighboring cells in the same group
            # a neighboring cell is defined as the next or previous cell with the same label
            # as seen by continuing clockwise around the perimeter, not necessarily the
            # cell directly adjacent in the grid
            for j in range(i + 1, cell_count + 1):
                k = j % cell_count
                if k != i and labels_perimeter[k] == label:
                    dist = min_dists[i][k]
                    self.flat_edges[i].append((k + index_offset, dist))
                    self.flat_edges[k].append((i + index_offset, dist))
                    break
            # connect non-neighboring cells to each other if they are part of the same group
            # we can skip any edges where there exist other paths that have the shortest
            # distances. For example, if the difference in the distances of vertex i to two
            # vertices (j and k) equals the distance of the
            # two end vertices to each other, only the closest vertex should be connected
            # since the path containing the vertices in between is among the shortest
            # paths between the two end vertices.
            # Theoretically, a fully connected graph would produce identical results.
            # However, this method is used to reduce the number of unnecessary edges in the graph.
            for j in range(i + 2, cell_count):
                if labels_perimeter[j] != label:
                    # this cell is not part of the same flat
                    continue
                dist_ij = min_dists[i][j]
                for k in range(i + 1, j):
                    if labels_perimeter[k] != label:
                        continue
                    dist_ik = min_dists[i][k]
                    dist_kj = min_dists[k][j]
                    if dist_ij == dist_ik + dist_kj:
                        break
                else:
                    self.flat_edges[i].append((j + index_offset, dist_ij))
                    self.flat_edges[j].append((i + index_offset, dist_ij))

        # connect each cell to the high/low terrain node
        for i, distance in enumerate(to_higher_perimeter):
            if distance == 0:
                continue
            self.high_edges.append((i + index_offset, distance))

        for i, distance in enumerate(to_lower_perimeter):
            if distance == 0:
                continue
            self.low_edges.append((i + index_offset, distance))


@jitclass
class GlobalGraph:
    flat_edges: Int64PairListList
    high_edges: Int64PairList
    low_edges: Int64PairList
    tile_rows: int
    tile_cols: int
    perimeter_cells_per_tile: int

    def __init__(self, tile_rows: int, tile_cols: int, perimeter_cells_per_tile: int):
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.perimeter_cells_per_tile = perimeter_cells_per_tile
        self.flat_edges = List.empty_list(Int64PairList)
        size = tile_rows * tile_cols * perimeter_cells_per_tile
        for _ in range(size):
            self.flat_edges.append(List.empty_list(Int64Pair))
        self.high_edges = List.empty_list(Int64Pair)
        self.low_edges = List.empty_list(Int64Pair)

    def add(self, local_graph: LocalGraph):
        for local_index, edge in enumerate(local_graph.flat_edges):
            global_index = local_index + local_graph.tile_index * len(
                local_graph.flat_edges
            )
            self.flat_edges[global_index] = edge
        self.high_edges.extend(local_graph.high_edges)
        self.low_edges.extend(local_graph.low_edges)

    def join(self):
        for i in range(self.tile_rows - 1):
            for j in range(self.tile_cols - 1):
                pass
                # join_adjacent_tiles()

    def _djikstra_dist_to_edge(self, edge_type: str, dist_to_low_edge: dict = {}):
        dist_to_edge = {}
        pq = List.empty_list(Int64Pair)

        # populate pq with the terrain node and its neighbors
        def populate_edges(edges):
            for neighbor, weight in edges:
                # ignore low edge cells for distance to high edge calculation
                if neighbor in dist_to_low_edge and dist_to_low_edge[neighbor] == 1:
                    continue
                dist_to_edge[neighbor] = weight
                heapq.heappush(pq, (weight, neighbor))

        if edge_type == "low":
            populate_edges(self.low_edges)
        elif edge_type == "high":
            populate_edges(self.high_edges)
        else:
            raise ValueError(f"Invalid edge_type: {edge_type}")

        while pq:
            dist, node = heapq.heappop(pq)
            if dist > dist_to_edge[node]:
                continue
            for neighbor, weight in self.flat_edges[node]:
                # ignore low edge cells for distance to high edge calculation
                if neighbor in dist_to_low_edge and dist_to_low_edge[neighbor] == 1:
                    continue
                if (
                    neighbor not in dist_to_edge
                    or dist + weight < dist_to_edge[neighbor]
                ):
                    dist_to_edge[neighbor] = dist + weight
                    heapq.heappush(pq, (dist + weight, neighbor))
        return dist_to_edge

    def _extract_perimeter_distances(
        self, dist_to_edge, perimeter_cells_per_tile, tile_count
    ):
        perimeter_distances = np.full(
            (tile_count, perimeter_cells_per_tile), -1, dtype=np.int64
        )
        for k, v in dist_to_edge.items():
            tile_index = k // perimeter_cells_per_tile
            cell_index = k % perimeter_cells_per_tile
            perimeter_distances[tile_index, cell_index] = v
        return perimeter_distances

    def _get_tile_exterior_edges(self, tile_index, perimeter_distances):
        exterior_edges = List()
        max_loops = 0
        loop_index_map = {}
        index = 0
        for dist_to_edge_tile in perimeter_distances[tile_index]:
            dist_to_edge_data = TileData(dist_to_edge_tile, 0)
            for i, dist in enumerate(dist_to_edge_tile):
                if dist <= 1:
                    continue
                row, col = dist_to_edge_data.get_row_col(i)
                if dist not in loop_index_map:
                    max_loops = max(max_loops, dist)
                    loop_index_map[dist] = index
                    index += 1
                    new_edge = List()
                    new_edge.append((row, col))
                    exterior_edges.append(new_edge)
                else:
                    exterior_edges[loop_index_map[dist]].append((row, col))
        return exterior_edges, max_loops, loop_index_map

    def solve_graph(self):
        # results in exterior_edges for high and low terrain
        # with max loops and index map for each tile
        # see solve high graph, solve low graph, and get_tile_exterior_edges

        dist_to_low_edge = self._djikstra_dist_to_edge("low")
        dist_to_high_edge = self._djikstra_dist_to_edge("high", dist_to_low_edge)
        # now get a map of all cells on the perimeter of each tile and the
        # distances to the nearest high and low edges

        dist_to_low_edge_tiles = self._extract_perimeter_distances(
            dist_to_low_edge,
            self.perimeter_cells_per_tile,
            self.tile_rows * self.tile_cols,
        )

        dist_to_high_tiles = self._extract_perimeter_distances(
            dist_to_high_edge,
            self.perimeter_cells_per_tile,
            self.tile_rows * self.tile_cols,
        )

        # get_tile_exterior_edges for all tiles
        # can't have lists of dicts in numba
        global_high_exterior_edges = List()
        global_max_high_loops = []
        global_loop_index_map_high = []
        global_low_exterior_edges = List()
        global_max_low_loops = []
        global_loop_index_map_low = []

        # we need for both low and high terrain
        # interior_edges,
        # exterior_edges,
        # loop_index_map,
        # max_high_loop,
