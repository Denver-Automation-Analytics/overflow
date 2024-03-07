import numpy as np
from numba import njit
import time

# write a function that creates a boolean flag matrix starting with all edge cells marked as true and all other cells marked as false
@njit
def initalize_flag(dem: np.array):
    """Initialize a boolean flag matrix with all edge cells marked as true and all other cells marked as false

    Args:
        dem (np.array): Digital Elevation Model

    Returns:
        np.array: Boolean flag matrix
    """
    flag = np.full(dem.shape, False, dtype=np.bool_)
    flag[0, :] = True # set the first row to True
    flag[-1, :] = True # set the last row to True
    flag[:, 0] = True # set the first column to True
    flag[:, -1] = True # set the last column to True
    return flag

# write a function that sorts an array with three columns based on the first column
@njit
def sort_array(array: np.array):
    """Sort a 2D array with three columns based on the first column

    Args:
        array (np.array): 2D array with three columns

    Returns:
        np.array: Sorted array
    """
    n = len(array)
    for i in range(n):
        for j in range(0, n-i-1):
            if array[j, 0] > array[j+1, 0]:
                array[j, :], array[j+1, :] = array[j+1, :].copy(), array[j, :].copy()
    return array

# write a function that deletes an item from a 2D array given the item index
@njit
def delete_array_item(array: np.array, idx: int):
    """Delete an item from a 2D array

    Args:
        array (np.array): 2D array
        idx (int): Index of the item to delete

    Returns:
        np.array: Array with the item deleted
    """
    result = np.empty((array.shape[0] - 1, array.shape[1]), dtype=array.dtype)
    j = 0
    for i in range(array.shape[0]):
        if i != idx:
            result[j] = array[i]
            j += 1
    return result

# write a function that stacks two arrays vertically
@njit
def vstack(a: np.array, b: np.array):
    """Stack two arrays vertically

    Args:
        a (np.array): First array
        b (np.array): Second array

    Returns:
        np.array: Vertically stacked array
    """
    rows, cols = a.shape[0] + b.shape[0], a.shape[1]
    result = np.empty((rows, cols), dtype=a.dtype)
    for i in range(a.shape[0]):
        for j in range(cols):
            result[i, j] = a[i, j]
    for i in range(b.shape[0]):
        for j in range(cols):
            result[i + a.shape[0], j] = b[i, j]
    return result

# write a function to combine three 1D arrays into a single 2D array with three columns
@njit
def column_stack(a: np.array, b: np.array, c: np.array, data_type: np.dtype):
    """Combine three 1D arrays into a single 2D array with three columns

    Args:
        a (np.array): 1D array
        b (np.array): 1D array
        c (np.array): 1D array
        data_type (np.dtype): Data type for the combined array

    Returns:
        np.array: Combined 2D array with three columns
    """
    rows = a.shape[0]
    result = np.empty((rows, 3), dtype=data_type)
    for i in range(rows):
        result[i, 0] = a[i]
        result[i, 1] = b[i]
        result[i, 2] = c[i]
    return result

# write a function that creates the priority que and returns the pq and index arrays for reference
@njit
def initalize_pq(dem: np.array, flag: np.array):
    """Create the priority que and return the pq and index arrays for reference

    Args:
        dem (np.array): Digital Elevation Model
        flag (np.array): Boolean flag matrix

    Returns:
        np.array: Priority que reference array
    """
    # create indices for the 2d array
    rows, cols = np.indices((dem.shape))
    # create empty lists for the pq and index arrays
    pq = []
    pq_row_idx = []
    pq_col_idx = []
    # iterate over each element in the dem array
    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            # if the corresponding element in the flag array is True,
            # add the dem element to the pq array and the indices to the index arrays
            if flag[i, j]:
                pq.append(dem[i, j])
                pq_row_idx.append(rows[i, j])
                pq_col_idx.append(cols[i, j])
    # convert the lists to arrays
    pq = np.array(pq, dtype=dem.dtype)
    pq_row_idx = np.array(pq_row_idx, dtype=np.int32)
    pq_col_idx = np.array(pq_col_idx, dtype=np.int32)
    # assign the data type for the combined array
    # dt = np.dtype([('col1', dem.dtype), ('col2', pq_row_idx.dtype), ('col3', pq_col_idx.dtype)])
    # combine the pq and index arrays into a single 2d reference array
    pq_ref = column_stack(pq, pq_row_idx, pq_col_idx, np.int32)
    # pq_ref = np.column_stack((pq, pq_row_idx, pq_col_idx))
    # sort the reference array by the first column (the pq values)
    #pq_ref = pq_ref[pq_ref[:, 0].argsort()]
    pq_ref = sort_array(pq_ref)
    # return the pq reference array [value, row, col]
    return pq_ref

# determine the nearest neighbor of a given cell
@njit
def nearest_neighbors(row, col, dem):
    """
    Determine the nearest neighbor of a given cell
    
    Args:
        row (int): Row index
        col (int): Column index
        dem (np.array): Digital Elevation Model

    Returns:
        list: List of tuples of the nearest neighbors
    """
    # create a list of tuples for the 8 possible neighbors
    neighbors = [(row-1, col),
                 (row+1, col),
                 (row, col-1),
                 (row, col+1),
                 (row-1, col-1),
                 (row-1, col+1),
                 (row+1, col-1),
                 (row+1, col+1)]
    # create a list of valid neighbors
    valid_neighbors = []
    for r, c in neighbors:
        if r >= 0 and r < dem.shape[0] and c >= 0 and c < dem.shape[1]:
            valid_neighbors.append((r, c))
    # return the rows and columns of the valid neighbors as a list of tuples [(row, col), (row, col), ...]
    return np.array(valid_neighbors, dtype=np.int32)

# check if a numpy array is empty
@njit
def is_empty(array: np.array):
    """
    Check if a numpy array is empty

    Args:
        array (np.array): Numpy array

    Returns:
        bool: True if array is empty, False otherwise
    """
    if array.size == 0:
        return True
    else:
        return False

# write a function that fills the depression pits in a digital elevation model
@njit
def priority_flood_fill(dem: np.array,
                        data_type: np.dtype):
    """
    Fill the depression pits in a digital elevation model using a regional growth algorithm
    source: https://doi.org/10.1016/j.cageo.2013.04.024
    Args:
        dem (np.array): Digital Elevation Model
        data_type (np.dtype): Data type of the input DEM

    Returns:
        np.array: Digital Elevation Model with filled pits
    """
    # initialize the flag matrix
    flag = initalize_flag(dem)
    # priority_queue is the priority queue reference array [value, row, col] sorted by DEM edge cells from low to high in elevation
    priority_queue = initalize_pq(dem, flag)
    # plain_queue is the plain queue reference array [value, row, col] which collects neighbors whose values are less than the current cell
    plain_queue = np.array([], dtype=data_type)

    # check if both the priority_queue and the plain_queue are empty
    while is_empty(priority_queue) == False:
        # check if the plain queue is empty
        if is_empty(plain_queue) == False:
            #print('The plain queue is not empty')
            # check if the plain_queue queue has more than one element
            if plain_queue.shape[0] > 1:
                #print('The plain queue has more than one element')
                #print(plain_queue.shape[0])
                # sort the plain queue by the first column (the elevation values)
                plain_queue = sort_array(plain_queue)
                # plain_queue = plain_queue[plain_queue[:, 0].argsort()]
                # assign the first element within plain_queue to a variable [value, row, col]
                c = plain_queue[0]
                dem_c, row_c, col_c = c[0], c[1], c[2] # elev, row, col
            else:
                # only one element within the plain queue
                #print('Only one element within the plain queue')
                c = plain_queue[0]
                dem_c, row_c, col_c = c[0], c[1], c[2] # elev, row, col
            # pop the first element out from the plain_queue such that the number of rows reduces by 1
            plain_queue = delete_array_item(plain_queue, 0)
            # plain_queue = np.delete(plain_queue, 0, 0)
            # determine the nearest neighbors of the cell
            neighbors = nearest_neighbors(row_c, col_c, dem)
            # iterate through the neighbors
            for n in neighbors:
                row_n, col_n = n[0], n[1]
                # check if the neighbor has been resolved yet or is an edge cell
                if flag[(row_n, col_n)] == True:
                    #print('Within the plain queue, passing over an edge cell or an already resolved neighbor cell')
                    pass
                else:
                    # if the neighbor has not yet been resolved and is not an edge cell
                    # check if it has an elevation less than cell c
                    if dem[n] < dem_c:
                        #print(f'Within the plain queue, corrected the elevation value from {dem[n]} to {dem_c}')
                        # raise the elevation at n to the elevation at c
                        dem[n] = dem_c
                        # set the flag at n to true to indicate that it has been resolved
                        flag[n] = True
                        # add n to the plain queue
                        # first check if the plain queue is empty
                        if is_empty(plain_queue):
                            # add the first element to the plain queue
                            plain_queue = np.array([dem[n], row_n, col_n]).reshape(1,3)
                        else:
                            # stack the ith element to the plain queue
                            plain_queue = vstack(plain_queue, np.array([dem[n], row_n, col_n]).reshape(1,3))
                            # plain_queue = np.vstack((plain_queue, (dem[n], row_n, col_n)))
                    else:
                        # push c into the priority queue
                        #print('Within the plain queue, passing over a neighbor cell with a higher elevation than cell')
                        priority_queue = vstack(priority_queue, np.array([dem_c, row_c, col_c]).reshape(1,3))
                        # priority_queue = np.vstack((priority_queue, (dem_c, row_c, col_c)))
        else:
            # there are no elements within the plain queue, so the first element from the priority_queue is popped out
            if priority_queue.shape[0] > 1:
                # sort the priority queue by the first column (the elevation values)
                priority_queue = sort_array(priority_queue)
                # priority_queue = priority_queue[priority_queue[:, 0].argsort()]
                # assign the first element within priority_queue to a variable [value, row, col]
                c = priority_queue[0]
                dem_c, row_c, col_c = c[0], c[1], c[2] # elev, row, col
                # pop the first element out from the plain_queue such that the number of rows reduces by 1
                priority_queue = delete_array_item(priority_queue, 0)
                # priority_queue = np.delete(priority_queue, 0, 0)
                # determine the nearest neighbors of the cell
                neighbors = nearest_neighbors(row_c, col_c, dem)
                # iterate through the neighbors
                for n in neighbors:
                    row_n, col_n = n[0], n[1]
                    # check if the neighbor has been resolved yet or is an edge cell
                    if flag[(row_n, col_n)] == True:
                        print('Within the priority queue 1, passing over an edge cell or an already resolved neighbor cell')
                        pass
                    else:
                        # if the neighbor has not yet been resolved and is not an edge cell
                        # check if it its elevation is less equal to cell c
                        if dem[n] < dem_c:
                            print(f'Within the priority queue 2, corrected the elevation value from {dem[n]} to {dem_c}')
                            # set the dem value at n to the cell value at c
                            dem[n] = dem_c
                            # set the flag at n to true to indicate that it has been resolved
                            flag[n] = True
                            # add n to the plain queue
                            # first check if the plain queue is empty
                            if is_empty(plain_queue):
                                # add the first element to the plain queue
                                plain_queue = np.array([dem[n], row_n, col_n]).reshape(1,3)
                            else:
                                # stack the ith element to the plain queue
                                plain_queue = vstack(plain_queue, np.array([dem[n], row_n, col_n]).reshape(1,3))
                                # plain_queue = np.vstack((plain_queue, (dem[n], row_n, col_n)))
                        else:
                            print('Within the priority queue 3, passing over a neighbor cell with a higher elevation than cell')
                            pass
            else:
                # only one element within the priority queue
                c = priority_queue[0]
                dem_c, row_c, col_c = c[0], c[1], c[2] # elev, row, col
                # pop the first element out from the plain_queue such that the number of rows reduces by 1
                priority_queue = np.array([], dtype=data_type)
                # determine the nearest neighbors of the cell
                neighbors = nearest_neighbors(row_c, col_c, dem)
                # iterate through the neighbors
                for n in neighbors:
                    row_n, col_n = n[0], n[1]
                    # check if the neighbor has been resolved yet or is an edge cell
                    if flag[(row_n, col_n)] == True:
                        print('Within the priority queue 4, passing over an edge cell or an already resolved neighbor cell')
                        pass
                    else:
                        # if the neighbor has not yet been resolved and is not an edge cell
                        # check if it its elevation is less equal to cell c
                        if dem[n] < dem_c:
                            print(f'Within the priority queue 5, corrected the elevation value from {dem[n]} to {dem_c}')
                            # set the dem value at n to the cell value at c
                            dem[n] = dem_c
                            # set the flag at n to true to indicate that it has been resolved
                            flag[n] = True
                            # add n to the plain queue
                            # first check if the plain queue is empty
                            if is_empty(plain_queue):
                                # add the first element to the plain queue
                                plain_queue = np.array([dem[n], row_n, col_n]).reshape(1,3)
                            else:
                                # stack the ith element to the plain queue
                                plain_queue = vstack(plain_queue, np.array([dem[n], row_n, col_n]).reshape(1,3))
                                # plain_queue = np.vstack((plain_queue, (dem[n], row_n, col_n)))
                        else:
                            print('Within the priority queue 6, passing over a neighbor cell with a higher elevation than cell')
                            pass
    return dem

###########################################
# test the function with an example DEM
###########################################

# set a random seed for repeatable results
np.random.seed(32)
test_dem_input = np.random.randint(10, 50, size=(100, 100), dtype=np.int32)
test_dem_input_copy = test_dem_input.copy()

# fill the pits in the DEM
start_time = time.time()
filled_dem = priority_flood_fill(test_dem_input, test_dem_input.dtype)
end_time = time.time()
print(f'Filling the pits took {end_time - start_time} seconds')