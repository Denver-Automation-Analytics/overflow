import numpy as np
from numba import njit

@njit(parallel=True)
def breach_single_cell_pits_in_chunk(chunk,nodata_value)-> tuple[np.ndarray,np.ndarray] :
    """
    This function is used to breach single cell pits in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with breached single cell pits.

    Parameters
    ----------
    chunk : np.ndarray
        A chunk of a DEM.

    Returns
    -------
    np.ndarray
        A chunk of a DEM with breached single cell pits.
    """
    dx=[1,1,1,0,-1,-1,-1,0]
    dy=[-1,0,1,1,1,0,-1,-1]
    dx2=[2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1]
    dy2=[-2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2]
    breachcell=[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,0]
    # Create a copy of the chunk
    chunk_copy = chunk.copy()
    # Get the shape of the chunk
    rows, cols = chunk.shape
    # Loop through each cell in the chunk
    unsolved_pits_raster = np.zeros_like(chunk, dtype=np.int8)
    for row in range(2,rows-2):
        for col in range(2,cols-2):
            z=chunk[row,col]
            if z != nodata_value:
                flag = True
                for k in range(8):
                    
                    if int(row+dy[k]) < 0 or int(col+dx[k]) < 0 :
                        pass
                    else:
                        zn=chunk[row+dy[k],col+dx[k]]
                        
                        if zn < z and zn != nodata_value:
                            flag = False
                            break
                            
                if flag:
                    unsolved=True
                    for k in range(16):
                        zn=chunk[row+dy2[k],col+dx2[k]]
                        if zn < z and zn != nodata_value:
                            unsolved=False
                            if int(row+dy[breachcell[k]]) < 0 or int(col+dx[breachcell[k]]) < 0:
                                pass
                            else:
                                chunk_copy[row+dy[breachcell[k]],col+dx[breachcell[k]]]=(z+zn)/2
                    if unsolved:
                        unsolved_pits_raster[row,col]=1
                        

    return chunk_copy,unsolved_pits_raster



chunk = np.array(
    [   [-999,-999,-999, -999, -999, -999, -999,-999,-999],
        [-999,-999,-999, -999, -999, -999, -999,-999,-999],
        [-999,-999,100, 101, 90, 97, 90,-999,-999],
        [-999, -999,103, 102, 80, 96, 95,-999,-999],
        [-999, -999,94, 95, 96, 95, 94,-999,-999],
        [-999, -999,97, 98, 50, 94, 90,-999,-999],
        [-999, -999,95, 90, 85, 40, 92,-999,-999],
        [-999,-999,-999, -999, -999, -999, -999,-999,-999],
        [-999,-999,-999, -999, -999, -999, -999,-999,-999]
    ]
)
chunk,unsolved_pits_raster = breach_single_cell_pits_in_chunk(chunk,-999)
print(chunk,unsolved_pits_raster)