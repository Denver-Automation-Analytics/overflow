import numpy as np


def breach_cingle_cell_pits(chunk: np.ndarray) -> np.ndarray:
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
    for row in range(rows):
        for col in range(cols):
            z=chunk[row,col]
            if z != -9999:
                flag = True
                try:
                    for k in range(8):
                        zn=chunk[row+dy[k],col+dx[k]]
                        if zn < z and zn != -9999:
                            flag = False
                            break
                    if flag:
                        print("Pit found at: ",row,col)
                        for k in range(16):
                            zn=chunk[row+dy2[k],col+dx2[k]]
                            if zn < z and zn != -9999:
                                chunk_copy[row+dy[breachcell[k]],col+dx[breachcell[k]]]=(z+zn)/2
                except Exception as e:
                    print(e)
                            
            
    return chunk_copy


if __name__ == "__main__":
    chunk = breach_cingle_cell_pits(chunk)