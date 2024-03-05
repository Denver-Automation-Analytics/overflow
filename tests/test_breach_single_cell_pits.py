import pytest
import numpy as np
from overflow.breach_pits import breach_single_cell_pits_in_chunk



def test_breach_cingle_cell_pits_in_chunk():
    dx=[1,1,1,0,-1,-1,-1,0]
    dy=[-1,0,1,1,1,0,-1,-1]
    dx2=[2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1]
    dy2=[-2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2]
    
    chunk = np.array(
    [   [-999,-999,-999, -999, -999, -999, -999,-999,-999],
        [-999,-999,-999, -999, -999, -999, -999,-999,-999],
        [-999,-999,100, 101, 90, 97, 90,-999,-999],
        [-999, -999,103, 102, 80, 96, 95,-999,-999],
        [-999, -999,94, 95, 96, 95, 94,-999,-999],
        [-999, -999,97, 98, 95, 94, 90,-999,-999],
        [-999, -999,95, 90, 85, 40, 92,-999,-999],
        [-999,-999,-999, -999, -999, -999, -999,-999,-999],
        [-999,-999,-999, -999, -999, -999, -999,-999,-999]
    ]
    )
    nodata_value=-999
    breach,unsolved_pits=breach_single_cell_pits_in_chunk(chunk,nodata_value)
    rows, cols = breach.shape
    # Loop through each cell in the chunk
    for row in range(2,rows-2):
        for col in range(2,cols-2):
            z=breach[row,col]
            if z != nodata_value:
                for k in range(8):
                    if int(row+dy[k]) < 0 or int(col+dx[k]) < 0 :
                        pass
                    else:
                        zn=breach[row+dy[k],col+dx[k]]
                        if zn < z and zn != nodata_value:
                                flag = False   
                if flag:
                    for k in range(16):
                        zn=breach[row+dy2[k],col+dx2[k]]
                        assert zn < z and zn != nodata_value and unsolved_pits[row,col]!=1
            
                        
    
    