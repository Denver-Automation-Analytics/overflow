import pytest
import numpy as np
from osgeo import gdal
from overflow.breach_single_cell_pits import breach_single_cell_pits_in_chunk,breach_single_cell_pits


@pytest.fixture
def raster_file_path():
    """Create a random raster band for testing.

    Yields:
        gdal.Band: A raster band of size 100x100 with random float32 data.
    """
    output_path="/vsimem/test_raster_breach.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, 100, 100, 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    array  = np.array(
    [   
        [100, 101, 90, 97, 90],
        [103, 102, 80, 96, 95],
        [94, 95, 96, 95, 94],
        [97, 98, 95, 94, 90],
        [95, 90, 85, 40, 92],
    ]
    )
    band.WriteArray(array)
    band.SetNoDataValue(-9999)
    dataset.FlushCache()
    dataset=None
    yield output_path
    gdal.Unlink(output_path)


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
            
def test_breach_single_cell_pits(raster_file_path):
    dx=[1,1,1,0,-1,-1,-1,0]
    dy=[-1,0,1,1,1,0,-1,-1]
    dx2=[2,2,2,2,2,1,0,-1,-2,-2,-2,-2,-2,-1,0,1]
    dy2=[-2,-1,0,1,2,2,2,2,2,1,0,-1,-2,-2,-2,-2]
    results_path="/vsimem/test_breach_single_cell_pits.tif"
    breach_single_cell_pits(raster_file_path,results_path,chunk_size=5)
    result=gdal.Open(results_path)
    band = result.GetRasterBand(1)
    nodata_value=band.GetNoDataValue()
    band = result.ReadAsArray(0, 0, result.RasterXSize, result.RasterYSize) 
    rows= result.RasterYSize
    cols= result.RasterXSize
    for row in range(2,rows-2):
        for col in range(2,cols-2):
            z=band [row,col]
            if z != nodata_value:
                for k in range(8):
                    if int(row+dy[k]) < 0 or int(col+dx[k]) < 0 :
                        pass
                    else:
                        zn=band [row+dy[k],col+dx[k]]
                        if zn < z and zn != nodata_value:
                                flag = False   
                if flag:
                    for k in range(16):
                        zn=band[row+dy2[k],col+dx2[k]]
                        assert zn < z and zn != nodata_value

    
    