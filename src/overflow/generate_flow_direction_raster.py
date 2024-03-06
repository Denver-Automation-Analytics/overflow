import numpy as np
from numba import njit, prange
from osgeo import gdal
from breach_single_cell_pits import breach_single_cell_pits_in_chunk
from util.raster import raster_chunker

@njit(parallel=True)
def generate_flow_direction_raster(chunk,cell_size,nodata_value)-> tuple[np.ndarray,np.ndarray] :
    """
    This function is used to calculate flow direction in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with flow direction values.

    Parameters
    ----------
    chunk : np.ndarray
        A chunk of a DEM.

    Returns
    -------
    np.ndarray
        A chunk of a DEM with flow direction values.
    """
    D8_Directions_Dict={0:1,1:2,2:4,3:8,4:16,5:32,6:64,7:128}
    dx=[1,1,1,0,-1,-1,-1,0]
    dy=[-1,0,1,1,1,0,-1,-1]

    chunk_copy = chunk.copy()
    # Get the shape of the chunk
    rows, cols = chunk.shape
    # Loop through each cell in the chunk
    
    # pylint: disable=not-an-iterable
    for row in range(2,rows-2):
        
        for col in range(2,cols-2):
            z=chunk[row,col]
            if z != nodata_value:
                slopes=[]
                for k in range(8):
                    if chunk[row+dy[k],col+dx[k]] != nodata_value:
                        slopes.append((z - chunk[row+dy[k],col+dx[k]])/cell_size)
                    
            m=max(slopes)
            #print("slope:",m)
            index_max=slopes.index(m)
            #print("Max index:",index_max)
            chunk_copy[row,col]=D8_Directions_Dict[index_max]
                 
    return chunk_copy

def flow_direction_from_chunks(input_path,output_path,chunk_size=1000):
    input_raster = gdal.Open(input_path)
    cell_size=1
    
    projection = input_raster.GetProjection()
    transform = input_raster.GetGeoTransform()
    
    band = input_raster.GetRasterBand(1)
    nodata_value=band.GetNoDataValue()
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, input_raster.RasterYSize,input_raster.RasterXSize, 1, gdal.GDT_Float32)
    
    dataset.SetProjection(projection)
    dataset.SetGeoTransform(transform)
    output_band=dataset.GetRasterBand(1)
    
    for chunk in raster_chunker(band,chunk_size=chunk_size,chunk_buffer_size=2):
        result= generate_flow_direction_raster(chunk.data,cell_size,nodata_value)
        chunk.from_numpy(result)
        chunk.write(output_band) 
    
        

flow_direction_from_chunks("/workspaces/overflow/data/USGS_1M_13_x49y442_CO_DRCOG_2020_B20.tif","/workspaces/overflow/data/testFDIR5.tif",chunk_size=1000)