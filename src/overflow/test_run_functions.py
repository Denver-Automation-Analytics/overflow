from breach_single_cell_pits import breach_single_cell_pits
from generate_flow_direction_raster import flow_direction_from_chunks


breach_single_cell_pits("/workspaces/overflow/data/USGS_1M_13_x49y442_CO_DRCOG_2020_B20.tif","/workspaces/overflow/data/breached.tif",chunk_size=1000)
flow_direction_from_chunks("/workspaces/overflow/data/breached.tif","/workspaces/overflow/data/FDIR.tif",chunk_size=1000)