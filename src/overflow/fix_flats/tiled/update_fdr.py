import os
from overflow.fix_flats.core.fix_flats import d8_masked_flow_dirs
from overflow.util.raster import open_dataset, raster_chunker, RasterChunk


def update_fdr(dem_band, fdr_band, fixed_fdr_band, working_dir, chunk_size):
    flat_mask_ds = open_dataset(os.path.join(working_dir, "flat_mask.tif"))
    flat_mask_band = flat_mask_ds.GetRasterBand(1)
    # update fdr using d8_masked_flow_dirs
    for fdr_tile in raster_chunker(fdr_band, chunk_size, 1):
        dem_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        dem_tile.read(dem_band)
        flat_mask_tile = RasterChunk(fdr_tile.row, fdr_tile.col, chunk_size, 1)
        flat_mask_tile.read(flat_mask_band)
        d8_masked_flow_dirs(dem_tile.data, flat_mask_tile.data, fdr_tile.data)
        fdr_tile.write(fixed_fdr_band)
    fixed_fdr_band.FlushCache()
