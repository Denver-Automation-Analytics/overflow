import tempfile
import shutil
import os
from osgeo import gdal
from overflow.constants import DEFAULT_CHUNK_SIZE, FLOW_DIRECTION_NODATA
from overflow.util.raster import open_dataset, create_dataset
from overflow.fix_flats.tiled.create_global_state import create_global_state
from overflow.fix_flats.tiled.flat_mask import create_flat_mask
from overflow.fix_flats.tiled.update_fdr import update_fdr

LABELS_FILENAME = "labels.tif"
FLAT_MASK_FILENAME = "flat_mask.tif"


def setup_datasets(
    dem_filepath: str,
    fdr_filepath: str,
    fixed_fdr_filepath: str,
    working_dir: str,
):
    dem_ds = open_dataset(dem_filepath)
    fdr_ds = open_dataset(fdr_filepath)
    dem_band = dem_ds.GetRasterBand(1)
    x_size = dem_band.XSize
    y_size = dem_band.YSize
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()
    fixed_fdr_ds = create_dataset(
        fixed_fdr_filepath,
        FLOW_DIRECTION_NODATA,
        gdal.GDT_Byte,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    labels_ds = create_dataset(
        os.path.join(working_dir, LABELS_FILENAME),
        0,
        gdal.GDT_Int64,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    flat_mask_ds = create_dataset(
        os.path.join(working_dir, FLAT_MASK_FILENAME),
        0,
        gdal.GDT_Int64,
        x_size,
        y_size,
        geotransform,
        projection,
    )
    return (dem_ds, fdr_ds, fixed_fdr_ds, labels_ds, flat_mask_ds)


def fix_flats_tiled(
    dem_filepath: str,
    fdr_filepath: str,
    output_filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    working_dir: str = None,
):
    cleanup_working_dir = False
    if working_dir is None:
        working_dir = tempfile.mkdtemp()
        cleanup_working_dir = True
    dem_ds, fdr_ds, fixed_fdr_ds, labels_ds, flat_mask_ds = setup_datasets(
        dem_filepath, fdr_filepath, output_filepath, working_dir
    )
    dem_band = dem_ds.GetRasterBand(1)
    fdr_band = fdr_ds.GetRasterBand(1)
    fixed_fdr_band = fixed_fdr_ds.GetRasterBand(1)
    labels_band = labels_ds.GetRasterBand(1)
    flat_mask_band = flat_mask_ds.GetRasterBand(1)

    global_state = create_global_state(dem_band, fdr_band, labels_band, chunk_size)
    dist_to_low_edge_tiles, dist_to_high_edge_tiles = global_state.graph.solve_graph()
    create_flat_mask(
        chunk_size,
        flat_mask_band,
        labels_band,
        fdr_band,
        global_state,
        dist_to_high_edge_tiles,
        dist_to_low_edge_tiles,
    )
    update_fdr(dem_band, fdr_band, fixed_fdr_band, working_dir, chunk_size)
    if cleanup_working_dir:
        shutil.rmtree(working_dir)
    dem_ds = None
    fdr_ds = None
    fixed_fdr_ds = None
    labels_ds = None
    flat_mask_ds = None
