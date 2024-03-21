from osgeo import gdal
from contextlib import contextmanager


gdal.UseExceptions()


@contextmanager
def setup_bands(input_path, output_path, eType=gdal.GDT_Float32, nodata_value=None):
    n_bands = 1
    input_raster = gdal.Open(input_path)
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(
        output_path,
        input_raster.RasterXSize,
        input_raster.RasterYSize,
        n_bands,
        eType,
    )

    projection = input_raster.GetProjection()
    output_dataset.SetProjection(projection)
    transform = input_raster.GetGeoTransform()
    output_dataset.SetGeoTransform(transform)

    band_id = 1
    input_band = input_raster.GetRasterBand(band_id)
    output_band = output_dataset.GetRasterBand(band_id)
    input_no_data_value = input_band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = input_no_data_value
    output_band.SetNoDataValue(nodata_value)

    yield input_band, output_band, nodata_value
