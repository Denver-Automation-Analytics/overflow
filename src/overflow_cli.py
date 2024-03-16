import click

from overflow.breach_single_cell_pits import breach_single_cell_pits
from overflow.generate_flow_direction_raster import flow_direction
from overflow.breach_paths_least_cost import breach_paths_least_cost
from overflow.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_RADIUS,
    DEFAULT_MAX_PITS,
)


@click.group()
def main():
    """The main entry point for the command line interface."""


@main.command(name="breach-single-cell-pits")
@click.option(
    "--input_file",
    help="path to the GDAL supported raster dataset for the DEM",
)
@click.option("--output_file", help="path to the output file (must be GeoTiff)")
@click.option("--chunk_size", help="chunk size", default=DEFAULT_CHUNK_SIZE)
def breach_single_cell_pits_cli(input_file: str, output_file: str, chunk_size: int):
    """
    This function is used to breach single cell pits in a DEM.
    The function takes filepath to a GDAL supported raster dataset as
    input and prodeces an output DEM with breached single cell pits.

    Parameters
    ----------
    input_file : str
        Path to the input dem file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing

    Returns
    -------
    None
    """
    try:

        breach_single_cell_pits(input_file, output_file, chunk_size)
    except Exception as exc:
        print(
            f"breach_single_cell_pits failed with the following exception: {str(exc)}"
        )
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies


@main.command(name="flow-direction")
@click.option(
    "--input_file",
    help="path to the DEM file",
)
@click.option("--output_file", help="path to the output file")
@click.option("--chunk_size", help="chunk size", default=DEFAULT_CHUNK_SIZE)
def flow_direction_cli(input_file: str, output_file: str, chunk_size: int):
    """
    This function is used to generate flow direction rasters from chunks of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with delineated flow direction.
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
        raise click.Abort()
    """
    pass


@main.command(name="breach-paths-least-cost")
@click.option(
    "--input_file",
    help="path to the GDAL supported raster dataset for the DEM",
)
@click.option("--output_file", help="path to the output file (must be GeoTiff)")
@click.option("--chunk_size", help="chunk size", default=DEFAULT_CHUNK_SIZE)
@click.option("--search_radius", help="search radius", default=DEFAULT_SEARCH_RADIUS)
@click.option("--max_pits", help="max pits", default=DEFAULT_MAX_PITS)
def breach_paths_least_cost_cli(
    input_file: str,
    output_file: str,
    chunk_size: int,
    search_radius: int,
    max_pits: int,
):
    """
    This function is used to breach paths of least cost for pits in a DEM.
    The function takes filepath to a GDAL supported raster dataset as
    input and prodeces an output DEM with breached paths of least cost.
    Only pits that can be solved within the search radius are solved.

    Parameters
    ----------
    input_file : str
        Path to the input dem file
    output_file : str
        Path to the output file
    chunk_size : int
        Size of the chunk to be used for processing. Larger chunk sizes will use more memory.
    search_radius : int
        Search radius in cells to look for solution paths. Larger search radius will use more memory.
    max_pits : int
        Maximum number of pits to solve at once. This is equivalent to the number of threads to use.
        Warning: each additional thread increases the memory usage.

    Returns
    -------
    None
    """
    try:
        breach_paths_least_cost(
            input_file, output_file, chunk_size, search_radius, max_pits
        )
    except Exception as exc:
        print(
            f"breach_paths_least_cost failed with the following exception: {str(exc)}"
        )
        # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
        raise click.Abort()
