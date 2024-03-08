

import click

from overflow.breach_single_cell_pits import breach_single_cell_pits
from overflow.generate_flow_direction_raster import flow_direction_from_chunks



@click.group()
def main():
    """The main entry point for the command line interface."""



@main.command(name="breach-single-cell-pits")

@click.option(
    "--input_file",
    help="path to the DEM file",
)
@click.option("--output_file", help="path to the output file")

@click.option("--chunk_size", help="chunk size", default=1000)

def breach_single_cell_pits_cli(input_file: str, output_file: str, chunk_size: int):

    """
    This function is used to breach single cell pits in a chunk of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with breached single cell pits.

    Parameters
    ----------
    input_file : str
        Path to the input dem file
    output_file : str
        Path to the output file

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

@main.command(name="generate-flow-direction-raster")
@click.option(
    "--input_file",
    help="path to the DEM file",
)
@click.option("--output_file", help="path to the output file")
@click.option("--chunk_size", help="chunk size", default=1000)

def flow_direction_from_chunks_cli(input_file: str, output_file: str, chunk_size: int):
    """
    This function is used to generate flow direction rasters from chunks of a DEM.
    The function takes a chunk of a DEM as input and returns a chunk of DEM with delineated flow direction.

    Parameters
    ----------
    input_file : str
        Path to the input dem file
    output_file : str
        Path to the output file

    Returns
    -------
    None
    """
    try:
        flow_direction_from_chunks(input_file, output_file, chunk_size)
    except Exception as exc:
        print(
            f"flow_direction_from_chunks failed with the following exception: {str(exc)}"

        )
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
