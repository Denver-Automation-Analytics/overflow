import click

from overflow.breach_single_cell_pits import breach_single_cell_pits
from overflow.fill_pits import depression_fill


@click.group()
def main():
    """The main entry point for the command line interface."""


@main.command(name="breach-single-cell-pits")
@click.option(
    "--input_file",
    help="path to the DEM file",
)
@click.option("--output_file", help="path to the output file")
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
        breach_single_cell_pits(input_file, output_file, chunk_size=2000)
    except Exception as exc:
        print(
            f"breach_single_cell_pits failed with the following exception: {str(exc)}"
        )
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies


@main.command(name="depression-fill")
@click.option(
    "--input_file",
    help="path to the DEM file",
)
@click.option("--output_file", help="path to the output file")
def flood_fill_cli(input_file: str, output_file: str, chunk_size: int = 2000):
    """

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
        depression_fill(input_file, output_file, chunk_size)
    except Exception as exc:
        print(
            f"breach_single_cell_pits failed with the following exception: {str(exc)}"
        )
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies
