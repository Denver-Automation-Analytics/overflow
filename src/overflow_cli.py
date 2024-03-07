
import click

from overflow.breach_single_cell_pits import breach_single_cell_pits as breach_single_cell_pits_impl


@click.group()
def main():
    """The main entry point for the command line interface."""


@main.command()
@click.option(
    "--input_file",
    help="path to the DEM file",
)
@click.option("--output_file", help="path to the output file")
@click.option("--chunk_size", help="chunk size", default=1000)
def breach_single_cell_pits(input_file: str, output_file: str, chunk_size: int):
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
        breach_single_cell_pits_impl(input_file, output_file, chunk_size)
    except Exception as exc:
        print(
            f"breach_single_cell_pits failed with the following exception: {str(exc)}"
        )
        raise click.Abort()  # exit with non-zero exit code. Everytime zero is returned on failure a baby kitten dies