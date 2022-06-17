import click


@click.command()
@click.option("--filepath", help="The filepath containing the dataset.")
def run(filepath: str) -> None:
    """Runs the SMO algorithm"""
    click.echo(f"Running SMO ...")


if __name__ == "__main__":
    run()
