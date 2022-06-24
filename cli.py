import click

from src import optimizer, utils


@click.group()
def cli():
    pass


@cli.command()
def fit():
    """Training action."""
    if utils.dir_is_empty("data/"):
        click.echo(
            "[warning] 'data/' folder does not contain any data - please do add them there and re-run command."
        )
        return

    x, y = utils.load_training_data()

    smo = optimizer.SMO(C=1)

    smo.fit(x, y)


if __name__ == "__main__":
    cli()
