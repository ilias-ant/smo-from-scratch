import click
from sklearn.model_selection import train_test_split

from src import optimizer, tuner, utils

SEED = 1234567


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--C",
    default=1.0,
    help="Regularization parameter. "
    "The strength of the regularization is inversely proportional to C - defaults to 1.0",
)
@click.option(
    "--filepath", default="data/gisette_scale", help="The relative data filepath."
)
def fit(c, filepath):
    """Perform a simple training of the SMO-based classifier, given a C."""
    if utils.dir_is_empty("data/"):
        click.echo(
            "[warning] 'data/' folder does not contain any data - consider running `sh datasets.sh` first."
        )
        return

    click.echo("- loading dataset...")
    x, y = utils.load_data(filepath)

    click.echo("- splitting into training, testing, development.")
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.24, random_state=SEED
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_test, y_test, test_size=0.5, random_state=SEED
    )

    click.echo(f"- shape of training design matrix: {X_train.shape}")
    click.echo(f"- shape of training labels: {y_train.shape}")

    click.echo(f"- shape of testing design matrix: {X_test.shape}")
    click.echo(f"- shape of testing labels: {y_test.shape}")

    click.echo(f"- shape of development design matrix: {X_dev.shape}")
    click.echo(f"- shape of development labels: {y_dev.shape}")

    print(f"- training SMO-based classifier for C={c} (may take a while ...)")
    smo = optimizer.SMO(C=c)

    b, w = smo.fit(X_train, y_train)

    utils.write_to_file(b=b, w=w)
    click.echo("- estimated parameters have been saved to files: estimated_*.txt")


@cli.command()
@click.option(
    "--filepath", default="data/gisette_scale", help="The relative data filepath."
)
def tune(filepath):
    """Perform a hyperparameter tuning of the SMO-based classifier."""
    if utils.dir_is_empty("data/"):
        click.echo(
            "[warning] 'data/' folder does not contain any data - consider running `sh datasets.sh` first."
        )
        return

    click.echo("- loading dataset...")
    x, y = utils.load_data(filepath)

    click.echo("- splitting into training, testing, development.")
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.24, random_state=SEED
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_test, y_test, test_size=0.5, random_state=SEED
    )

    click.echo(f"- shape of training design matrix: {X_train.shape}")
    click.echo(f"- shape of training labels: {y_train.shape}")

    click.echo(f"- shape of testing design matrix: {X_test.shape}")
    click.echo(f"- shape of testing labels: {y_test.shape}")

    click.echo(f"- shape of development design matrix: {X_dev.shape}")
    click.echo(f"- shape of development labels: {y_dev.shape}")

    t = tuner.Tuner(C_range=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
    best_hparam_set = t.perform(
        train_data=(X_train, y_train), validation_data=(X_dev, y_dev)
    )

    click.echo(f"- best hparam set: {best_hparam_set}")


if __name__ == "__main__":
    cli()
