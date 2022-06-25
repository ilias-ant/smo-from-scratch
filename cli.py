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
def fit(c):
    """Perform a simple training of the SMO-based classifier, given a C."""
    if utils.dir_is_empty("data/"):
        click.echo(
            "[warning] 'data/' folder does not contain any data - consider running `sh datasets.sh` first."
        )
        return

    click.echo("- loading dataset...")
    x, y = utils.load_training_data()

    click.echo("- splitting into training, testing, development.")
    X_train, _, y_train, _ = train_test_split(x, y, test_size=0.18, random_state=SEED)

    click.echo(f"- shape of training design matrix: {X_train.shape}")
    click.echo(f"- shape of training labels: {y_train.shape}")

    print(f"- training SMO-based classifier for C={c} (may take a while ...)")
    smo = optimizer.SMO(C=c)
    _, b, w = smo.fit(X_train, y_train)

    click.echo(f"- b: {b}")
    click.echo(f"- w: {w}")


@cli.command()
def tune():
    """Perform a hyperparameter tuning of the SMO-based classifier."""
    if utils.dir_is_empty("data/"):
        click.echo(
            "[warning] 'data/' folder does not contain any data - consider running `sh datasets.sh` first."
        )
        return

    click.echo("- loading dataset...")
    x, y = utils.load_training_data()

    click.echo("- splitting into training, testing, development.")
    X_train, X_dev, y_train, y_dev = train_test_split(
        x, y, test_size=0.18, random_state=SEED
    )

    click.echo(f"- shape of training design matrix: {X_train.shape}")
    click.echo(f"- shape of training labels: {y_train.shape}")

    click.echo(f"- shape of development design matrix: {X_dev.shape}")
    click.echo(f"- shape of development labels: {y_dev.shape}")

    t = tuner.Tuner(C_range=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000])
    t.perform(train_data=(X_train, y_train), validation_data=(X_dev, y_dev))


if __name__ == "__main__":
    cli()
