import click
from sklearn.model_selection import train_test_split

from src import optimizer, tuner, utils

SEED = 1234567


@click.group()
def cli():
    pass


@cli.command()
def fit():
    """Training action."""
    if utils.dir_is_empty("data/"):
        click.echo(
            "[warning] 'data/' folder does not contain any data - consider running `sh datasets.sh` first."
        )
        return

    click.echo("- loading dataset...")
    x, y = utils.manual_load_training_data()

    X_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=SEED)
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, test_size=0.375, random_state=SEED
    )

    click.echo(f"- shape of design matrix: {X_train.shape}")
    click.echo(f"- shape of labels: {y_train.shape}")

    C = 0.5
    click.echo(f"- using C: {C}")
    smo = optimizer.SMO(C=C)
    alpha, b, w = smo.fit(X_train, y_train)

    print(f"alpha: {alpha}")
    print(f"b: {b}")
    print(f"w: {w}")


@cli.command()
def tune():
    """Tuning action."""
    if utils.dir_is_empty("data/"):
        click.echo(
            "[warning] 'data/' folder does not contain any data - consider running `sh datasets.sh` first."
        )
        return

    click.echo("- loading dataset...")
    x, y = utils.manual_load_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=0.375, random_state=SEED
    )

    t = tuner.Tuner(C_range=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10])
    t.perform(train_data=(X_train, y_train), validation_data=(X_dev, y_dev))


if __name__ == "__main__":
    cli()
