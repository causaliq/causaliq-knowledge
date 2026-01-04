"""Command-line interface for causaliq-knowledge."""

import click


@click.command(name="causaliq-knowledge")
@click.version_option(version="0.1.0")
@click.argument(
    "name",
    metavar="NAME",
    required=True,
    nargs=1,
)
@click.option("--greet", default="Hello", help="Greeting to use")
def cli(name: str, greet: str) -> None:
    """
    Simple CLI example.

    NAME is the person to greet
    """
    click.echo(f"{greet}, {name}!")


def main() -> None:
    """Entry point for the CLI."""
    cli(prog_name="causaliq-knowledge (cqknow)")


if __name__ == "__main__":  # pragma: no cover
    main()
