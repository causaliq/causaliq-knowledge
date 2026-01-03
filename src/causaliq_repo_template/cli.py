"""Command-line interface for causaliq-repo-template."""

import click


@click.command(name="causaliq-repo-template")
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
    cli(prog_name="causaliq-repo-template (cqcrt)")


if __name__ == "__main__":  # pragma: no cover
    main()
