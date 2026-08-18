"""Command line entry point."""

from typing import Annotated

import typer
from loguru import logger

from .catalog import build_client, download_assets, fetch_item
from .core.config import get_settings
from .core.logging import configure_logging

app = typer.Typer(
    help="Work with Tanager scenes from Planet's open STAC catalog.",
    add_completion=False,
)

SCENE_ID_HELP = "Identifier of another scene in the configured collection."
ITEM_URL_HELP = (
    "Full URL of a STAC item, for scenes that live outside the configured "
    "catalog and collection. Takes precedence over --scene-id."
)
ASSET_HELP = (
    "Catalog key of an asset to download; repeat the option to select several. "
    "Defaults to the assets configured in the settings."
)
OVERWRITE_HELP = "Download assets again even if they are already present."


@app.command()
def download(
    scene_id: Annotated[str | None, typer.Option(help=SCENE_ID_HELP)] = None,
    item_url: Annotated[str | None, typer.Option(help=ITEM_URL_HELP)] = None,
    assets: Annotated[
        list[str] | None,
        typer.Option("--asset", "-a", help=ASSET_HELP),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help=OVERWRITE_HELP)] = False,
) -> None:
    """Download the assets of the scene under study."""
    settings = get_settings()
    configure_logging(settings)

    if scene_id is not None:
        settings = settings.model_copy(update={"scene_id": scene_id})

    with build_client(settings) as client:
        item = fetch_item(item_url or settings.item_url, client=client)
        paths = download_assets(
            item,
            assets or settings.default_assets,
            destination_dir=settings.data_dir,
            client=client,
            chunk_size=settings.download_chunk_size,
            overwrite=overwrite,
        )

    for name, path in paths.items():
        logger.info("{} is available at {}", name, path)


if __name__ == "__main__":
    app()
