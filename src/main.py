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

SCENE_ID_HELP = (
    "Identifier of a scene in the configured collection; repeat the option to "
    "download several. Defaults to the scenes configured in the settings."
)
ITEM_URL_HELP = (
    "Full URL of a STAC item, for scenes that live outside the configured "
    "catalog and collection; repeat the option for several. Takes precedence "
    "over --scene-id."
)
ASSET_HELP = (
    "Catalog key of an asset to download; repeat the option to select several. "
    "Defaults to the assets configured in the settings."
)
OVERWRITE_HELP = "Download assets again even if they are already present."


@app.command()
def download(
    scene_ids: Annotated[
        list[str] | None,
        typer.Option("--scene-id", "-s", help=SCENE_ID_HELP),
    ] = None,
    item_urls: Annotated[
        list[str] | None,
        typer.Option("--item-url", help=ITEM_URL_HELP),
    ] = None,
    assets: Annotated[
        list[str] | None,
        typer.Option("--asset", "-a", help=ASSET_HELP),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help=OVERWRITE_HELP)] = False,
) -> None:
    """Download the assets of the scenes under study."""
    settings = get_settings()
    configure_logging(settings)

    urls = item_urls or [
        settings.item_url(scene_id) for scene_id in scene_ids or settings.scene_ids
    ]

    with build_client(settings) as client:
        for url in urls:
            item = fetch_item(url, client=client)
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
