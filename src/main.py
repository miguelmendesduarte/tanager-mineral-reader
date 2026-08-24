"""Command line entry point."""

from typing import Annotated

import typer
from loguru import logger

from .catalog import build_client, download_assets, fetch_item
from .core.config import get_settings
from .core.logging import configure_logging
from .readers import Cube
from .spectra import (
    archive_size,
    convolve,
    deepest_feature,
    extra_blur,
    fetch_archive,
    read_spectra,
)

app = typer.Typer(
    help="Map surface mineralogy from Tanager hyperspectral imagery.",
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
LIBRARY_OVERWRITE_HELP = (
    "Download the spectral library again even if it is already present."
)
REFERENCE_SCENE_HELP = (
    "Scene whose bands the reference spectra are averaged onto. Defaults to "
    "the first scene configured in the settings."
)
SR_ASSET = "ortho_sr_hdf5"


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


@app.command()
def library(
    overwrite: Annotated[bool, typer.Option(help=LIBRARY_OVERWRITE_HELP)] = False,
) -> None:
    """Download the USGS spectral library the pixel spectra are matched against."""
    settings = get_settings()
    configure_logging(settings)

    with build_client(settings) as client:
        size = archive_size(
            settings.splib07_item_url,
            settings.splib07_archive_name,
            client=client,
        )
        path = fetch_archive(
            settings.splib07_url,
            settings.splib07_dir,
            archive_name=settings.splib07_archive_name,
            client=client,
            chunk_size=settings.download_chunk_size,
            expected_size=size,
            overwrite=overwrite,
        )

    logger.info("The spectral library is available at {}", path)


@app.command()
def references(
    scene_id: Annotated[str | None, typer.Option(help=REFERENCE_SCENE_HELP)] = None,
) -> None:
    """Average the configured reference spectra onto a scene's bands.

    Reports where each mineral's strongest absorption lands once averaged, so
    the result can be checked against the wavelengths the mineralogy
    literature puts them at.
    """
    settings = get_settings()
    configure_logging(settings)

    scene = scene_id or settings.scene_ids[0]
    low, high = settings.match_range

    with Cube(settings.scene_asset(scene, SR_ASSET)) as cube:
        bands = cube.wavelengths

    logger.info(
        "Averaging {} reference spectra onto the {} bands of {}",
        len(settings.species),
        len(bands),
        scene,
    )
    logger.info(
        "{:<16} {:<38} {:>9} {:>7} {:>10}",
        "group",
        "reference spectrum",
        "position",
        "depth",
        "extra blur",
    )

    for spectrum in read_spectra(settings.splib07_archive, settings.species):
        values = convolve(spectrum, bands)
        position, depth = deepest_feature(bands.centres, values, low, high)
        blur = extra_blur(spectrum, bands, low, high)
        logger.info(
            "{:<16} {:<38} {:>6.1f} nm {:>7.3f} {:>+7.2f} nm",
            settings.group_of(spectrum.name),
            spectrum.name,
            position,
            depth,
            blur,
        )


if __name__ == "__main__":
    app()
