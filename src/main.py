"""Command line entry point."""

from typing import Annotated

import numpy as np
import typer
from loguru import logger

from .agreement import Agreement, by_confidence, compare
from .catalog import build_client, download_assets, fetch_item
from .core.config import Settings, get_settings
from .core.exceptions import AgreementError
from .core.logging import configure_logging
from .readers import Cube
from .spectra import (
    Mapped,
    archive_size,
    convolve,
    deepest_feature,
    extra_blur,
    fetch_archive,
    map_scene,
    read_spectra,
)
from .spectra import (
    convolve as convolve_spectrum,
)
from .writers import (
    agreement_matrix,
    mineral_map,
    spectrum_against_reference,
    write_map,
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
MAP_SCENE_HELP = "Scene to map. Defaults to the first scene configured in the settings."
AGREE_SCENE_HELP = (
    "Scene to compare; repeat the option. Defaults to every scene configured "
    "in the settings."
)
SR_ASSET = "ortho_sr_hdf5"
PAIR = 2
MINIMUM_EXAMPLE = 100


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
        len(settings.references),
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

    for spectrum in read_spectra(settings.splib07_archive, settings.references):
        values = convolve(spectrum, bands)
        position, depth = deepest_feature(bands.centres, values, low, high)
        blur = extra_blur(spectrum, bands, low, high)
        logger.info(
            "{:<16} {:<38} {:>6.1f} nm {:>7.3f} {:>+7.2f} nm",
            settings.group_of(spectrum.name) or "not a mineral",
            spectrum.name,
            position,
            depth,
            blur,
        )


@app.command()
def minerals(
    scene_id: Annotated[str | None, typer.Option(help=MAP_SCENE_HELP)] = None,
) -> None:
    """Work out which mineral, if any, each pixel of a scene shows."""
    settings = get_settings()
    configure_logging(settings)

    scene = scene_id or settings.scene_ids[0]
    mapped = map_scene(settings.scene_asset(scene, SR_ASSET), settings)
    write_map(mapped, mapped.grid, settings.map_path(scene))

    named = int(mapped.named.sum())
    logger.info(
        "{} of {} pixels of {} carry a mineral ({:.1f}%)",
        named,
        mapped.labels.size,
        scene,
        100 * named / mapped.labels.size,
    )


@app.command()
def agreement(
    scene_ids: Annotated[
        list[str] | None,
        typer.Option("--scene-id", "-s", help=AGREE_SCENE_HELP),
    ] = None,
) -> None:
    """Measure how far two maps of the same ground agree.

    Skipped when the scenes given are one observation delivered in pieces, or
    do not overlap, since neither case has anything to compare.
    """
    settings = get_settings()
    configure_logging(settings)

    scenes = tuple(scene_ids or settings.scene_ids)
    if len(scenes) < PAIR:
        logger.info("Skipped: {} scene given, and agreement needs two", len(scenes))
        return

    maps, strips = [], []
    for scene in scenes[:PAIR]:
        path = settings.scene_asset(scene, SR_ASSET)
        with Cube(path) as cube:
            strips.append(cube.strip_id)
        maps.append(map_scene(path, settings))

    try:
        result = compare(maps[0], maps[1], strips=(strips[0], strips[1]))
    except AgreementError as reason:
        logger.info("Skipped: {}", reason)
        return

    logger.info("Cohen's kappa {:.2f} over {} pixels", result.kappa, result.compared)
    _print_matrix(result, scenes[:PAIR])
    for group, share in sorted(result.per_group().items(), key=lambda item: -item[1]):
        logger.info("   {:18s} {:5.1f}%", group, 100 * share)
    for label, (rate, count) in by_confidence(maps[0], maps[1]).items():
        logger.info("   {:18s} {:5.1f}%  over {} pixels", label, 100 * rate, count)


def _print_matrix(result: Agreement, scenes: tuple[str, ...]) -> None:
    """Show the matrix itself, not only what was derived from it."""
    width = 13
    names = [name[: width - 2] for name in result.columns()]
    logger.info("{} down, {} across", scenes[0], scenes[1])
    logger.info(" " * 18 + "".join(f"{name:>{width}}" for name in [*names, "total"]))
    for group, counts, total in result.rows():
        cells = "".join(f"{count:>{width},}" for count in [*counts, total])
        logger.info("{:<18}{}", group[: width + 3], cells)


@app.command()
def figures() -> None:
    """Draw the results: the maps, the agreement, and a worked example."""
    settings = get_settings()
    configure_logging(settings)

    maps, strips = [], []
    for scene in settings.scene_ids:
        path = settings.scene_asset(scene, SR_ASSET)
        with Cube(path) as cube:
            strips.append(cube.strip_id)
        mapped = map_scene(path, settings)
        maps.append(mapped)
        mineral_map(mapped, settings.output_dir / f"map_{scene}.png", scene)

    _draw_examples(settings, maps[0], settings.scene_ids[0])

    if len(maps) >= PAIR:
        result = compare(maps[0], maps[1], strips=(strips[0], strips[1]))
        agreement_matrix(
            result,
            settings.output_dir / "agreement.png",
            (settings.scene_ids[0], settings.scene_ids[1]),
        )

    logger.info("Figures are in {}", settings.output_dir)


def _draw_examples(settings: Settings, mapped: Mapped, scene: str) -> None:
    """Draw a typical settled pixel of each mineral against its reference.

    Typical means the single pixel whose absorption depth sits closest to the
    median of its group — one real spectrum, not an average of many. The
    deepest pixel in a scene is an outlier and flatters the method; the middle
    one is what the map is mostly made of.
    """
    low, high = settings.match_range

    with Cube(settings.scene_asset(scene, SR_ASSET)) as cube:
        bands = cube.wavelengths
        selected = cube.bands_between(low, high)
        wavelengths = bands.centres[selected]

        for index, group in enumerate(mapped.groups):
            chosen = mapped.named & mapped.resolved & (mapped.labels == index)
            if chosen.sum() < MINIMUM_EXAMPLE:
                continue
            depths = np.where(chosen, mapped.depth, np.nan)
            middle = float(np.nanmedian(depths[chosen]))
            row, column = np.unravel_index(
                int(np.nanargmin(np.abs(depths - middle))), depths.shape
            )

            name = next(n for n in settings.species if settings.group_of(n) == group)
            spectrum = read_spectra(settings.splib07_archive, [name])[0]
            spectrum_against_reference(
                wavelengths,
                np.asarray(
                    cube.read_spectrum(int(row), int(column))[selected], np.float64
                ),
                convolve_spectrum(spectrum, bands)[selected],
                settings.output_dir / f"spectrum_{group}.png",
                f"A typical {group.replace('_', ' ')} pixel, and {name.split('_')[0]}",
                reference_label=name.split("_")[0].lower(),
                caption=(
                    f"Median-depth pixel of {int(chosen.sum()):,} settled "
                    f"{group.replace('_', ' ')} pixels in {scene}. "
                    f"Reference {name}."
                ),
            )


if __name__ == "__main__":
    app()
