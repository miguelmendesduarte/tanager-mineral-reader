# Tanager Mineral Reader

Works out which mineral is on the ground in each pixel of a Tanager-1
hyperspectral scene, by matching its spectrum against the USGS Spectral
Library. Pixels the data cannot support are left unnamed.

`docs/report.md` explains the method and the results.

## Prerequisites

- **uv**: this project uses `uv` to manage dependencies. Install it via the [official docs](https://docs.astral.sh/uv/getting-started/) if it's not already on your system.
- **Python**: Make sure you have a compatible version of Python installed. See `pyproject.toml` for the required version.
- **Disk**: about 8 GB (1 GB per scene and 5.5 GB for the spectral library).

## Installation

1. **Clone the repository** to your desired folder:

    ```bash
    git clone git@github.com:miguelmendesduarte/tanager-mineral-reader.git <desired-folder-name>
    ```

2. Navigate to the project folder:

    ```bash
    cd <desired-folder-name>
    ```

3. **Install** dependencies:

    ```bash
    uv sync --all-extras
    ```

4. (Optional) Install `pre-commit`:

    ```bash
    uv run pre-commit install
    ```

## Usage

Run these in order. Downloads are cached, so only the first run is slow.

```bash
uv run python -m src.main download    # the scenes
uv run python -m src.main library     # the reference spectra, once
uv run python -m src.main minerals    # the map
uv run python -m src.main figures     # the figures
```

Two more, for checking rather than producing:

```bash
uv run python -m src.main references  # where the reference spectra land
uv run python -m src.main agreement   # whether two dates of the same ground agree
```

Add `--help` to any command for its options.

Results go to `outputs/`: a GeoTIFF per scene and the figures.

## Configuration

Every option lives in `.env`. Copy `.env.example`, which documents each one.

To map somewhere else, set `SCENE_IDS` to the scenes covering it, from
[Planet's open Tanager catalog](https://www.planet.com/data/stac/browser/tanager-core-imagery/catalog.json).
