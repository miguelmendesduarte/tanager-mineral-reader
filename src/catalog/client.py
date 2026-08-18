"""Access to Planet's open STAC catalog."""

import httpx
from loguru import logger

from ..core.config import Settings
from ..core.exceptions import ItemFetchError
from .models import Item


def build_client(settings: Settings) -> httpx.Client:
    """Create the HTTP client used for catalog and asset requests.

    Args:
        settings: Application settings.

    Returns:
        httpx.Client: A client that follows redirects, as the asset hrefs point
            at a storage bucket that redirects.
    """
    return httpx.Client(timeout=settings.request_timeout, follow_redirects=True)


def fetch_item(url: str, *, client: httpx.Client) -> Item:
    """Fetch a STAC item and parse the metadata we care about.

    Args:
        url: URL of the item JSON.
        client: HTTP client used for the request.

    Returns:
        Item: The parsed item.

    Raises:
        ItemFetchError: If the item cannot be retrieved.
    """
    logger.info("Fetching STAC item from {}", url)

    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ItemFetchError(url) from error

    item = Item.model_validate(response.json())
    logger.info(
        "Item {} acquired {} over {} exposes {} assets",
        item.id,
        item.acquired_at,
        item.location,
        len(item.assets),
    )
    return item
