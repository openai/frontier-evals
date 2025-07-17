"""
Stub Firebase-related endpoints so tests run deterministically.

Exports
-------
install_online_guard_async  – Playwright **async** API
install_online_guard_sync   – Playwright **sync** API
"""

from __future__ import annotations

import json
import logging
import warnings
import sys
import traceback
from typing import Optional, TYPE_CHECKING, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid importing Playwright at runtime
if TYPE_CHECKING:
    from playwright.async_api import BrowserContext as AsyncBrowserContext, Page as AsyncPage, Route as AsyncRoute, Request as AsyncRequest
    from playwright.sync_api import BrowserContext as SyncBrowserContext, Page as SyncPage, Route as SyncRoute, Request as SyncRequest

# --------------------------------------------------------------------------- #
# Dummy payloads returned by Firebase endpoints
# --------------------------------------------------------------------------- #
_FAKE_INSTALLATIONS = json.dumps(
    {
        "fid": "CI_FAKE_FID",
        "refreshToken": "CI_FAKE_REFRESH",
        "authToken": {"token": "CI_FAKE_TOKEN", "expiresIn": "604800s"},
    }
)
_FAKE_REMOTECONFIG = json.dumps({"entries": {}})
_FAKE_APPCHECK = json.dumps({"token": "CI_FAKE_APP_CHECK", "ttl": "3600s"})
_FAKE_FIREBASE_LOG = "{}"


def _mock_body(url: str) -> Optional[str]:
    """Return mock response body based on URL pattern."""
    try:
        logger.debug(f"Mocking response for URL: {url}")
        if not url:
            logger.warning("Empty URL provided to _mock_body")
            return None

        url_lower = url.lower()

        if "firebaseinstallations" in url_lower:
            logger.debug("Returning fake installations data")
            return _FAKE_INSTALLATIONS

        if "firebaseremoteconfig" in url_lower:
            logger.debug("Returning fake remote config data")
            return _FAKE_REMOTECONFIG

        if "firebaseappcheck" in url_lower:
            logger.debug("Returning fake app check data")
            return _FAKE_APPCHECK

        if any(x in url_lower for x in ["firebaselogging", "firebaseperformance"]):
            logger.debug("Returning fake firebase log")
            return _FAKE_FIREBASE_LOG

        logger.debug(f"No mock data for URL: {url}")
        return None

    except Exception as e:
        logger.error(f"Error in _mock_body: {e}\n{traceback.format_exc()}")
        return None


# --------------------------------------------------------------------------- #
# Async Playwright helper
# --------------------------------------------------------------------------- #
async def install_online_guard_async(
    context: Any, page: Any  # Using Any to avoid type issues during import
) -> None:
    """Attach a route that fulfils Firebase requests with dummy JSON."""
    try:
        from playwright.async_api import Route as AsyncRoute, Request as AsyncRequest
    except ImportError as e:
        warnings.warn(f"Failed to import Playwright async API: {e}")
        return

    async def _router(route: AsyncRoute, request: AsyncRequest) -> None:
        body = _mock_body(request.url)
        if body is not None:
            logger.debug("🛡  stubbed %s", request.url)
            await route.fulfill(status=200, content_type="application/json", body=body)
        else:
            await route.continue_()

    await context.route("**/*", _router)


# --------------------------------------------------------------------------- #
# Sync Playwright helper
# --------------------------------------------------------------------------- #
def install_online_guard_sync(
    context: Any, page: Any  # Using Any to avoid type issues during import
) -> None:
    """Attach a route that fulfils Firebase requests with dummy JSON."""
    try:
        from playwright.sync_api import Route as SyncRoute, Request as SyncRequest
    except ImportError as e:
        warnings.warn(f"Failed to import Playwright sync API: {e}")
        return

    def _router(route: SyncRoute, request: SyncRequest) -> None:
        body = _mock_body(request.url)
        if body is not None:
            logger.debug("🛡  stubbed %s", request.url)
            route.fulfill(status=200, content_type="application/json", body=body)
        else:
            route.continue_()

    context.route("**/*", _router)


__all__ = ["install_online_guard_async", "install_online_guard_sync"]

# """
# Stub Firebase-related endpoints so tests run deterministically.
