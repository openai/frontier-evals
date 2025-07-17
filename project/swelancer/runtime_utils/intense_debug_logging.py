"""
Intense debug‑logging utilities for Playwright tests.

Exports
-------
install_sniffer_async          – Network sniffer for Playwright **async** API
attach_debug_listeners_async   – JS‑console/network listeners for Playwright **async** API
install_sniffer_sync           – Network sniffer for Playwright **sync** API
attach_debug_listeners_sync    – JS‑console/network listeners for Playwright **sync** API
GOOD_REQS                      – List[str] of successful requests during bootstrap window
BAD_REQS                       – List[str] of failed/erroneous requests during bootstrap window

Usage example
-------------
```python
from utils.intense_debug_logging import (
    install_sniffer_async,
    attach_debug_listeners_async,
    GOOD_REQS,
    BAD_REQS,
)

await install_sniffer_async(context, page)
attach_debug_listeners_async(page)

try:
    await start_chat_fab.click()
except TimeoutError:
    LOGGER.info("==== first 5‑second GOOD URLs -- FAB FAILED ====")
    for url in GOOD_REQS:
        LOGGER.info(url)
    LOGGER.info("==== BAD / failed URLs -- FAB FAILED ====")
    for url in BAD_REQS:
        LOGGER.info(url)
    raise
```
"""
from __future__ import annotations

import datetime
import hashlib
import itertools
import json
import logging
import sys
import textwrap
import time
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported *only* for static type‑checking – not at runtime.
    from playwright.async_api import Page as AsyncPage, BrowserContext as AsyncBrowserContext
    from playwright.sync_api import Page as SyncPage, BrowserContext as SyncBrowserContext

# --------------------------------------------------------------------------- #
# Logger configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public globals – collected during bootstrap
# --------------------------------------------------------------------------- #
GOOD_REQS: List[str] = []
BAD_REQS: List[str] = []

_BOOT_T0: float = 0.0
_BOOT_WIN: int = 5  # seconds

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _log(label: str, obj: Any, max_len: int = 800) -> None:
    """Emit JSON‑serialised payload trimmed to *max_len* chars."""
    try:
        payload = json.dumps(obj, default=str, separators=(",", ":"))
    except Exception as exc:  # pragma: no cover
        LOGGER.error("⚠️  %s <non‑serialisable: %s>", label, exc)
    else:
        LOGGER.error("%s %s", label, payload[:max_len])


async def _grab_dom(page) -> str:
    html = await page.evaluate(
        """() => {
            const clone = document.documentElement.cloneNode(true);
            clone.querySelectorAll('[data-testid],[data-test-id],[aria-label]').forEach(e=>{
                e.removeAttribute('data-testid');
                e.removeAttribute('data-test-id');
            });
            return clone.outerHTML.replace(/\\s+/g,' ');
        }"""
    )
    return html[:90000]


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]


async def _dump_state(page, stage: str, net_log: List[dict]) -> None:
    banners = await page.evaluate("""() => ({
        offline: !!document.querySelector('[data-testid="connectionBanner"]'),
        modal  : !!document.querySelector('[role="dialog"],[data-testid*="Modal"]'),
        loader : !!document.querySelector('[data-testid*="Loader"],.spinner,.loading'),
        toast  : !!document.querySelector('[role="alert"]')
    })""")
    _log(f"🔎[{stage}]banners", banners)

    recent_errors = list(itertools.islice(net_log, 0, 10))
    _log(f"🌐[{stage}]netErrs", recent_errors)

    dom_meta = {"hash": _hash(await _grab_dom(page)), "len": len(await page.content())}
    _log(f"📄[{stage}]dom", dom_meta)

    cx, cy = await page.evaluate("() => [innerWidth-40, innerHeight-40]")
    top_el = await page.evaluate(
        "([x,y]) => (document.elementFromPoint(x,y)?.outerHTML||'').slice(0,120)",
        [cx, cy],
    )
    _log(f"🧩[{stage}]topEl", top_el)

    _log(f"⏰[{stage}]utc", datetime.datetime.utcnow().isoformat())


async def dump_fab_debug(page, fab, stage: str) -> None:
    """Diagnostic helper – surfaces FAB element state."""
    try:
        bb = await fab.bounding_box() or {}
        cx = bb.get("x", 0) + bb.get("width", 0) / 2
        cy = bb.get("y", 0) + bb.get("height", 0) / 2
        top_el = await page.evaluate(
            "([x,y]) => (document.elementFromPoint(x,y)?.outerHTML||'').slice(0,150)",
            [cx, cy],
        )
        payload = {
            "visible": await fab.is_visible(),
            "enabled": await fab.is_enabled(),
            "pointer": await fab.evaluate("e=>getComputedStyle(e).pointerEvents"),
            "bbox": bb,
            "topEl": top_el,
        }
        _log(f"🕵️[{stage}]fab", payload)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("⚠️ dump_fab_debug failed: %s", exc)


async def debug_dump(page, title: str = "DOM-DUMP") -> None:
    """Light‑weight DOM dump helper for post‑mortem debugging."""
    try:
        LOGGER.info("\n" + "==== %s ====".center(60, "=") , title)
        LOGGER.info(textwrap.shorten(await page.content(), width=5000, placeholder="…truncated…"))
        LOGGER.info("URL was ➜ %s", page.url)
    except Exception as exc:
        LOGGER.error("Failed to dump DOM: %s", exc)


def emit_bootstrap_urls() -> None:
    """Print captured GOOD/BAD URLs for the bootstrap window."""
    _log(f"====GOOD(<{_BOOT_WIN}s)", GOOD_REQS)
    _log(f"====BAD (<{_BOOT_WIN}s)", BAD_REQS)

# --------------------------------------------------------------------------- #
# Listener helpers (console, JS errors, network failures)
# --------------------------------------------------------------------------- #

def _attach_debug_listeners(page):
    page.on("console", lambda m: LOGGER.error("🖥️  %s › %s", m.type, m.text))
    page.on("pageerror", lambda e: LOGGER.error("💥 JS exception › %s", e))
    page.on(
        "response",
        lambda r: r.status >= 400 and LOGGER.error("🌐 %s %s", r.status, r.url),
    )
    page.on("requestfailed", lambda req: LOGGER.error("🚨 request failed › %s", req.url))


def _install_sniffer(page):
    """Capture GOOD/BAD network requests during initial bootstrap window."""
    global _BOOT_T0
    GOOD_REQS.clear()
    BAD_REQS.clear()
    _BOOT_T0 = time.time()

    dt = lambda: time.time() - _BOOT_T0  # noqa: E731

    page.on("request", lambda r: LOGGER.debug("▶ %.3fs %s", dt(), r.url))

    def on_resp(resp):
        if dt() > _BOOT_WIN:
            return
        url = resp.url
        if 200 <= resp.status < 400:
            GOOD_REQS.append(url)
        else:
            BAD_REQS.append(f"{resp.status} {url}")
        LOGGER.debug("✓ %.3fs %s %s", dt(), resp.status, url)

    page.on("response", on_resp)

    def on_fail(req):
        if dt() > _BOOT_WIN:
            return
        BAD_REQS.append(f"{req.failure} {req.url}")
        LOGGER.error("✘ %.3fs %s %s", dt(), req.failure, req.url)

    page.on("requestfailed", on_fail)


# --------------------------------------------------------------------------- #
# Async wrappers
# --------------------------------------------------------------------------- #
async def attach_debug_listeners_async(page: 'AsyncPage') -> None:
    """Attach console/JS/network listeners (async API)."""
    _attach_debug_listeners(page)


async def install_sniffer_async(context: 'AsyncBrowserContext', page: 'AsyncPage') -> None:
    """Attach bootstrap network sniffer (async API)."""
    _install_sniffer(page)


# --------------------------------------------------------------------------- #
# Sync wrappers
# --------------------------------------------------------------------------- #

def attach_debug_listeners_sync(page: 'SyncPage') -> None:
    """Attach console/JS/network listeners (sync API)."""
    _attach_debug_listeners(page)


def install_sniffer_sync(context: 'SyncBrowserContext', page: 'SyncPage') -> None:
    """Attach bootstrap network sniffer (sync API)."""
    _install_sniffer(page)


__all__ = [
    # Async
    "install_sniffer_async",
    "attach_debug_listeners_async",
    # Sync
    "install_sniffer_sync",
    "attach_debug_listeners_sync",
    # Globals
    "GOOD_REQS",
    "BAD_REQS",
    "LOGGER",
]
