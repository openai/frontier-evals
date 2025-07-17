# conftest.py
from playwright.async_api import Browser as AsyncBrowser
from playwright.async_api import expect as async_expect
from playwright.sync_api import Browser as SyncBrowser
from playwright.sync_api import expect as sync_expect

# ---------------------------------------------------------------------------
# Global test-wide timeout (milliseconds)
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 60_000  # 60 seconds


def _patch_browser_class(browser_cls, *, is_async: bool):
    """
    Replace .new_context() and .new_page() on the given Browser subclass
    so that every context / page they create inherits DEFAULT_TIMEOUT
    for actions & navigations.
    Works for both sync and async Browser classes.
    """
    orig_new_context = browser_cls.new_context
    orig_new_page = browser_cls.new_page

    if is_async:  # async_api.Browser

        async def new_context(self, *a, **kw):
            ctx = await orig_new_context(self, *a, **kw)
            ctx.set_default_timeout(DEFAULT_TIMEOUT)
            ctx.set_default_navigation_timeout(DEFAULT_TIMEOUT)
            return ctx

        async def new_page(self, *a, **kw):
            pg = await orig_new_page(self, *a, **kw)
            pg.set_default_timeout(DEFAULT_TIMEOUT)
            pg.set_default_navigation_timeout(DEFAULT_TIMEOUT)
            return pg
    else:  # sync_api.Browser

        def new_context(self, *a, **kw):
            ctx = orig_new_context(self, *a, **kw)
            ctx.set_default_timeout(DEFAULT_TIMEOUT)
            ctx.set_default_navigation_timeout(DEFAULT_TIMEOUT)
            return ctx

        def new_page(self, *a, **kw):
            pg = orig_new_page(self, *a, **kw)
            pg.set_default_timeout(DEFAULT_TIMEOUT)
            pg.set_default_navigation_timeout(DEFAULT_TIMEOUT)
            return pg

    browser_cls.new_context = new_context
    browser_cls.new_page = new_page


def pytest_configure(config):
    """
    Runs once, before tests are collected.
    * Sets the assertion timeout for BOTH expect objects.
    * Monkey-patches Browser classes so every context/page obeys DEFAULT_TIMEOUT.
    """
    sync_expect.set_options(timeout=DEFAULT_TIMEOUT)
    async_expect.set_options(timeout=DEFAULT_TIMEOUT)
    _patch_browser_class(SyncBrowser, is_async=False)
    _patch_browser_class(AsyncBrowser, is_async=True)


def pytest_addoption(parser):
    parser.addoption(
        "--user-tool-trace",
        action="store_true",
        default=False,
        help="Enable tracing of user-tool operations in Playwright tests.",
    )
