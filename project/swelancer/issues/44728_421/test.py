import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)
from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync

static_email = "testuserexpensify1+63@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "John",
    last_name: str = "Doe",
    trace_enabled: bool = False,
) -> tuple[Browser, Page, str]:
    """
    Logs a user into the Expensify NewDot site.

    Parameters
    ----------
    p : PlaywrightContextManager
        The Playwright context manager instance.
    first_name : str
        First name for the user.
    last_name : str
        Last name for the user.
    trace_enabled : bool
        Whether Playwright tracing should be started for the newly
        created browser context.

    Returns
    -------
    tuple[Browser, Page, str]
        The browser instance, newly opened page, and the user email.
    """
    browser = p.chromium.launch(
        headless=True,
        slow_mo=1000,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    # Create a dedicated context so we can manage tracing easily
    context = browser.new_context()

    # Start tracing immediately after context creation if required
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)
    user_email = static_email

    # Begin login / account–creation flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    try:
        page.locator('button[tabindex="0"]').click()
    except Exception:
        # The extra click is occasionally necessary during the auth flow;
        # ignore failures if the button isn't present.
        pass
    page.locator("text='Something else'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()

    return browser, page, user_email


def test_44728(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    trace_path = None

    with sync_playwright() as p:
        # ------------------------------------------------------------------ #
        # Test setup
        # ------------------------------------------------------------------ #
        browser, page, user_email = login_user(p, trace_enabled=trace_enabled)
        context = page.context  # Retrieve the context for later clean-up

        try:
            # ------------------------------------------------------------------ #
            # Test execution
            # ------------------------------------------------------------------ #
            personal_chat = page.get_by_text("(you)")
            personal_chat.click()

            concierge_text = page.locator("text=Concierge").nth(1)
            font_weight = concierge_text.evaluate("el => window.getComputedStyle(el).fontWeight")
            assert int(font_weight) >= 700, "The Concierge chat is not marked as unread."

        finally:
            # ------------------------------------------------------------------ #
            # Tear-down
            # ------------------------------------------------------------------ #
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                # Export and stop tracing
                context.tracing.stop(path=trace_path)
                # Clean the trace file to make it portable
                trace_cleaner(trace_path)

            context.close()
            browser.close()
