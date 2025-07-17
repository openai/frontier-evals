import os
import re
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import (
    expect,
    sync_playwright,
)
from playwright.sync_api._context_manager import PlaywrightContextManager


def get_user_email() -> str:
    return f"nitish.expensify+{64}@gmail.com"


def get_magic_code() -> str:
    return "123456"


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool = False,
    first_name: str = "Milan",
    last_name: str = "T",
) -> tuple[Browser, BrowserContext, Page]:
    """
    Launches a browser + context, optionally starts tracing,
    and performs the Expensify login flow.

    Returns
    -------
    tuple
        (browser, context, page)
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=2000,
    )

    # Create an explicit context so we can control tracing on it
    context = browser.new_context()

    # Start tracing immediately after context creation if requested
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = get_user_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()

    try:
        expect(page.get_by_role("button", name="Sign in")).to_be_visible()
        page.get_by_test_id("validateCode").fill(get_magic_code())
        try:
            page.get_by_role(
                "button",
                name=re.compile(r"^(continue|sign.?in)$", re.I),
            ).click(timeout=20000)
        except Exception:
            pass
    except Exception:
        # New user flow
        page.get_by_role("button", name="Join").click()

    # If this is a *brand new* account, complete display-name setup
    try:
        expect(page.locator('div[aria-label="Track and budget expenses"]').nth(0)).to_be_visible()
        page.locator('div[aria-label="Track and budget expenses"]').nth(0).click()
        page.locator('input[name="fname"]').nth(0).fill(first_name)
        page.locator('input[name="lname"]').nth(0).fill(last_name)
        page.get_by_role("button", name="Continue").nth(0).click()
    except Exception:
        # Account already set up – nothing to do
        pass

    # Dismiss potential welcome modal
    try:
        get_started_here = page.locator("div", has_text="Get started here!").nth(-1)
        expect(get_started_here).to_be_visible()
        page.get_by_label("Close").nth(-1).click()
    except Exception:
        pass

    return browser, context, page


def test_check_background(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    trace_path: str | None = None

    with sync_playwright() as p:
        browser: Browser | None = None
        context: BrowserContext | None = None
        page: Page | None = None

        try:
            # Log the user in (this also starts tracing if enabled)
            browser, context, page = login_user(p, trace_enabled=trace_enabled)

            # Navigate to the page under test
            page.goto(
                "https://dev.new.expensify.com:8082/settings/profile/status",
                timeout=60000,
            )

            # Validate background colour
            div_element = page.locator("#BaseOverlay")
            tag_style = div_element.get_attribute("style")
            expected_tag_style = "rgb(230, 225, 218)"
            assert expected_tag_style in tag_style, (
                f"Expected overlay to have '{expected_tag_style}' background-color, "
                f"but got '{tag_style}'"
            )

        finally:
            # Handle trace export/clean-up if tracing was enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Always close the context and browser if they were created
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
