import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync

original_email = "nitish.expensify@gmail.com"
APP_PASSWORD = "ojhi eovp lbkz ktfb"


def get_user_email():
    return f"nitish.expensify+{72}@gmail.com"


def get_magic_code():
    return "123456"


def login_user(p: PlaywrightContextManager, trace_enabled: bool = False):
    """
    Launches a browser, creates a context (optionally starting playwright tracing)
    and logs‐in / signs‐up the user, returning the trio (browser, context, page)
    so that the caller can properly close / stop tracing later on.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=1000,
    )
    context = browser.new_context(ignore_https_errors=True)

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
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
    except Exception:
        page.get_by_role("button", name="Join").click()

    return browser, context, page


def test_hide_thread(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = context = page = None  # Initialise for use in finally
        try:
            # Log in / sign up user (the context may start tracing within)
            browser, context, page = login_user(p, trace_enabled)

            # Test steps
            page.get_by_label("Workspaces").click()
            page.get_by_label("Back").click()
            page.locator('div[data-testid="DisplayNamesWithTooltip"]', has_text="#room").nth(
                0
            ).click()
            page.get_by_role("textbox").fill("hey")
            page.locator('button[aria-label="Send"]').nth(-1).click()
            page.locator("div").filter(has_text=re.compile("^hey$")).nth(-1).click(button="right")
            page.get_by_text("Reply in thread").click()
            page.locator('div[data-testid="DisplayNamesWithTooltip"]', has_text="#room").nth(
                0
            ).click()
            expect(
                page.locator(
                    'button[aria-label="Navigates to a chat"][role="button"][tabindex="0"]',
                    has_text="No activity yet",
                )
            ).not_to_be_visible()

        finally:
            # Handle tracing export / cleaning if enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Always close the browser and context if they were created
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
