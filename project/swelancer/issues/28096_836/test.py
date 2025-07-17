import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright._impl._browser import Browser
from playwright._impl._browser_context import BrowserContext
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email():
    int(time.time())
    return "kenworktestemail+960@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launches a Chromium browser, creates a context & page, performs the sign-up flow,
    and (optionally) starts Playwright tracing.

    Returns
    -------
    (browser, context, page, user_email)
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins, site-per-process",
        ],
    )
    context = browser.new_context(ignore_https_errors=True)

    # Start tracing if requested
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Install the online guard immediately after page creation
    user_email = generate_random_email()

    # Sign-up / onboarding steps
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        # The button might not always be present; continue regardless.
        pass

    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)

    return browser, context, page, user_email


def test_choose_file_name(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)
        try:
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_text("Submit expense").click()
            page.wait_for_timeout(2000)
            expect(page.get_by_text("Choose file", exact=True)).to_be_visible()
        finally:
            # Stop tracing & clean traces if enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()  # Ensure context is closed before the browser
            browser.close()
