import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email():
    timestamp = int(time.time())
    # Keeping the original behaviour — timestamp isn't used anywhere but
    # we preserve it for future-proofing
    _ = timestamp
    return "kenworktestemail+828@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Ayush",
    last_name: str = "G",
    trace_enabled: bool = False,
) -> tuple[Browser, Page, str]:
    """
    Launches a browser, signs-up/logs-in a user and returns the browser, page and
    generated e-mail address.

    If trace_enabled is True, tracing starts immediately after the browser
    context is created.
    """
    browser = p.chromium.launch(
        headless=True,
        devtools=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins, site-per-process",
        ],
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Install the online guard immediately after page creation

    # Start tracing right away if requested
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    user_email = generate_random_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(3000)
    except Exception:
        pass
    try:
        page.locator('button[tabindex="0"]').click()
    except Exception:
        pass
    if page.locator("text='Track and budget expenses'"):
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        try:
            get_started_button = page.locator('text="Get started"')
            expect(get_started_button).to_be_visible()
            get_started_button.click()
        except Exception:
            pass
    return browser, page, user_email


def _stop_and_clean_trace(context, trace_enabled: bool):
    """
    Helper to stop tracing (if enabled) and post-process the trace file.
    """
    if not trace_enabled:
        return
    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_console_error(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        # Initialise variables for scope visibility
        browser = None
        page = None
        try:
            browser, page, user_email = login_user(
                p, first_name="Ayush", last_name="G", trace_enabled=trace_enabled
            )
            page_errors = []
            page.on("pageerror", lambda message: page_errors.append(message))

            page.locator('span:text("Ayush G (you)")').click()
            error_count_before_click = len(page_errors)

            current_url = page.url
            page.get_by_role("textbox").fill(f"{current_url}/duplicates/review")
            page.get_by_role("button", name="Send").click()
            page.wait_for_timeout(3000)

            page.get_by_test_id("a").click()
            try:
                not_here_page_div = page.locator('div:text("Hmm... it\'s not here")')
                expect(not_here_page_div).to_be_visible()
            except Exception:
                page.get_by_role("button", name="Keep all").click()

            page.wait_for_timeout(3000)
            error_count_after_click = len(page_errors)
            assert error_count_before_click == error_count_after_click
        finally:
            # Stop tracing (if we actually started it)
            if page is not None:
                _stop_and_clean_trace(page.context, trace_enabled)
                # Close the context before closing the browser
                page.context.close()
            # Close browser if it was opened
            if browser is not None:
                browser.close()
