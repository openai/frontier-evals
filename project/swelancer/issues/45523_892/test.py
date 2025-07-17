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
    int(time.time())
    return "turingtemp+994508@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
) -> tuple[Browser, Page, object, str]:
    """
    Logs-in/creates a user and returns (browser, page, context, user_email).
    If trace_enabled is True, tracing is started on the returned context.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(ignore_https_errors=True)

    # Start tracing immediately after context creation when requested
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install guard immediately after page creation

    user_email = generate_random_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)

    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(4000)
    except Exception:
        pass

    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(2000)
    except Exception:
        pass

    return browser, page, context, user_email


def _stop_trace_if_enabled(context, trace_enabled: bool):
    """
    Helper to stop tracing, save the file and clean it whenever trace is enabled.
    """
    if trace_enabled and context is not None:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_emojiart(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = page = context = None  # for safety in finally
        try:
            browser, page, context, user_email = login_user(p, trace_enabled=trace_enabled)

            # Test steps
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_text("Start chat").click()
            page.wait_for_timeout(3000)
            page.get_by_label("Milan T (you)").last.click()
            page.get_by_role("textbox").fill(
                "💙💗💗💗💗💗💗\n"
                "💙❤❤❤❤❤  💛💗\n"
                "💙❤HELLO ♪❤💛💗\n"
                "💙❤👍☀👎❤ 💛💗\n"
                "💙❤☮ O˥˥ƎH❤💛💗\n"
                "💙❤❤❤❤❤   💛💗\n"
                "💚💚💚💚💚💚💛\n"
            )
            page.keyboard.press("Enter")
            page.get_by_label("Emoji").nth(1).click()
            page.wait_for_timeout(2000)

            crash_message = "Uh-oh, something went wrong!"
            assert crash_message not in page.content(), (
                f"Crash message '{crash_message}' was found on the page"
            )

        finally:
            # Stop tracing (if enabled) and clean the trace
            _stop_trace_if_enabled(context, trace_enabled)

            # Close context safely
            if context is not None:
                context.close()

            # Close browser safely
            if browser is not None:
                browser.close()
