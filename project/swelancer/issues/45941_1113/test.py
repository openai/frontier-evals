import os
import sys
import time
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email():
    int(time.time())
    return "t56826807+1733396881@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "A",
    last_name: str = "Khan",
    trace_enabled: bool = False,
):
    """
    Launches the browser, creates a context (optionally starts tracing),
    returns (browser, page, context, trace_started_bool).
    """
    browser: Browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=500,
    )
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)  # Installed guard immediately after page creation

    user_email = generate_random_email()

    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    time.sleep(1)

    try:
        page.locator('button[tabindex="0"]').click()
        time.sleep(1)
    except Exception:
        pass

    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    time.sleep(1)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    time.sleep(1)

    return browser, page, context


def test_old_dot_page_redirection(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    trace_path = None

    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Login user (tracing may start inside login_user)
            browser, page, context = login_user(p, trace_enabled=trace_enabled)

            # Begin test steps
            page.goto(
                "https://dev.new.expensify.com:8082/settings/exit-survey/reason",
                timeout=60000,
            )
            time.sleep(1)
            expect(page).to_have_url(
                "https://dev.new.expensify.com:8082/settings/exit-survey/reason"
            )

            page.get_by_text("I don't understand how to use New Expensify.").click()
            expect(page.get_by_text("I don't understand how to use New Expensify.")).to_be_visible()

            page.get_by_text("Next").click()
            page.get_by_role("textbox", name="Your response").fill("N/A")
            expect(page.get_by_role("textbox", name="Your response")).to_have_value("N/A")
            page.get_by_role("button", name="Next").click()
            page.get_by_role("button", name="Switch to Expensify Classic").click()

            try:
                with context.expect_page() as new_page_info:
                    new_page = new_page_info.value
                    assert new_page is not None, "A new tab should have opened but it didn't."
            except PlaywrightTimeoutError:
                assert False, (
                    "Failed to open new tab after clicking the "
                    "'Switch to Expensify Classic' button."
                )

            time.sleep(1)

        finally:
            # Stop tracing (if enabled) and clean trace
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Ensure context and browser are closed
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
