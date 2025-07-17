import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email():
    return "freelanceapptest+6787475@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    *,
    trace_enabled: bool = False,
    first_name: str = "A",
    last_name: str = "Khan",
) -> tuple[Browser, "BrowserContext", Page]:
    """
    Launches a Chromium browser, creates a context (starting tracing if requested),
    signs in a new user and returns the browser, context and page objects.
    """
    browser = p.chromium.launch(
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

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation

    user_email = generate_random_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(2000)

    return browser, context, page


def test_add_stop_button(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = context = page = None  # defined here for access inside finally

    with sync_playwright() as p:
        try:
            browser, context, page = login_user(p, trace_enabled=trace_enabled)

            # Test steps
            page.get_by_text("A Khan (you)").click()
            page.get_by_role("button", name="Create").click()
            page.get_by_label("Track expense").click()
            page.get_by_role("button", name="Distance").click()
            expect(page.get_by_role("button", name="Add stop")).not_to_be_visible()

            page.get_by_test_id("IOURequestStartPage").get_by_role("button", name="Start").click()
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "West Pine Street"
            )
            page.get_by_text("Louisville, CO, USA ").click()
            expect(page.get_by_role("button", name="Add stop")).not_to_be_visible()

            page.get_by_test_id("IOURequestStartPage").get_by_role("button", name="Stop").click()
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "East Pine Street"
            )
            page.get_by_text("Nederland, CO, USA").click()
            expect(page.get_by_role("button", name="Add stop")).to_be_visible()

        finally:
            # Handle trace export & cleaning
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Ensure context and browser are closed even if the test fails
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
