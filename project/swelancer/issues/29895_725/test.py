import os
import sys
import uuid

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import BrowserContext, expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)


def login_user(
    p: PlaywrightContextManager, first_name: str = "A", last_name: str = "Khan"
) -> tuple[Browser, BrowserContext, Page]:
    """
    Logs a user into NewDot and returns the browser, context & page objects.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = "checkissue+29895@gmail.com"
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass
    page.wait_for_timeout(5000)
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)
    return browser, context, page


def test_duplicate_waypoints(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None
    page = None

    with sync_playwright() as p:
        try:
            # Create browser & perform login
            browser, context, page = login_user(p)

            # If tracing is enabled, start tracing immediately after context creation
            if trace_enabled:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # Begin actual test steps
            page.get_by_text("A Khan (you)").click()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Create").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Track expense").click()
            page.get_by_role("button", name="Distance").click()
            page.wait_for_timeout(1000)
            page.get_by_test_id("IOURequestStartPage").get_by_role("button", name="Start").click()
            page.wait_for_timeout(1000)
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "India Gate"
            )
            page.wait_for_timeout(1000)
            page.get_by_text("New Delhi, Delhi, India", exact=True).click()
            page.wait_for_timeout(1000)
            page.get_by_test_id("IOURequestStartPage").get_by_role("button", name="Stop").click()
            page.wait_for_timeout(1000)
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "India Gate"
            )
            page.wait_for_timeout(1000)
            page.get_by_text("New Delhi, Delhi, India", exact=True).click()
            page.wait_for_timeout(1000)
            expect(page.get_by_role("button", name="Add stop")).to_be_visible()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Add stop").click()
            page.wait_for_timeout(1000)
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "India Gate"
            )
            page.wait_for_timeout(1000)
            page.get_by_text("New Delhi, Delhi, India", exact=True).click()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Next").nth(1).click()
            page.wait_for_timeout(1000)
            expect(page.get_by_role("button", name="Track expense")).not_to_be_visible()
            page.wait_for_timeout(1000)
        finally:
            # Stop tracing & clean trace files, if enabled
            if trace_enabled and context is not None:
                try:
                    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                    context.tracing.stop(path=trace_path)
                    trace_cleaner(trace_path)
                except Exception:
                    # Ensure browser closes even if trace stopping encounters an error
                    pass

            # Close the browser and context if they were created
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
