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
    task_number = 28265
    return f"checkissue+{task_number}@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
) -> tuple[Browser, Page, "BrowserContext", str]:
    """
    Creates the browser, context, and page, and performs sign-up/login.
    If trace_enabled is True, tracing is started immediately after context creation.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=500,
    )
    context = browser.new_context(ignore_https_errors=True)

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)
    user_email = generate_random_email()

    # Begin sign-up / login flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(2000)
    except Exception:
        pass
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(2000)
    # Return everything we might need
    return browser, page, context, user_email


def test_spanish_emoji_text(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        # Create browser / context / page (and possibly start tracing) via helper
        browser, page, context, user_email = login_user(p, trace_enabled=trace_enabled)

        try:
            # ==== Original test steps ====
            page.wait_for_timeout(1000)
            page.locator('span:text("Milan T (you)")').click()
            page.wait_for_timeout(1000)
            page.get_by_role("textbox").click()
            page.wait_for_timeout(1000)
            page.get_by_role("textbox").fill("Hi :rosa:")
            page.wait_for_timeout(1000)
            page.get_by_label("My settings").click()
            page.wait_for_timeout(1000)
            page.get_by_text("Preferences").click()
            page.wait_for_timeout(1000)
            page.get_by_text("Language").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Spanish").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Recibidos").click()
            page.wait_for_timeout(2000)
            expect(page.get_by_text("🌹", exact=True)).to_be_visible()
        finally:
            # ==== Tracing teardown (if enabled) ====
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the browser regardless of tracing
            browser.close()
