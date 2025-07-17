import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync


def generate_random_email():
    return "freelanceapptest+643895@gmail.com"


def login_user(
    p: PlaywrightContextManager, first_name: str = "A", last_name: str = "Khan"
) -> tuple[Browser, Page]:
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)

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
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)
    return browser, page


def set_default_currency(
    browser: Browser, page: Page, search_text: str, currency: str
) -> tuple[Browser, Page]:
    default_currency_button = page.locator('div[role="menuitem"]').get_by_text("Default currency")
    expect(default_currency_button).to_be_visible()
    default_currency_button.click()
    input_currency = page.locator('input[aria-label="Default currency"]')
    input_currency.fill(search_text)
    expect(page.get_by_test_id("WorkspaceProfileCurrencyPage")).to_contain_text(currency)
    return browser, page


def check_currency_in_workspace_chat(browser: Browser, page: Page) -> tuple[Browser, Page]:
    plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
    expect(plus_icon).to_be_visible()
    plus_icon.click()
    new_workspace_button = page.locator('div[aria-label="New workspace"]')
    expect(new_workspace_button).to_be_visible()
    new_workspace_button.click()
    browser, page = set_default_currency(browser, page, "rupee", "INR - ₹")
    return browser, page


def test_currency_search(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    # Keep a reference to created contexts for trace handling
    contexts = []

    with sync_playwright() as p:
        try:
            # Login user and optionally start tracing
            browser, page = login_user(p)
            context = page.context
            contexts.append(context)

            if trace_enabled:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # Main test logic
            browser, page = check_currency_in_workspace_chat(browser, page)

        finally:
            # Stop tracing and clean traces, independent of test outcome
            if trace_enabled:
                for ctx in contexts:
                    try:
                        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                        ctx.tracing.stop(path=trace_path)
                        trace_cleaner(trace_path)
                    except Exception:
                        # Ensure we don't mask original exceptions
                        pass

            # Close contexts
            try:
                for ctx in contexts:
                    ctx.close()
            except Exception:
                pass

            # Ensure browser is closed
            try:
                browser.close()
            except Exception:
                pass
