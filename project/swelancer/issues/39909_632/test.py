import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from time import time

from playwright._impl._browser_context import BrowserContext
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright


def generate_random_email():
    timestamp = int(time())
    return f"bobexpensifytest+{timestamp}@gmail.com"


def login_user(page: Page, user_email: str, first_name="Milan", last_name="T") -> Page:
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
    except Exception:
        pass
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    return page


def create_new_workspace(page: Page) -> Page:
    plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
    plus_icon.click()
    new_workspace_button = page.locator('div[aria-label="New workspace"]')
    new_workspace_button.click()
    back_arrow_button = page.locator('button[aria-label="Back"]')
    back_arrow_button.click()
    return page


def test_fallback_avatar_background(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context: BrowserContext = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page: Page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard

            bob_email = "blablabla+39909@example.com"
            user_email = "blablabla@example.com"

            page = login_user(page, bob_email, first_name="Bob")
            page = create_new_workspace(page)

            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            plus_icon.click()

            submit_expense_button = page.locator('div[aria-label="Start chat"]')
            submit_expense_button.click()

            page.locator('input[aria-label="Name, email, or phone number"]').last.click()
            page.keyboard.type(user_email)

            user_button = page.locator(f'button[aria-label="{user_email}"] div').first
            style_attribute = user_button.get_attribute("style")
            assert "background-color: rgb(230, 225, 218)" in style_attribute

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()  # Ensure context is closed
            browser.close()
