import os
import sys
import time
import uuid

import pytest
from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import BrowserContext, expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)


def generate_random_email() -> str:
    timestamp = int(time.time())
    return f"t56826807+{timestamp}@gmail.com"


def login_user(
    playwright: PlaywrightContextManager,
    first_name: str,
    last_name: str,
    trace_enabled: bool = False,
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launches a browser, creates a context (with optional tracing),
    logs the user in and returns browser, context, page and the generated email.
    """
    browser = playwright.chromium.launch(
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
    context = browser.new_context(
        viewport={"width": 360, "height": 640},
        user_agent="Mozilla/5.0 (Linux; Android 10; Mobile; rv:68.0) Gecko/68.0 Firefox/68.0",
    )

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Installed Online Guard immediately after page creation
    user_email = generate_random_email()

    # Begin sign-up / login flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        # In case the extra click is not required
        pass

    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.wait_for_timeout(1000)
    page.locator("text='Manage my team's expenses'").click()
    page.get_by_role("button", name="Continue").click()
    time.sleep(1)

    return browser, context, page, user_email


def create_profile(page: Page, first_name: str, last_name: str) -> None:
    page.locator('input[name="fwork"]').fill("Turingg")
    page.get_by_role("button", name="Continue").last.click()
    time.sleep(1)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)


def check_you(page: Page) -> None:
    page.locator('button[aria-label="Start chat (Floating action)"][tabindex="0"]').click()
    time.sleep(1)
    page.locator('div[aria-label="Start chat"][role="menuitem"]').click()
    time.sleep(1)
    you = page.locator('button[aria-label="Yagna Sai (you)"]')
    assert you.is_visible()


@pytest.mark.parametrize("first_name, last_name", [("Yagna", "Sai")])
def test_run(first_name: str, last_name: str, pytestconfig) -> None:
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as playwright:
        browser, context, page, user_email = login_user(
            playwright, first_name, last_name, trace_enabled=trace_enabled
        )

        try:
            create_profile(page, first_name, last_name)
            check_you(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()  # Ensure context is closed before closing browser
            browser.close()


if __name__ == "__main__":
    pytest.main([__file__])
