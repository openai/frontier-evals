import os
import sys
import time
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import Page, expect, sync_playwright
from utils.online_guard import install_online_guard_sync


def generate_random_email(timestamp: int = int(time.time())):
    return "testerbro7077+989357@gmail.com"


def create_user(
    page: Page, firstname: str, lastname: str | None = None, timestamp: int | None = None
):
    timestamp = timestamp or int(time.time())
    page.get_by_test_id("username").fill(generate_random_email(timestamp))
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Join").click()
    page.get_by_text("Track and budget expenses").click()
    page.get_by_role("textbox", name="First name").fill(firstname)
    page.get_by_role("textbox", name="Last name").fill(lastname or f"{timestamp}")
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Continue").click()


def close_button_if_present(page: Page):
    """
    Occasionally, there is a close button that prevents any clicks on the page as
    it covers most of the screen. This button cannot be seen visually.
    """
    close_button = page.locator('button[aria-label="Close"]')
    if close_button.is_visible():
        close_button.click()


def test_horizontal_avatar_stack(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        # Launch browser and create context
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            slow_mo=500,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()

        # Start tracing if enabled
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            # Begin test workflow
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)

            int(time.time())
            create_user(page, "TestUser", "me")
            page.wait_for_timeout(1000)

            close_button_if_present(page)
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_label("Start chat", exact=True).click()

            email_suffixes = [1, 2, 3]
            for suffix in email_suffixes:
                page.get_by_test_id("selection-list-text-input").fill(
                    f"testuser7077+989357{suffix}@gmail.com"
                )
                page.get_by_role("button", name="Add to group").click()

            page.get_by_role("button", name="Next").nth(0).click()
            page.get_by_role("button", name="Start group").click()
            page.get_by_label("Create").last.click()
            page.get_by_label("Split expense").click()
            page.get_by_label("Manual").click()
            page.get_by_placeholder("0").fill("125")
            page.get_by_role("button", name="Next").nth(0).click()
            page.get_by_role("button", name="Split ").click()

            expect(
                page.locator('div[style="align-items: center; flex-direction: row; height: 32px;"]')
            ).to_be_visible()

            page.wait_for_timeout(1000)

        finally:
            # Stop tracing and clean traces if enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close context and browser
            context.close()
            browser.close()
