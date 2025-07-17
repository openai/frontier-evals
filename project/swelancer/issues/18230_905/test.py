import os
import sys
import time
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import Page, expect, sync_playwright


def generate_random_email(timestamp: int = int(time.time())):
    return "testerbro7077+86675667@gmail.com"


def create_user(
    page: Page, firstname: str, lastname: str | None = None, timestamp: int | None = None
):
    page.evaluate(
        "\n        Onyx.merge('nvp_onboarding', { hasCompletedGuidedSetupFlow: false });\n            "
    )
    page.reload()
    page.wait_for_timeout(1000)
    timestamp = timestamp or int(time.time())
    page.get_by_test_id("username").fill(generate_random_email())
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Join").click()
    page.get_by_text("Track and budget expenses").click()
    page.get_by_role("textbox", name="First name").fill(firstname)
    page.get_by_role("textbox", name="Last name").fill(lastname)
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


def test_emoji_picker(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(
                context, page
            )  # Install online guard immediately after page creation
            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
            create_user(page, "TestUser", "me")
            page.wait_for_timeout(1000)
            close_button_if_present(page)
            page.locator('button[aria-label="Navigates to a chat"]').nth(1).click()
            page.get_by_role("textbox").fill(":pizza")
            expect(page.get_by_label("pizza+0}")).to_be_visible()
            close_button_if_present(page)
            page.keyboard.press("Enter")
            pizza_text = page.locator('span:has-text(":pizza")').inner_text()
            page.get_by_role("textbox").fill(pizza_text)
            page.wait_for_timeout(1000)
            expect(page.get_by_label("pizza+0}")).to_be_visible()
            assert "\u2066" not in pizza_text, "Text should not contain '\u2066'"
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()
            browser.close()
