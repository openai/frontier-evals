import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright.sync_api import Locator, Page, sync_playwright
from utils.online_guard import install_online_guard_sync


def generate_random_email(timestamp: int = int(time.time())):
    return "testerbro7077+7832526@gmail.com"


def create_user(
    page: Page, firstname: str, lastname: str | None = None, timestamp: int | None = None
):
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


def is_element_visible_in_viewport(page: Page, elem: Locator):
    bounding_box = elem.bounding_box()
    if bounding_box:
        x, y, width, height = (
            bounding_box["x"],
            bounding_box["y"],
            bounding_box["width"],
            bounding_box["height"],
        )
        viewport_size = page.viewport_size
        viewport_width = viewport_size["width"]
        viewport_height = viewport_size["height"]
        if x + width > 0 and x < viewport_width and (y + height > 0) and (y < viewport_height):
            return True
        else:
            return False
    else:
        return False


def test_menu_item(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None
    with sync_playwright() as p:
        try:
            int(time.time())
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

            page = context.new_page()
            install_online_guard_sync(context, page)
            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
            create_user(page, "TestUser", "me")
            page.wait_for_timeout(1000)
            long_email = "testuser8PvDmrZ4gE2zDyRNFUYJf1u5i3N3tscMk3Jv4lhsk03vYEYQsGIw@gmail.com"
            close_button_if_present(page)
            page.get_by_text("TestUser me (you)").click()
            page.get_by_label("Create").last.click()
            page.get_by_label("Assign task").click()
            page.get_by_role("textbox", name="Title").fill("Title")
            page.get_by_role("textbox", name="Description (optional)").fill("Desc")
            page.keyboard.press("Enter")
            page.get_by_role("menuitem", name="Assignee").click()
            page.get_by_test_id("selection-list-text-input").fill(long_email)
            page.get_by_label(long_email).click()
            elem = page.get_by_label(long_email).get_by_role("img").nth(1)
            assert is_element_visible_in_viewport(page, elem) is True, (
                "Element is not visible in viewport"
            )
        finally:
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                # Stop tracing and export the trace
                context.tracing.stop(path=trace_path)
                # Clean the trace file for external sharing
                trace_cleaner(trace_path)

            # Close context and browser if they were created
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
