import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright


def launch_browser(platform: str = "desktop", browser_type: str = "chromium"):
    """
    Launch a persistent browser with the specified platform
    """
    playwright = sync_playwright().start()
    data_dir = f"{browser_type}_{platform}_context"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    mobile_context = {}
    if platform.lower() == "mobile":
        mobile_context = {"viewport": {"width": 540, "height": 864}, "is_mobile": True}

    context = playwright[browser_type].launch_persistent_context(
        data_dir,
        headless=True,
        **mobile_context,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=500,
    )
    page = context.pages[0]
    return playwright, context, page


def login_user_if_not_logged_in(
    page: Page,
    first_name: str = "Asaad",
    last_name: str = "I",
    email: str = "testuser+29001@gmail.com",
    platform: str = "desktop",
):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.wait_for_load_state("load")
    try:
        expect(page.locator('input[type="email"]')).to_be_visible(timeout=3000)
    except Exception:
        # Already logged-in
        return

    page.locator('input[type="email"]').fill(email)
    page.locator("button", has_text="Continue").click()
    page.locator("button", has_text="Join").click()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("form").get_by_role("button", name="Continue").click()

    if platform.lower() == "mobile":
        page.get_by_label("Last chat message preview").filter(
            has_text="task for Track an expense"
        ).click()
        page.get_by_label("Back").click()


def verify_start_chat_focus(page: Page):
    """
    Verify that the start chat page is focused when navigating back and forth between start chat and start room pages
    """
    page.get_by_label("My settings").click()
    page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").click()

    if page.get_by_label("row").count() == 0:
        page.get_by_label("New workspace").first.click()
        page.get_by_label("Back").click()

    page.get_by_label("Inbox").click()
    page.get_by_label("Start chat (Floating action)").click()
    page.get_by_label("Start chat", exact=True).click()

    for _ in range(3):
        page.get_by_label("Room", exact=True).click()
        page.get_by_test_id("NewChatSelectorPage").get_by_label("Chat").click()

    expect(page.get_by_test_id("selection-list-text-input")).to_be_focused()


def test_start_chat_focus(pytestconfig):
    """
    Test that the start chat page is focused
    When navigating back and forth between start chat and start room pages
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    playwright = None
    context = None

    try:
        platform = "desktop"
        browser_type = "chromium"
        playwright, context, page = launch_browser(platform, browser_type)
        install_online_guard_sync(context, page)

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        email = "expensify66+29001@gmail.com"
        login_user_if_not_logged_in(page, "Asaad", "I", email, platform)
        verify_start_chat_focus(page)

    finally:
        if context:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                # Stop tracing and save the trace
                try:
                    context.tracing.stop(path=trace_path)
                    trace_cleaner(trace_path)
                except Exception:
                    # Ensure that even if tracing stop fails, we still close context
                    pass
            # Close browser context
            try:
                context.close()
            except Exception:
                pass

        if playwright:
            try:
                playwright.stop()
            except Exception:
                pass
