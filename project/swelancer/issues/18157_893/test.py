import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email() -> str:
    return "testerbro7077+8699649@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool = False,
    first_name: str = "Milan",
    last_name: str = "T",
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launches a Chromium browser, creates a new context & page,
    optionally starts tracing, and performs onboarding steps.

    Returns
    -------
    (browser, context, page, user_email)
    """
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
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation
    user_email = generate_random_email()

    # Begin sign-in / onboarding flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    time.sleep(1)

    # Sometimes the "Continue" button has to be clicked twice
    try:
        page.locator('button[tabindex="0"]').click()
        time.sleep(1)
    except Exception:
        pass

    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        time.sleep(1)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        time.sleep(1)
    except Exception:
        # If onboarding has already been completed, skip the flow.
        pass

    return browser, context, page, user_email


def test_character_limit_in_address_field(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = None
        context = None
        try:
            browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)

            setting_icon = page.get_by_label("My settings")
            expect(setting_icon).to_be_visible()
            setting_icon.click()
            time.sleep(1)

            page.locator('div:has-text("Address")').last.click()

            address_field = page.get_by_label("Address line 1").get_by_role("textbox")
            address_field.click()
            address_field.fill(
                "we can close this to focus on more urgent issue this is just a minor viewing discrepancy ~we can close this to focus on more urgent"
            )

            actual_value = address_field.input_value()
            expected_value = "we can close this to focus on more urgent issue th"
            assert actual_value == expected_value, (
                f"Expected '{expected_value}' but got '{actual_value}'"
            )

        finally:
            # Stop tracing & clean trace if enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the context first, then the browser
            if context is not None:
                context.close()

            if browser is not None:
                browser.close()
