import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email():
    int(time.time())
    return "milan.thummar7476848@gmail.com"


def login_user(
    p: PlaywrightContextManager, first_name="Milan", last_name="T"
) -> tuple[Browser, Page, str]:
    """
    Starts a headless Chromium browser, signs the user in (or completes any
    onboarding flows that appear) and returns the browser, page and email used.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)  # Install the online guard immediately after creation
    user_email = generate_random_email()

    # Begin login / onboarding flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)

    # Handle “Let’s go” button if it shows up twice
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    # Handle optional onboarding screens
    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(1000)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
    except Exception:
        pass

    page.wait_for_timeout(1000)
    return (browser, page, user_email)


def enable_report_fields(browser: Browser, page: Page, user_email: str):
    more_features_button = page.locator('div[aria-label="More features"]')
    expect(more_features_button).to_be_visible()
    page.wait_for_timeout(1000)
    more_features_button.click()
    page.wait_for_timeout(1000)

    report_fields_button = page.locator('div[aria-label="Report fields"][role="menuitem"]')
    if not report_fields_button.is_visible():
        report_fields_switch = page.locator('button[aria-label="Set up custom fields for spend."]')
        expect(report_fields_switch).to_be_visible()
        page.wait_for_timeout(1000)
        report_fields_switch.click()
        page.wait_for_timeout(1000)

    try:
        page.locator('button[data-tag="pressable"]', has_text="Upgrade").last.click()
        page.wait_for_timeout(1000)
        page.locator('button[data-tag="pressable"]', has_text="Got it, thanks").last.click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    page.wait_for_timeout(1000)
    expect(report_fields_button).to_be_visible()
    page.wait_for_timeout(1000)
    report_fields_button.click()
    page.wait_for_timeout(1000)
    return (browser, page, user_email)


def create_new_workspace(
    browser: Browser, page: Page, user_email: str
) -> tuple[Browser, Page, str]:
    plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
    expect(plus_icon).to_be_visible()
    plus_icon.click()
    page.wait_for_timeout(1000)

    new_workspace_button = page.locator('div[aria-label="New workspace"]')
    expect(new_workspace_button).to_be_visible()
    new_workspace_button.click()
    page.wait_for_timeout(1000)
    return (browser, page, user_email)


def add_report_field_list_value(page: Page, value: str) -> Page:
    page.locator("button", has_text="Add value").last.click()
    page.wait_for_timeout(1000)
    page.locator('input[aria-label="Value"]').last.fill(value)
    page.wait_for_timeout(1000)
    page.locator('button[data-listener="Enter"]', has_text="Save").last.click()
    page.wait_for_timeout(1000)
    return page


def test_report_fields_not_visible_after_desabling_feature(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    context = None
    with sync_playwright() as p:
        try:
            # Login / setup
            browser, page, user_email = login_user(p)
            context = page.context  # Obtain reference to the context

            # Start tracing if requested
            if trace_enabled and context is not None:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # Main test steps
            browser, page, user_email = create_new_workspace(browser, page, user_email)
            browser, page, user_email = enable_report_fields(browser, page, user_email)

            # Create a new report field
            page.locator("button", has_text="Add field").last.click()
            page.wait_for_timeout(1000)
            page.locator('div[role="menuitem"]', has_text="Name").last.click()
            page.wait_for_timeout(1000)
            page.locator('input[aria-label="Name"]').last.fill("MyListReportField")
            page.wait_for_timeout(1000)
            page.locator('button[data-listener="Enter"]', has_text="Save").last.click()
            page.wait_for_timeout(1000)
            page.locator('div[role="menuitem"]', has_text="Type").last.click()
            page.wait_for_timeout(1000)
            page.locator('button[aria-label="List"]', has_text="List").last.click()
            page.wait_for_timeout(1000)
            page.locator('div[role="menuitem"][tabindex="0"]', has_text="List values").last.click()
            page.wait_for_timeout(1000)

            # Add list values
            page = add_report_field_list_value(page, "TestValueA")
            page = add_report_field_list_value(page, "TestValueB")
            page = add_report_field_list_value(page, "TestValueC")

            # Navigate back and make assertions
            page.locator('button[aria-label="Back"]').last.click()
            page.wait_for_timeout(1000)
            list_values_menuitem = page.locator(
                'div[role="menuitem"][tabindex="0"]', has_text="List values"
            ).last
            expect(list_values_menuitem).to_be_visible()
            list_values_bounding_box = list_values_menuitem.bounding_box()

            initial_values_menuitem = page.locator(
                'div[role="menuitem"][tabindex="0"]', has_text="Initial value"
            ).last
            expect(initial_values_menuitem).to_be_visible()
            initial_values_bounding_box = initial_values_menuitem.bounding_box()

            assert list_values_bounding_box["y"] < initial_values_bounding_box["y"]

            list_values = page.locator("div", has_text="TestValueA, TestValueB, TestValueC").last
            expect(list_values).to_be_visible()
            page.wait_for_timeout(2000)

        finally:
            # Stop tracing and clean up
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close context and browser if they were created
            try:
                if context is not None:
                    context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass
