import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def login_user(
    p: PlaywrightContextManager, first_name: str = "Milan", last_name: str = "T"
) -> tuple[Browser, Page, str]:
    """
    Logs a user in and returns a tuple of (browser, page, user_email).
    NOTE: This helper spins up its own browser instance.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    # Create an explicit context and page, then install the online guard
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = "freelanceapptest+1733261135@gmail.com"
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        # Button might not be present for some flows; ignore if it isn't
        pass
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)
    return browser, page, user_email


def open_submit_expense_and_click_on_next_button(
    browser: Browser, page: Page, user_email: str
) -> tuple[Browser, Page, str]:
    plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
    expect(plus_icon).to_be_visible()
    plus_icon.click()
    page.wait_for_timeout(1000)

    new_workspace_button = page.locator('div[aria-label="Submit expense"]')
    expect(new_workspace_button).to_be_visible()
    new_workspace_button.click()
    page.wait_for_timeout(1000)

    manual_button = page.locator('button[aria-label="Manual"]')
    expect(manual_button).to_be_visible()
    manual_button.click()
    page.wait_for_timeout(1000)

    page.get_by_placeholder("0").fill("0")
    page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
    page.wait_for_timeout(2000)
    return browser, page, user_email


def test_error_message_top_margin(pytestconfig):
    """
    Ensures that the distance between the error message (‘Please enter a valid amount…’)
    and the ‘Next’ button is exactly 14 px.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None
    with sync_playwright() as p:
        try:
            # ------------------- Test flow begins ------------------- #
            browser, page, user_email = login_user(p)

            # Acquire the context that was implicitly created in login_user()
            context = page.context

            # Start tracing if requested
            if trace_enabled:
                context.tracing.start(
                    screenshots=True,
                    snapshots=True,
                    sources=True,
                )

            browser, page, user_email = open_submit_expense_and_click_on_next_button(
                browser, page, user_email
            )

            error_message = page.get_by_text(
                "Please enter a valid amount before continuing.", exact=True
            )
            next_button = page.locator("#numPadContainerView").get_by_role("button", name="Next")

            page.wait_for_timeout(1000)

            error_message_box = error_message.bounding_box()
            next_button_box = next_button.bounding_box()

            assert error_message_box is not None, "Error message bounding box not found."
            assert next_button_box is not None, "Next button bounding box not found."

            distance_between_elements = next_button_box["y"] - (
                error_message_box["y"] + error_message_box["height"]
            )
            print(f"Distance between elements: {distance_between_elements}px")
            assert distance_between_elements == 14, (
                f"Expected 14px distance, but found {distance_between_elements}px"
            )

            page.wait_for_timeout(1000)
            # ------------------- Test flow ends ------------------- #
        finally:
            # Handle tracing export & cleaning
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the context if it was created
            if context is not None:
                context.close()

            # Close the browser if it was created
            if browser is not None:
                browser.close()
