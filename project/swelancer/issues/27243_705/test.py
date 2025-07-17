import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright.sync_api import (
    Browser,
    Page,
    expect,
    sync_playwright,
)


def generate_random_email():
    return "expensifyapp97+99999988@gmail.com"


def login_user(
    p, trace_enabled: bool = False, first_name: str = "Milan", last_name: str = "T"
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launches a browser, creates a context & page, optionally starts tracing,
    signs the user in, and returns the browser, context, page, and email used.
    """
    proxy = {"server": "http://127.0.0.1:8080/"}
    args = [
        "--disable-web-security",
        "--disable-features=IsolateOrigins,site-per-process",
    ]

    browser: Browser = p.chromium.launch(headless=True, args=args, proxy=proxy)
    context: BrowserContext = browser.new_context()

    # Start tracing immediately after context creation if enabled
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)  # Installed online guard

    user_email = generate_random_email()

    # ----------- Sign-up / onboarding flow -----------
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    time.sleep(1)
    try:
        page.locator('button[tabindex="0"]').click()
        time.sleep(1)
    except Exception:
        pass

    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    time.sleep(1)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    time.sleep(1)

    return browser, context, page, user_email


def test_cursor_disabled(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)

        try:
            # Navigate to personal chat
            personal_chat = page.get_by_text("Milan T (you)")
            expect(personal_chat).to_be_visible()
            personal_chat.click()
            time.sleep(1)

            # Create a task
            plus_create_icon = page.locator('button[aria-label="Create"]').last
            expect(plus_create_icon).to_be_visible()
            plus_create_icon.click()
            time.sleep(1)

            submit_expense_button = page.locator('div[aria-label="Assign task"]')
            expect(submit_expense_button).to_be_visible()
            submit_expense_button.click()
            time.sleep(1)

            page.get_by_role("textbox", name="Title").fill("Samuel")
            page.get_by_role("button", name="Next").click()
            page.get_by_role("button", name="Confirm task").click()
            page.get_by_role("checkbox", name="Task").click()

            # Validate cursor style on description element
            page.locator('div[aria-label="Chat message"]').last.click()
            element = page.locator('text="Description"')
            element.click()
            cursor_style = element.evaluate("element => window.getComputedStyle(element).cursor")
            assert cursor_style == "default", (
                f"Expected cursor to be disabled, but it was '{cursor_style}'"
            )
            print(f"Cursor style: {cursor_style}")

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()  # Ensure context is closed before browser
            browser.close()
