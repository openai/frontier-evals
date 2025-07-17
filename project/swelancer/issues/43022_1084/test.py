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


# Utility to generate a random email
def generate_random_email():
    timestamp = int(time.time())
    return f"milan.thummar-{timestamp}@gmail.com"


def login_user(
    p: PlaywrightContextManager, first_name="Milan", last_name="T"
) -> tuple[Browser, Page, str]:
    # Launch chromium and open new page
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    page = browser.new_page()
    install_online_guard_sync(
        page.context, page
    )  # Install online guard immediately after page creation
    user_email = generate_random_email()

    # Step 1: Open expensify url
    page.goto("https://dev.new.expensify.com:8082/")

    # Step 2: Enter email and click continue
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(5000)

    # Step 3: Click join button
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(10000)

    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
        # Step 4: Select 'Track and budget expenses' in onboarding page and click Continue
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(1000)

        # Step 5: Enter first name, last name and click continue
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(1000)
    except Exception:
        # If onboarding is skipped or already completed, silently continue
        pass

    return browser, page, user_email


def test_settings_header_style(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None
    with sync_playwright() as p:
        try:
            # Step 1: Login user
            browser, page, user_email = login_user(p)
            context = page.context  # Obtain the implicitly created context

            # Start tracing if the flag is enabled
            if trace_enabled:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # Step 2: Go to workspace settings
            my_settings_button = page.locator('button[aria-label="My settings"]')
            expect(my_settings_button).to_be_visible()
            my_settings_button.click()
            page.wait_for_timeout(1000)

            workspace_settings_button = page.locator(
                'div[aria-label="Workspaces"][role="menuitem"]'
            )
            expect(workspace_settings_button).to_be_visible()
            workspace_settings_button.click()
            page.wait_for_timeout(1000)

            # Step 3: Rocket icon should be visible in header
            rocket_icon = page.locator(
                "div:nth-child(4) > div:nth-child(2) > div:nth-child(2) > div > div > div > div > div > div > div > div > div:nth-child(2) > div > div > svg"
            )
            expect(rocket_icon).to_be_visible()

            # Step 4: "Create new workspace" should be visible in h1 (font-size=22px)
            h1_styled_title = page.locator(
                'div[style*="font-size: 22px;"]', has_text="Create a workspace"
            )
            expect(h1_styled_title).to_be_visible()

            page.wait_for_timeout(2000)
        finally:
            # Stop tracing and clean up if enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the browser if it was created
            if browser is not None:
                browser.close()
