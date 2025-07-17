import os
import re
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

TEST_URL = "https://dev.new.expensify.com:8082/"
USER_EMAIL = "xecuycismfsga+k1nrfwcn@gmail.com"


def wait(page: Page, for_seconds: int = 1):
    page.wait_for_timeout(for_seconds * 1000)


def login(
    p: PlaywrightContextManager,
    trace_enabled: bool = False,
    if_phone: bool = False,
) -> tuple[Browser, BrowserContext, Page]:
    """
    Logs the user into the Expensify app.

    Returns
    -------
    Tuple of (browser, context, page)
    """
    # Step 1: Launch browser
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    if if_phone:
        phone = p.devices["iPhone 12 Pro"]
        context = browser.new_context(
            **phone,
            permissions=["clipboard-read", "clipboard-write"],
            reduced_motion="no-preference",
        )
    else:
        context = browser.new_context()

    # Start tracing immediately after context creation if enabled
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation

    # Navigate to login page
    page.goto(TEST_URL, timeout=120000)

    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()
    phone_or_email_input.fill(USER_EMAIL)

    continue_button = page.locator('button[tabindex="0"]')
    expect(continue_button).to_be_visible()
    continue_button.click()

    # Step 2: Handle possible "Join" flow or magic code
    wait(page)

    join_button = page.locator('button:has-text("Join")')
    if join_button.count() > 0:
        print("Join button found. This is a new user.")
        join_button.click()
    else:
        print("Join button not found. This is an existing user. Use Magic Code to sign in.")
        magic_code = "123456"
        print(f"Magic code: {magic_code}")

        validate_code_input = page.locator('input[data-testid="validateCode"]')
        expect(validate_code_input).to_be_visible()
        validate_code_input.fill(magic_code)
        try:
            page.get_by_role(
                "button",
                name=re.compile(r"^(continue|sign.?in)$", re.I),
            ).click(timeout=20000)
        except Exception:
            pass

    return browser, context, page


def _export_and_clean_trace(context, trace_enabled: bool):
    """
    Helper to stop tracing, export the trace, and clean it.
    """
    if not trace_enabled:
        return

    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_submit_expense_domain_recipient(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        # Step 1: Login user
        browser, context, page = login(p, trace_enabled=trace_enabled)

        try:
            # Step 2: Click on + icon and click on "Submit Expense"
            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            expect(plus_icon).to_be_visible()
            plus_icon.click()

            # Step 3: Click on the "Submit Expense" button
            submit_expense_button = page.locator('div[aria-label="Submit expense"]')
            expect(submit_expense_button).to_be_visible()
            submit_expense_button.click()

            # Step 4: Click on "Manual" button and enter amount
            manual_button = page.locator('button[aria-label="Manual"]')
            expect(manual_button).to_be_visible()
            manual_button.click()

            # Step 5: Fill in an amount
            page.locator('input[role="presentation"]').fill("20")

            # Step 6: Click on Next button
            next_button = page.locator("#numPadContainerView").get_by_role("button", name="Next")
            expect(next_button).to_be_visible()
            next_button.click()

            # Step 7: Input recipient email
            input_recipient = page.get_by_test_id("selection-list-text-input")
            expect(input_recipient).to_be_visible()
            input_recipient.fill("+@applause.expensifail.com")

            # Step 8: Wait for 3 seconds
            wait(page, 3)

            # Step 9: Verify that no results found message is displayed
            expect(page.get_by_text("No results found")).to_be_visible()

        finally:
            # Stop and clean trace if enabled
            _export_and_clean_trace(context, trace_enabled)

            # Close context and browser
            context.close()
            browser.close()
