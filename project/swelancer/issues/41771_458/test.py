import os
import re
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync


def launch_browser(playwright, platform: str = "desktop"):
    """
    Launches a Chromium browser and returns the browser, context, and page.

    Returning the context allows us to control tracing, and returning the
    browser lets us close everything cleanly in the finally block.
    """
    mobile_context = {}
    if platform.lower() == "mobile":
        mobile_context = {"viewport": {"width": 540, "height": 864}, "is_mobile": True}

    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(**mobile_context)
    page = context.new_page()
    install_online_guard_sync(context, page)

    return browser, context, page


def login_user_if_not_logged_in(page: Page, platform: str = "desktop"):
    page.bring_to_front()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    email = "freelancer.test.exp+17334747386155@gmail.com"
    page.locator('input[type="email"]').fill(email)
    page.locator("button", has_text="Continue").click()
    page.wait_for_timeout(3000)
    page.get_by_test_id("validateCode").fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


def verify_receipt_violation_error_message(
    page: Page, member_name: str, approval_required_amount: int
):
    page.get_by_label("Navigates to a chat").get_by_text(member_name, exact=True).click()
    page_id = page.url.rstrip("/").split("/")[-1]
    page.get_by_test_id(f"report-screen-{page_id}").get_by_role(
        "button", name="View details"
    ).first.click()
    page.wait_for_timeout(5000)
    expect(
        page.get_by_text("Expense exceeds auto approval limit of undefined"),
        "The approval limit should not be undefined",
    ).to_have_count(0)

    # Determine the correct locator for the approval error message
    visible_locator = page.get_by_text(
        f"Expense exceeds auto approval limit of ${approval_required_amount}"
    )
    approval_error_message = (
        visible_locator if visible_locator.is_visible() else visible_locator.nth(1)
    )

    expect(
        approval_error_message,
        "The approval limit should be correct",
    ).to_have_count(1)


def test_receipt_violation_should_show_correct_error_message(pytestconfig):
    """
    Verifies that the correct error message appears when a receipt violates the
    approval limit. Optionally records a Playwright trace when
    --user-tool-trace is supplied.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    playwright = sync_playwright().start()
    browser = context = page = None  # For safer reference in finally

    try:
        # Launch browser and optionally start tracing
        browser, context, page = launch_browser(playwright)
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Test steps
        login_user_if_not_logged_in(page)
        page.wait_for_timeout(2000)
        verify_receipt_violation_error_message(page, "Asaad J", 100)

    finally:
        # Handle tracing export/cleanup if enabled
        if trace_enabled and context is not None:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Close browser and stop Playwright to free resources
        if browser is not None:
            browser.close()
        playwright.stop()
