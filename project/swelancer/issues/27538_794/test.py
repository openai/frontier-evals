import os
import re
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def get_user_email():
    return f"nitish.expensify+{87}@gmail.com"


def get_magic_code():
    return "123456"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
):
    """
    Launches a browser, signs/joins the user and returns the active page.
    If `trace_enabled` is True, tracing will start immediately after the
    browser context is created.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=2000,
    )

    # Create an explicit browser context and page
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after creation

    # Start tracing right after the context’s creation, if requested.
    if trace_enabled:
        page.context.tracing.start(screenshots=True, snapshots=True, sources=True)

    user_email = get_user_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()

    # Attempt sign-in flow
    try:
        expect(page.get_by_role("button", name="Sign in")).to_be_visible()
        page.get_by_test_id("validateCode").fill(get_magic_code())
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
    except Exception:
        # Fallback to join flow
        page.get_by_role("button", name="Join").click()

    # Handle potential onboarding flow
    try:
        expect(page.locator('div[aria-label="Track and budget expenses"]').nth(0)).to_be_visible()
        page.locator('div[aria-label="Track and budget expenses"]').nth(0).click()
        page.locator('input[name="fname"]').nth(0).fill(first_name)
        page.locator('input[name="lname"]').nth(0).fill(last_name)
        page.get_by_role("button", name="Continue").nth(0).click()
    except Exception:
        pass

    # Close potential modal if it appears
    try:
        get_started_here = page.locator("div", has_text="Get started here!").nth(-1)
        expect(get_started_here).to_be_visible()
        page.get_by_label("Close").nth(-1).click()
    except Exception:
        pass

    return page


def _stop_trace_and_cleanup(page, trace_enabled: bool):
    """
    Helper that stops tracing on `page.context` (if enabled), saves the trace
    to disk and cleans it up via `trace_cleaner`.
    """
    if not trace_enabled:
        return

    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    try:
        page.context.tracing.stop(path=trace_path)
    finally:
        # Even if stopping the trace fails, attempt cleaner only if the file exists
        if os.path.isfile(trace_path):
            trace_cleaner(trace_path)


def _close_browser(page):
    """
    Safely close the browser that owns the provided page.
    """
    try:
        page.context.close()  # Close the context first
    except Exception:
        pass
    try:
        page.context.browser.close()
    except Exception:
        pass  # Ignore any exception during browser close


def test_save_description_in_submitted_expense(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    page = None

    with sync_playwright() as p:
        try:
            page = login_user(p, trace_enabled=trace_enabled)
            page.locator('button[aria-label="Start chat (Floating action)"]').last.click()
            page.locator('text="Submit expense"').click()
            page.locator('text="Distance"').last.click()
            page.locator('text="Start"').last.click()
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "USA"
            )
            page.keyboard.press("Enter")
            page.keyboard.press("Tab")
            page.keyboard.press("Enter")
            page.locator('text="Stop"').last.click()
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "Ethiopia"
            )
            page.keyboard.press("Enter")
            page.keyboard.press("Tab")
            page.keyboard.press("Enter")
            page.locator("div", has_text="Add stop").last.click()
            page.locator('input[class="css-textinput-11aywtz r-placeholderTextColor-6taxm2"]').fill(
                "USA"
            )
            page.keyboard.press("Enter")
            page.keyboard.press("Tab")
            page.keyboard.press("Enter")
            page.locator("div", has_text="Ethiopia").last.click()
            page.locator('button[aria-label="More"]').last.click()
            page.locator('text="Delete waypoint"').last.click()
            page.locator('text="Delete"').last.click()
            error = page.locator('text="Route exceeds maximum distance limitation"').last
            expect(error).not_to_be_visible()
        finally:
            if page is not None:
                _stop_trace_and_cleanup(page, trace_enabled)
                _close_browser(page)
