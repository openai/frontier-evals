import os
import re
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync

FIRST_NAME = "Milan"
LAST_NAME = "T"


def get_user_email() -> str:
    return f"nitish.expensify+{92}@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    *,
    trace_enabled: bool = False,
):
    """
    Launches the browser, creates a context (with optional tracing), performs
    authentication and returns the tuple (browser, context, page).
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

    # Create an explicit context so we can start/stop tracing on it.
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = get_user_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    try:
        # Existing user flow – enter OTP.
        expect(page.get_by_role("button", name="Sign in")).to_be_visible()
        page.get_by_test_id("validateCode").fill("123456")
        try:
            page.get_by_role(
                "button",
                name=re.compile(r"^(continue|sign.?in)$", re.I),
            ).click(timeout=20000)
        except Exception:
            pass
    except Exception:
        # New user flow – click Join.
        page.get_by_role("button", name="Join").click()

    return browser, context, page


def navigate_to_profile(page: Page):
    page.locator('button[aria-label="My settings"]').click()
    page.locator('div[aria-label="Profile"]').first.click()


def edit_legal_name(page: Page):
    page.locator('div[role="menuitem"]', has_text="Legal name").click()
    fname_input_field = page.locator('div[aria-label="Legal first name"] input')
    lname_input_field = page.locator('div[aria-label="Legal last name"] input')
    save_button = page.get_by_role("button", name="Save")

    fname_input_field.fill(FIRST_NAME + "÷×")
    lname_input_field.fill(LAST_NAME + "÷×")
    save_button.click()

    error_msg = page.get_by_text("Name can only include Latin characters.").all()
    assert len(error_msg) == 2


def _export_and_clean_trace(context):
    """
    Helper that stops tracing on the supplied context, writes it to a unique
    file, then runs the trace cleaner on it.
    """
    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_fake_assignee_name_removed(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Log in (context creation happens inside login_user)
            browser, context, page = login_user(p, trace_enabled=trace_enabled)

            # Perform the actions under test
            navigate_to_profile(page)
            edit_legal_name(page)

        finally:
            # Handle tracing export/cleanup if enabled
            if trace_enabled and context is not None:
                _export_and_clean_trace(context)

            # Close the browser if it was opened
            if browser is not None:
                browser.close()
