import os
import re
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright


def get_user_email():
    return f"nitish.expensify+{9}@gmail.com"


def get_magic_code():
    return "123456"


def login_user(p, trace_enabled: bool = False):
    """
    Launches a browser, creates a context (+ tracing if enabled) and returns
    (browser, context, page) after completing the login flow.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=1000,
    )
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install guard immediately after page creation

    user_email = get_user_email()
    first_name = "John"
    last_name = "Doe"
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()

    # Attempt magic‐code flow; fall back to sign-up if needed
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
        # Sign-up flow
        page.get_by_role("button", name="Join").click()

    # Handle potential “Complete your profile” screen
    try:
        expect(page.locator('div[aria-label="Track and budget expenses"]').nth(0)).to_be_visible()
        page.locator('div[aria-label="Track and budget expenses"]').nth(0).click()
        page.locator('input[name="fname"]').nth(0).fill(first_name)
        page.locator('input[name="lname"]').nth(0).fill(last_name)
        page.get_by_role("button", name="Continue").nth(0).click()
    except Exception:
        pass

    # Close “Get started here” modal if present
    try:
        get_started_here = page.locator("div", has_text="Get started here!").nth(-1)
        expect(get_started_here).to_be_visible()
        page.get_by_label("Close").nth(-1).click()
    except Exception:
        pass

    return browser, context, page


def test_26228(pytestconfig):
    """
    Original test wrapped inside try/finally with optional Playwright tracing.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = context = page = None  # initialise for finally block

    with sync_playwright() as p:
        try:
            # Login and obtain browser / context / page
            browser, context, page = login_user(p, trace_enabled=trace_enabled)

            # Original assertions / steps
            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            expect(plus_icon).to_be_visible()
            plus_icon.click()

            start_chat_button = page.locator('div[aria-label="Start chat"]')
            expect(start_chat_button).to_be_visible()
            start_chat_button.press("Enter")

            input_box = page.get_by_test_id("selection-list-text-input")
            expect(input_box).to_be_visible()

        finally:
            # Stop tracing and clean trace file(s) if enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close browser if it was created
            if browser is not None:
                browser.close()
