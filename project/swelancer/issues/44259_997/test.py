import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import (
    Browser,
    Page,
    Playwright,
    expect,
    sync_playwright,
)


def generate_random_email() -> str:
    return "srkiitd+44259d@gmail.com"


def login_user(
    p: Playwright,
    trace_enabled: bool = False,
    first_name: str = "John",
    last_name: str = "Doe",
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launches a browser, creates a context & page, completes the basic onboarding
    flow and returns the browser, context, page and generated user email.
    """
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    # Create a context right away so that tracing can begin immediately.
    context = browser.new_context()

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Install the online guard immediately after page creation

    user_email = generate_random_email()

    # Begin login / onboarding flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)

    # Handle potential extra "Continue" click if the OTP prompt shows up twice
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    page.wait_for_timeout(5000)

    # Handle optional onboarding steps (only show up for new accounts)
    try:
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(1000)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    return browser, context, page, user_email


def _stop_trace_and_clean(context: BrowserContext, trace_enabled: bool) -> None:
    """
    Helper to stop tracing for a given context and run the trace cleaner,
    guarding against missing traces if tracing wasn't enabled.
    """
    if not trace_enabled:
        return

    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_44259(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser: Browser | None = None
        context: BrowserContext | None = None
        try:
            # Login / onboarding
            browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)

            # Begin actual assertions / interactions
            personal_chat = page.get_by_text("(you)")
            expect(personal_chat).to_be_visible()
            personal_chat.click()
            page.wait_for_timeout(1000)

            message = "Hello my friend"
            message_box = page.get_by_role("textbox")
            message_box.fill(message)
            page.wait_for_timeout(1000)

            # Reload and verify cursor retains position
            page.reload()
            page.wait_for_timeout(10000)

            refreshed_message_box = page.get_by_role("textbox")
            cursor_position = refreshed_message_box.evaluate(
                "el => window.getSelection().anchorOffset"
            )

            assert cursor_position == len(message), (
                f"Expected cursor position {len(message)}, but got {cursor_position}"
            )
        finally:
            # Always attempt to stop tracing and clean up.
            if context is not None:
                _stop_trace_and_clean(context, trace_enabled)
                context.close()  # Ensure context is closed before browser

            if browser is not None:
                browser.close()
