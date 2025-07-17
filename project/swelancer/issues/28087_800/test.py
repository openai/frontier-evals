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
    return f"nitish.expensify+{84}@gmail.com"


def get_magic_code():
    return "123456"


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool,
    first_name: str = "Milan",
    last_name: str = "T",
):
    """
    Launches a browser, logs-in (or signs-up) a user and returns
    (browser, context, page).  If `trace_enabled` is True, tracing
    starts immediately after the context is created.
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
    # `Browser.new_page()` implicitly creates a new incognito context.
    page = browser.new_page()
    context = page.context

    # Install online guard immediately after page and context creation
    install_online_guard_sync(context, page)

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    user_email = get_user_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
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
        page.get_by_role("button", name="Join").click()
    try:
        expect(page.locator('div[aria-label="Track and budget expenses"]').nth(0)).to_be_visible()
        page.locator('div[aria-label="Track and budget expenses"]').nth(0).click()
        page.locator('input[name="fname"]').nth(0).fill(first_name)
        page.locator('input[name="lname"]').nth(0).fill(last_name)
        page.get_by_role("button", name="Continue").nth(0).click()
    except Exception:
        pass
    try:
        get_started_here = page.locator("div", has_text="Get started here!").nth(-1)
        expect(get_started_here).to_be_visible()
        page.get_by_label("Close").nth(-1).click()
    except Exception:
        pass

    return browser, context, page


def test_save_description_in_submitted_expense(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Launch browser and log in the user (tracing handled inside)
            browser, context, page = login_user(p, trace_enabled)

            # Perform the remaining test steps
            page.locator('button[aria-label="Start chat (Floating action)"]').last.click()
            page.locator("div", has_text="New Workspace").last.click()
            page.locator('button[aria-label="Back"]').last.click()
            page.locator('button[aria-label="Inbox"]').last.click()
            page.locator('div[placeholder="Write something..."]').last.fill("yo!")
            page.keyboard.press("Enter")
            page.locator("div", has_text="Today at").last.click(button="right")
            page.locator("div", has_text="Reply in thread").last.click()
            page.locator('div[placeholder="Write something..."]').last.fill("Levi")
            page.keyboard.press("Enter")
            page.locator('button[aria-label="yo!"]').last.click()
            page.locator('div[aria-label="Leave"]').last.click()
            side_bar = page.locator('div[data-testid="BaseSidebarScreen"]').last
            expect(side_bar).to_be_visible()

        finally:
            # Handle tracing export / cleaning
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Ensure the context and browser are closed even if the test failed
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
