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
    return f"nitish.expensify+{98}@gmail.com"


def get_magic_code():
    return "123456"


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool,
    first_name: str = "Milan",
    last_name: str = "T",
):
    """
    Launches a browser, creates a context (+ tracing if enabled) and logs the
    provided user in.  Returns (browser, context, page).
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

    context = browser.new_context()

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation

    # ----------------- original login flow -----------------
    user_email = get_user_email()
    first_name = "John"
    last_name = "Doe"
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()

    try:
        expect(page.get_by_role("button", name="Sign in")).to_be_visible()
        page.get_by_test_id("validateCode").fill(get_magic_code())
        try:
            page.get_by_role(
                "button",
                name=re.compile(r"^(continue|sign.?in)$", re.I),
            ).click(timeout=20000)
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


def test_close_account_bottom_margin(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = context = page = None
        try:
            # Create browser/context/page and login
            browser, context, page = login_user(p, trace_enabled)

            # ----------------- original assertions begin -----------------
            profile = page.locator('button[aria-label="My settings"]')
            expect(profile).to_be_visible()
            profile.click()

            troubleshoot = page.locator('div[role="menuitem"]').nth(9)
            expect(troubleshoot).to_be_visible()
            troubleshoot.click()

            forced_offline = page.locator('button[aria-label="Force offline"]')
            expect(forced_offline).to_be_visible()
            forced_offline.click()

            dob = page.locator('div[role="menuitem"]').nth(4)
            expect(dob).to_be_visible()
            dob.click()

            page.locator("text='Close account'").click()
            close_account = page.locator('button[role="button"]').last
            expect(close_account).to_be_visible()

            close_account_mb = close_account.evaluate(
                "element => window.getComputedStyle(element.parentElement).marginBottom"
            )

            if close_account_mb == "12px":
                assert False, "Close account button moves up when offline."
            else:
                assert True, "Close account button does not moves up when offline."
            # ----------------- original assertions end -----------------

        finally:
            # Handle tracing stop & cleaning
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close browser if it was created
            if browser is not None:
                browser.close()
