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


def login_user(
    p: PlaywrightContextManager,
    user_email: str,
    trace_enabled: bool = False,
) -> tuple[Browser, "BrowserContext", Page, str]:
    """
    Starts the browser, creates a context (optionally enabling tracing),
    logs the user in and returns useful objects for further actions.
    """
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        slow_mo=500,
        proxy={"server": "http://127.0.0.1:8080/"},
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    context_args = {
        "geolocation": {"latitude": 37.7749, "longitude": 122.4194},
        "permissions": ["geolocation"],
        "ignore_https_errors": True,
    }
    context = browser.new_context(**context_args)

    # Start tracing right after context creation if requested
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation

    page.goto("https://127.0.0.1:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("SignInPage").get_by_test_id("validateCode").fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    page.wait_for_timeout(2000)
    return browser, context, page, user_email


def test_disable_category(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser_b = context_b = page_b = None  # pre-initialise for finally block
        try:
            browser_b, context_b, page_b, user_email_b = login_user(
                p, "t1533148+123@gmail.com", trace_enabled=trace_enabled
            )

            page_b.wait_for_timeout(2000)
            page_b.get_by_test_id("7823501529575664").click()
            page_b.wait_for_timeout(1000)
            page_b.get_by_label("View details").click()
            expense1 = page_b.get_by_text("Cash").nth(1)
            expense1.click()
            page_b.wait_for_timeout(1000)
            cat = page_b.get_by_text("Category", exact=True)
            cat.click()
            page_b.wait_for_timeout(2000)

            recent_section = page_b.get_by_text("Recent")
            all_section = page_b.get_by_text("All")
            search_section = page_b.locator('input[aria-label="Search"]')
            expect(recent_section).not_to_be_visible()
            expect(all_section).not_to_be_visible()
            expect(search_section).not_to_be_visible()

            page_b.wait_for_timeout(3000)

        finally:
            if trace_enabled and context_b is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context_b.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            if context_b is not None:
                context_b.close()

            if browser_b is not None:
                browser_b.close()
