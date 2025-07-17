import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import random
import re
import time

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email() -> str:
    return "redx224+6856@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
) -> tuple[Browser, Page, str]:
    """
    Launches a browser, signs up a new user and optionally starts a tracing
    session on the created browser context.
    """
    browser: Browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    # `browser.new_page()` implicitly creates a new browser context.
    page: Page = browser.new_page()
    context = page.context

    # Install online guard immediately after context/page creation
    install_online_guard_sync(context, page)

    # Start tracing immediately after the context is created if requested.
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    user_email: str = generate_random_email()

    # Begin signup / onboarding flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()
    page.wait_for_selector("text=What do you want to do today?")
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(500)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(500)

    return browser, page, user_email


def test_issue_18909(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = None
        page = None
        try:
            # Launch browser / sign-up flow (with tracing if enabled)
            browser, page, user_email = login_user(p, trace_enabled=trace_enabled)

            # Perform the assertions / interactions specific to this test
            own_chat = page.get_by_text("Milan T (you)")
            expect(own_chat).to_be_visible()
            own_chat.click()
            page.wait_for_timeout(500)

            emoji = page.locator('div[aria-label="Emoji"]').nth(1)
            expect(emoji).to_be_visible()
            emoji.click()
            page.wait_for_timeout(500)

            page.locator('input[aria-label="Search"]').nth(0).fill("avsfb")
            search_div = page.locator("div").filter(has_text=re.compile(r"^Search$")).first
            page.wait_for_timeout(500)
            expect(search_div).to_have_css("padding", "8px 16px 12px")

        finally:
            # Stop tracing and clean up any generated trace files
            if trace_enabled and page is not None:
                context = page.context
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the browser context before closing the browser
            if page is not None:
                page.context.close()

            if browser is not None:
                browser.close()
