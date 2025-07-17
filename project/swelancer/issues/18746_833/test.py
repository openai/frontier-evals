import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync


def generate_random_email():
    return "testerbro7077+462468@gmail.com"


def login_user(
    p: PlaywrightContextManager, first_name="John", last_name="Doe"
) -> tuple[Browser, Page, str, str]:
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        slow_mo=500,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    page = browser.new_page()
    install_online_guard_sync(page.context, page)
    user_email = generate_random_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(2000)
    except Exception:
        pass
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(2000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(2000)
    return browser, page, first_name, last_name


def test_expensify_at_here(pytestconfig):
    """
    Test that @here mention appears as expected in the chat composer.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser, page, first_name, last_name = login_user(p)
        context = page.context  # grab the implicit context created by new_page()

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page.wait_for_timeout(2000)
            click_WS = page.locator("div", has_text=f"{first_name} {last_name} (you)").nth(-1)
            click_WS.click()
            page.wait_for_timeout(2000)
            input_text_in_chat = page.locator('div[role="textbox"]').nth(-1)
            input_text_in_chat.fill("@h")
            page.wait_for_timeout(2000)
            text_here = page.locator("div", has_text="@here").nth(-1)
            expect(text_here).to_be_visible()
            page.wait_for_timeout(2000)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            browser.close()
