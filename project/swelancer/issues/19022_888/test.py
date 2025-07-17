import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._browser_context import BrowserContext
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool = False,
    first_name: str = "Ayush",
    last_name: str = "G",
) -> tuple[Browser, BrowserContext, Page]:
    """
    Logs in a user on NewDot and optionally starts tracing immediately after the
    browser context is created.

    Returns
    -------
    (browser, context, page)
    """
    browser: Browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
    )

    context: BrowserContext = browser.new_context()

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)

    # --- Begin login workflow ---
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill("checkissue+19022@gmail.com")
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(5000)
    except Exception:
        # If the button isn't present, proceed without failing.
        pass

    if page.locator("text=What do you want to do today?").is_visible():
        page.wait_for_timeout(1000)
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(1000)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(1000)
    # --- End login workflow ---

    return browser, context, page


def test_save_description_in_submitted_expense(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = context = page = None  # pre-declare for finally block
        try:
            browser, context, page = login_user(p, trace_enabled=trace_enabled)

            # ----------------- Begin test steps -----------------
            page.locator('span:text("Ayush G (you)")').click()

            plus_create_icon = page.locator('button[aria-label="Create"]').last
            expect(plus_create_icon).to_be_visible()
            plus_create_icon.click()

            page.get_by_text("Assign task").click()
            page.wait_for_timeout(2000)

            page.get_by_role("textbox", name="Title").fill("title1")
            page.get_by_role("textbox", name="Title").press("Tab")
            page.get_by_role("textbox", name="Description (optional)").fill("description1")
            page.get_by_role("button", name="Next").click()
            page.locator('button div:text("Confirm task")').click()
            page.wait_for_timeout(1000)

            task_title_div = page.locator('div[data-testid="comment"] div:has-text("title1")')
            task_title_div.click()

            expanded_task_title_div = page.locator('div[aria-label="description1"]').first
            expanded_task_title_div.click()
            page.wait_for_timeout(1000)

            # Delete description characters one by one (13 backspaces)
            for _ in range(13):
                page.keyboard.press("Backspace")

            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Save").click()
            page.wait_for_timeout(2000)

            field_required_warning = page.locator('div:text("This field is required.")')
            expect(field_required_warning).to_be_hidden()
            # ----------------- End test steps -----------------

        finally:
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            if context is not None:
                context.close()

            if browser is not None:
                browser.close()
