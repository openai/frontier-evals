import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def login_user(
    p: PlaywrightContextManager, first_name="A", last_name="Khan"
) -> tuple[Browser, Page, str]:
    browser = p.chromium.launch(
        headless=True,
        slow_mo=1000,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = "exxpp00112+3eds@gmail.com"
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)
    return browser, page, user_email


def set_default_currency(browser: Browser, page: Page, currency: str) -> tuple[Browser, Page]:
    default_currency_button = page.locator('div[role="menuitem"]').get_by_text("Default currency")
    expect(default_currency_button).to_be_visible()
    default_currency_button.click()
    input_currency = page.locator('input[aria-label="Default currency"]')
    input_currency.fill(currency)
    input_currency.press("Enter")
    return browser, page


def test_submit_expense_domain_recipient(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = None
        context = None  # will be set once we have a page
        try:
            browser, page, user_email = login_user(p)
            context = page.context

            if trace_enabled:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # ---------- Original test steps ----------
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_label("New workspace").click()

            browser, page = set_default_currency(browser, page, "BDT - Tk")

            page.get_by_text("More features").click()
            page.get_by_label("Set up custom fields for").click()
            page.get_by_role("button", name="Upgrade").click()
            page.get_by_role("button", name="Got it, thanks").click()
            page.get_by_test_id("WorkspaceInitialPage").get_by_text("Report fields").click()
            page.get_by_role("button", name="Add field").click()
            page.get_by_test_id("CreateReportFieldsPage").get_by_text("Name").click()
            page.get_by_role("textbox", name="Name").fill("Test Field")
            page.get_by_test_id("TextSelectorModal").get_by_role("button", name="Save").click()
            page.get_by_test_id("CreateReportFieldsPage").get_by_text("Type").click()
            page.get_by_label("Text").click()
            page.get_by_role("button", name="Save").click()
            page.get_by_test_id("WorkspaceInitialPage").get_by_label("Back").click()
            page.get_by_label("Inbox").click()
            page.locator(
                'button[aria-label="Navigates to a chat"]',
                has_text="A Khan's Workspace",
            ).click()
            page.locator('button[aria-label="Create"]').last.click()
            page.get_by_text("Submit expense", exact=True).click()
            page.get_by_label("Manual").click()
            page.get_by_placeholder("0").fill("1")
            page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
            page.get_by_text("Merchant").click()
            page.get_by_role("textbox", name="Merchant").click()
            page.get_by_role("textbox", name="Merchant").fill("12")
            page.get_by_role("button", name="Save").click()
            page.get_by_role("button", name="Submit BDT 1.00").click()
            page.get_by_label("View details").click()
            expect(page.get_by_role("menuitem").get_by_text("Test Field")).to_be_visible()
            # ---------- End of original test steps ----------

        finally:
            # Stop tracing and clean trace file if enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the context if it was created
            if context is not None:
                context.close()

            # Close the browser if it was opened
            if browser is not None:
                browser.close()
