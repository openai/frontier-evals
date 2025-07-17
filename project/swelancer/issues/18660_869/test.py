import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync


def create_user(page: Page, email: str, firstname: str, lastname: str):
    onboarding = False
    page.get_by_test_id("username").fill(email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()
    page.wait_for_timeout(4000)
    if page.locator("text=What do you want to do today?").is_visible():
        onboarding = True
        page.get_by_text("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(firstname)
        page.get_by_role("textbox", name="Last name").fill(lastname)
        page.get_by_role("button", name="Continue").click()

    return email, onboarding


def _maybe_start_tracing(context, trace_enabled: bool):
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _maybe_stop_tracing(context, trace_enabled: bool):
    if trace_enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_generic(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        context_1 = browser.new_context()
        context_2 = browser.new_context()

        _maybe_start_tracing(context_1, trace_enabled)
        _maybe_start_tracing(context_2, trace_enabled)

        try:
            # create users
            page_1 = context_1.new_page()
            install_online_guard_sync(context_1, page_1)
            page_1.goto("https://dev.new.expensify.com:8082/")
            email = "jellystarfish99+186609@gmail.com"
            email_1, onboarding1 = create_user(page_1, email, "John", "Doe")

            page_2 = context_2.new_page()
            install_online_guard_sync(context_2, page_2)
            page_2.goto("https://dev.new.expensify.com:8082/")
            email = "jellystarfish99+1866010@gmail.com"
            email_2, onboarding2 = create_user(page_2, email, "Jane", "Doe")

            fab = page_1.locator('button[aria-label="Start chat (Floating action)"]')
            expect(fab).to_be_visible()
            fab.click()
            page_1.wait_for_timeout(1000)

            chat = page_1.get_by_text("Start chat")
            expect(chat).to_be_visible()
            chat.click()
            page_1.wait_for_timeout(1000)

            page_1.locator('input[aria-label="Name, email, or phone number"]').fill(email_2)
            page_1.wait_for_timeout(2000)

            if onboarding1:
                page_1.get_by_text("Jane Doe").click()
            else:
                page_1.get_by_text("jellystarfish99+1866010@gmail.com").last.click()

            page_1.wait_for_timeout(1000)

            create = page_1.locator('button[aria-label="Create"]').nth(1)
            expect(create).to_be_visible()
            create.click()
            page_1.wait_for_timeout(1000)

            task = page_1.get_by_text("Assign task")
            expect(task).to_be_visible()
            task.click()
            page_1.wait_for_timeout(1000)

            page_1.locator('input[aria-label="Title"]').fill("Test")

            next_button = page_1.locator("button", has_text="Next")
            expect(next_button).to_be_visible()
            next_button.click()
            page_1.wait_for_timeout(1000)

            description_click = page_1.get_by_text("Description").nth(1)
            expect(description_click).to_be_visible()
            description_click.click()
            page_1.wait_for_timeout(1000)

            description_field = page_1.locator('div[aria-label="Description (optional)"]').nth(3)
            is_focused = description_field.evaluate("element => element === document.activeElement")

            assert is_focused
            page_1.wait_for_timeout(2000)

        finally:
            _maybe_stop_tracing(context_1, trace_enabled)
            _maybe_stop_tracing(context_2, trace_enabled)
            context_1.close()
            context_2.close()
            browser.close()
