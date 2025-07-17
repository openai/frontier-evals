import os
import sys
import uuid

import pytest
from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

# Add project root to PYTHONPATH
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

static_email = "testuserexpensify1+82@gmail.com"


def login_user(page: Page, email: str, first_name: str, last_name: str):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.locator('button[tabindex="0"]').click()
    join_button = page.get_by_role("button", name="Join")
    join_button.click()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()


def create_workspace_and_add_member(page: Page, member_email: str):
    page.locator('button[aria-label="Workspaces"]').click()
    page.get_by_test_id("WorkspaceSwitcherPage").get_by_role("button", name="New workspace").click()
    page.locator('div[aria-label="Members"]').click()
    page.locator("button", has_text="Invite member").click()
    page.locator('input[aria-label="Name, email, or phone number"]').fill(member_email)
    page.locator("button", has_text=member_email).click()
    page.locator('button[data-listener="Enter"]', has_text="Next").click()
    page.locator('button[data-listener="Enter"]', has_text="Invite").click()
    page.locator('button[aria-label="Back"]').click()
    page.locator('button[aria-label="Back"]').click()


def browse_to_workspace(page: Page, workspace_name: str):
    page.locator('button[aria-label="Workspaces"]').click()
    try:
        page.locator(f'button[aria-label="{workspace_name}"]').click()
    except Exception:
        page.locator(f'button[aria-label="{workspace_name}"]').nth(1).click()
    page.locator('button[aria-label="Navigates to a chat"]', has_text=workspace_name).nth(0).click()


def submit_expense(page: Page, amount: int, merchant: str):
    page.locator('button[aria-label="Create"]').nth(2).click()
    page.locator('div[aria-label="Submit expense"]').click()
    page.locator('button[aria-label="Manual"]').click()
    page.locator('input[placeholder="0"]').fill(str(amount))
    page.locator('button[data-listener="Enter"]', has_text="Next").nth(0).click()
    page.locator('div[role="menuitem"]', has_text="Merchant").click()
    page.locator('input[aria-label="Merchant"]').fill(merchant)
    page.locator('button[data-listener="Enter"]', has_text="Save").click()
    page.locator('button[data-listener="Enter"]', has_text="Submit").click()


def wait_for_text_with_assertion(
    page: Page,
    locator,
    text: str,
    max_retries: int = 10,
    interval: int = 1,
    assertion_message: str = "Text not found",
):
    """
    Polls the given locator’s inner text until the expected text is present or
    the max number of retries is reached.
    """
    for attempt in range(max_retries):
        try:
            if text in locator.inner_text():
                print(f"'{text}' found in the inner text.")
                return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with exception: {e}")
        page.wait_for_timeout(interval * 1000)
    raise AssertionError(assertion_message)


@pytest.mark.usefixtures()
def test_duplicate_and_hold_expense(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            slow_mo=1000,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard

            user1_email = static_email
            user2_email = "testuser@gmail.com"
            first_name1, last_name1 = ("Chris", "Brown")
            workspace_name = f"{first_name1} {last_name1}'s Workspace"

            login_user(page, user1_email, first_name1, last_name1)
            create_workspace_and_add_member(page, user2_email)
            browse_to_workspace(page, workspace_name)

            # Submit expenses
            submit_expense(page, 100, "Merchant 1")
            submit_expense(page, 200, "Merchant 2")
            submit_expense(page, 100, "Merchant 1")

            # Verify duplicate / hold flow
            page.locator(
                'button[aria-label="View details"]', has_text=f"{workspace_name} owes:"
            ).last.click()

            first_message = page.locator('div[aria-label="Cash"][data-tag="pressable"]').first

            wait_for_text_with_assertion(
                page,
                first_message,
                "Duplicate",
                max_retries=10,
                interval=1,
                assertion_message="Expected 'Duplicate' text not found in the first message.",
            )

            first_message.hover()
            page.locator('button[aria-label="Menu"]').click()
            page.locator('div[aria-label="Hold"]').click()
            page.locator('input[aria-label="Reason"]').fill("Random reason")
            page.locator('button[data-listener="Enter"]', has_text="Hold expense").click()

            wait_for_text_with_assertion(
                page,
                first_message,
                "Hold",
                max_retries=5,
                interval=1,
                assertion_message="Expense's text not updated to 'Hold' from 'Duplicate'.",
            )

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()  # Close context before closing the browser
            browser.close()
