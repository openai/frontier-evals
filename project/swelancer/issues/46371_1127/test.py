import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

import pytest
from playwright.sync_api import expect, sync_playwright


@pytest.fixture(scope="function")
def page(request):
    """
    Provides a Playwright page object wrapped with tracing support that is
    enabled when the `--user-tool-trace` option is passed to pytest.
    """
    trace_enabled = request.config.getoption("--user-tool-trace")

    # Start Playwright manually so that we have explicit control over teardown
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=500,
    )

    context_args = {
        "viewport": {"width": 1024, "height": 640},
        "timezone_id": "Asia/Dhaka",
    }
    context = browser.new_context(**context_args)

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Install the online guard immediately after page creation

    try:
        yield page
    finally:
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            # Export and clean the trace
            context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        context.close()  # Ensure the context is closed before closing the browser
        browser.close()
        playwright.stop()


def test_check_unhold_button_exists(page):
    first_name = "Test"
    last_name = "Name"
    workspace_name = f"{first_name} {last_name}'s Workspace"

    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()

    email = "rgarciatammy4+173307430716@gmail.com"
    phone_or_email_input.fill(email)

    continue_button = page.locator('button[tabindex="0"]')
    expect(continue_button).to_be_visible()
    continue_button.click()

    time.sleep(1)
    page.locator("button:has-text('join')").click()
    page.wait_for_timeout(3000)

    track_and_budget_expenses_button = page.locator("text='Track and budget expenses'").count()
    if track_and_budget_expenses_button > 0:
        try:
            page.locator("text='Track and budget expenses'").click()
            page.locator('input[aria-label="First name"]').fill(first_name)
            page.locator('input[aria-label="Last name"]').fill(last_name)
            page.locator('button[data-listener="Enter"]').click()
        except Exception:
            pass
        workspace_name = f"{first_name} {last_name}'s Workspace"
    else:
        workspace_name = f"{email.capitalize()}'s Workspace"

    settings = page.locator('button[aria-label="My settings"]')
    expect(settings).to_be_visible()
    settings.click()

    page.locator('div[aria-label="Workspaces"]').click()
    page.locator('button[aria-label="New workspace"]').first.click()
    page.locator('text="More features"').click()
    page.locator('button[aria-label="Document and reclaim eligible taxes."]').click()
    page.get_by_label("Back").click()
    page.get_by_label("Inbox").click()
    page.wait_for_timeout(2000)

    workspace_count = page.get_by_test_id("lhn-options-list").get_by_text(workspace_name).count()
    print(workspace_count)

    page.locator('button[aria-label="Navigates to a chat"]').nth(2).click()
    page.wait_for_timeout(2000)

    page.get_by_role("button", name="Create").click()
    page.get_by_label("Submit expense").get_by_text("Submit expense").click()
    page.get_by_label("Manual").click()
    page.get_by_placeholder("0").fill("111")
    page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
    page.get_by_test_id("selection-list").get_by_text("Merchant").click()
    page.get_by_role("textbox", name="Merchant").fill("merchant1")
    page.get_by_role("button", name="Save").click()
    page.get_by_role("button", name="Submit").click()

    page.get_by_role("button", name="Create").click()
    page.get_by_label("Submit expense").get_by_text("Submit expense").click()
    page.get_by_label("Manual").click()
    page.get_by_placeholder("0").fill("222")
    page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
    page.get_by_test_id("selection-list").get_by_text("Merchant").click()
    page.get_by_role("textbox", name="Merchant").fill("merchant2")
    page.get_by_role("button", name="Save").click()
    page.get_by_role("button", name="Submit").click()

    page.get_by_role("button", name="Create").click()
    page.get_by_label("Submit expense").get_by_text("Submit expense").click()
    page.get_by_label("Manual").click()
    page.get_by_placeholder("0").fill("333")
    page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
    page.get_by_test_id("selection-list").get_by_text("Merchant").click()
    page.get_by_role("textbox", name="Merchant").fill("merchant3")
    page.get_by_role("button", name="Save").click()
    page.get_by_role("button", name="Submit").click()

    page.wait_for_timeout(1000)
    page.get_by_role("button", name="View details").first.click()
    page.get_by_label("Cash").nth(2).click(button="right")
    page.get_by_label("Hold").click()
    page.get_by_role("textbox", name="Reason").fill("Reason1")
    page.get_by_role("button", name="Hold expense").click()

    page.wait_for_timeout(1000)
    page.get_by_label("Cash").nth(1).click(button="right")
    page.get_by_label("Hold").click()
    page.get_by_role("textbox", name="Reason").fill("Reason2")
    page.get_by_role("button", name="Hold expense").click()

    page.wait_for_timeout(1000)
    settings.click()
    page.locator('div[aria-label="Troubleshoot"]').click()
    page.locator('button[aria-label="Force offline"]').click()

    page.wait_for_timeout(1000)
    page.get_by_label("Inbox").click()
    page.locator('button[aria-label="Navigates to a chat"]').nth(2).click()
    page.wait_for_timeout(1000)

    page.get_by_role("button", name="View details").get_by_role("button").first.click()
    page.wait_for_timeout(1000)
    page.keyboard.press("Enter")

    page.get_by_role("button", name="View details").get_by_text(f"{workspace_name} owes").click()

    page.wait_for_timeout(1000)
    page.get_by_label("Cash").get_by_text("Hold").nth(1).click()

    page.wait_for_timeout(1000)
    page.mouse.click(0, 0)
    page.get_by_role("button", name="for merchant1").nth(1).click()

    page.wait_for_timeout(1000)
    expect(page.get_by_role("button", name="Unhold")).to_be_visible(timeout=10)
