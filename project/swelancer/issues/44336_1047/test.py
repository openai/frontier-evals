import os
import re
import sys
import uuid

import pytest

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import async_playwright, expect
from utils.browser import close_browser
from utils.login import complete_onboarding, sign_up
from utils.online_guard import install_online_guard_async
from utils.trace_cleaner import trace_cleaner


async def start_browser_with_proxy():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        slow_mo=500,
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = await browser.new_context(ignore_https_errors=True)
    page = await context.new_page()
    await install_online_guard_async(context, page)
    return context, page, playwright


# Test function
@pytest.mark.asyncio
async def test_delete_expense_with_comments(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    # Email to use for the test
    derived_email = "testuserexpensify+26443232@gmail.com"

    # Start browser and (potentially) tracing
    context, page, playwright = await start_browser_with_proxy()
    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    try:
        # Step 1: Login / Sign up
        await sign_up(page, derived_email)

        # Complete onboarding
        try:
            await complete_onboarding(page)
        except Exception:
            pass

        # Step 2: Submit first expense
        await expect(page.get_by_label("Inbox")).to_be_visible()
        await page.get_by_label("Start chat (Floating action)").click()
        await page.get_by_text("Submit expense").first.click()

        await page.get_by_label("Manual").click()
        await page.get_by_placeholder("0").fill("100")
        await page.locator("#numPadContainerView").get_by_role("button", name="Next").click()

        await page.get_by_test_id("selection-list-text-input").fill("testuser@gmail.com")
        await page.get_by_test_id("selection-list").locator('button[tabindex="0"]').click()
        await page.get_by_role("button", name=re.compile(r"Submit [\w$€£¥]+")).click()
        await page.wait_for_timeout(2000)

        # Add comments to the expense
        view_details = page.get_by_label("View details")
        await view_details.wait_for()
        await view_details.click()
        await page.wait_for_timeout(1000)
        await page.get_by_role("textbox").click()
        await page.get_by_role("textbox").fill("1")
        await page.get_by_role("textbox").press("Enter")
        await page.get_by_role("textbox").click()
        await page.get_by_role("textbox").fill("2")
        await page.get_by_role("textbox").press("Enter")

        # Step 3: Delete the submitted expense
        await page.get_by_role("button").locator('svg[width="12"][height="12"]').click()
        await page.get_by_text("Delete expense").click()
        await page.get_by_role("button", name="Delete").click()

        # Verify the expense is deleted
        await page.wait_for_timeout(2000)
        assert not await page.get_by_test_id("ReportDetailsPage").is_visible(), (
            "Expected Report Details Page to be closed"
        )

        # Step 4: Submit another expense with a different amount and recipient
        await page.get_by_label("Start chat (Floating action)").click()
        await page.get_by_text("Submit expense").first.click()
        await page.get_by_placeholder("0").fill("200")
        await page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
        await page.get_by_test_id("selection-list-text-input").fill("testuser1@gmail.com")
        await page.get_by_label("testuser1@gmail.com").click()
        await page.get_by_role("button", name=re.compile(r"Submit [\w$€£¥]+")).click()

        # Step 5: Submit a third expense and add comments
        await page.locator('button[aria-label="Create"]').last.click()
        await page.get_by_text("Submit expense").click()
        await page.get_by_placeholder("0").fill("300")
        await page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
        await page.get_by_role("button", name=re.compile(r"Submit [\w$€£¥]+")).click()

        # Add comments and delete the expense
        await view_details.wait_for()
        await view_details.click()
        await page.get_by_text("Cash").first.click()
        await page.get_by_role("textbox").click()
        await page.get_by_role("textbox").fill("1")
        await page.get_by_role("textbox").press("Enter")
        await page.get_by_role("textbox").click()
        await page.get_by_role("textbox").fill("2")
        await page.get_by_role("textbox").press("Enter")

        # Step 6: Delete the last submitted expense
        await page.get_by_label(re.compile(r".*300\.00 expense")).last.click()
        await page.get_by_text("Delete expense").click()
        await page.get_by_role("button", name="Delete").click()

        # Verify the last expense is deleted
        await page.wait_for_timeout(2000)
        assert not await page.get_by_test_id("ReportDetailsPage").is_visible(), (
            "Expected Report Details Page to be closed"
        )

    finally:
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Ensure the browser is closed regardless of test outcome
        await close_browser(context, page, playwright)
