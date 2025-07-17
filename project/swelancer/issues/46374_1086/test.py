import os
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._page import Page
from playwright.async_api import async_playwright, expect
from utils.browser import close_browser
from utils.login import complete_onboarding, sign_up
from utils.online_guard import install_online_guard_async


async def start_browser(headless=True, mobile_device=None, launch_args=None):
    if launch_args is None:
        launch_args = ["--ignore-certificate-errors"]
    playwright = await async_playwright().start()
    device_profile = playwright.devices[mobile_device] if mobile_device else {}
    browser = await playwright.chromium.launch(headless=headless, args=launch_args)
    context = await browser.new_context(ignore_https_errors=True, **device_profile)
    page = await context.new_page()
    return context, page, playwright


async def track_expense(page: Page):
    await page.locator('button[aria-label="Start chat (Floating action)"]').click()
    await page.locator('div[aria-label="Track expense"]').click()
    await page.locator('button:has-text("Got it")').click()
    await page.locator('button[aria-label="Manual"]').click()
    await page.locator('input[role="presentation"]').fill("50")
    await page.locator('button[data-listener="Enter"]', has_text="Next").first.click()
    await page.locator('button:has-text("Track expense")').click()


@pytest.mark.asyncio
async def test_issue_46374(pytestconfig):
    """
    Sign up and check if Inbox is visible
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    context = page = playwright = None  # placeholders for finally block safety
    try:
        # Start browser
        context, page, playwright = await start_browser(
            launch_args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            mobile_device="iPhone 12 Pro",
        )
        await install_online_guard_async(context, page)

        # Start tracing if enabled
        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Set up email for the test
        derived_email = "testuserexpensify+15114188@gmail.com"

        # Step 1: Sign up / Login
        await sign_up(page, derived_email)

        # Complete onboarding
        try:
            await complete_onboarding(page)
        except Exception:
            pass

        # Verify Inbox visibility
        await expect(page.get_by_label("Inbox")).to_be_visible()
        await page.mouse.click(0, 0)

        # Step 2: Track expense
        await track_expense(page)
        await page.wait_for_timeout(1000)

        # Step 3: Navigate back and search
        back_button = page.locator('button[aria-label="Back"]')
        await back_button.wait_for()
        await back_button.click()

        await page.wait_for_timeout(4000)
        await page.locator('button[aria-label="Search"]').click()
        button = page.get_by_test_id("selection-list").locator('button[tabindex="0"]')
        await button.wait_for()

        # Long press on the expense.
        await button.hover()
        await page.mouse.down()
        await page.wait_for_timeout(1000)
        await page.mouse.up()

        # Select and delete the expense
        await page.locator('div[aria-label="Select"]').click()
        await page.locator('button:has-text("1 selected")').click()
        await page.locator('div[aria-label="Delete"]').click()
        await page.locator('button:has-text("Delete")').click()

        await page.wait_for_timeout(1000)
        b_button = page.locator('button[aria-label="Back"]')
        assert await b_button.count() == 0, "Back button was not found, This is accepted."

    finally:
        if context:
            # Stop tracing and clean up trace if it was started
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                await context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close browser resources
            await close_browser(context, page, playwright)
