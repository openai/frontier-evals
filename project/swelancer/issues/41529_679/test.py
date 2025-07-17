import os
import re
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

PROXY = False
PROXY_OTP = True

from playwright.async_api import Page, async_playwright, expect
from utils.browser import close_browser
from utils.email_handler import get_specific_derived_email
from utils.online_guard import install_online_guard_async

BASE_EMAIL = "testerfreelancer555@gmail.com"
PASSWORD = "snct ajbr yadx kwlq"
ISSUE_NUMBER = "41529"


async def sign_in_new_dot(page: Page, email: str, password: str):
    """
    Sign in into the new Expensify dot.
    """
    await page.goto("https://dev.new.expensify.com:8082", timeout=60000)
    await page.get_by_test_id("username").fill(email)
    await page.get_by_role("button", name="Continue").click()

    # Await OTP (dummy value for test)
    otp = "123456"
    await page.wait_for_timeout(1000)
    await page.get_by_test_id("validateCode").fill(otp)
    try:
        await page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass

    # Wait sign in to complete
    await page.get_by_text("Please enter the magic code").wait_for(state="hidden")


@pytest.mark.asyncio
async def test_distance_rate_and_workflow_graying_out(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    proxy = {"server": "http://127.0.0.1:8080/"} if PROXY else None
    launch_args = [
        "--ignore-certificate-errors",
        "--disable-web-security",
        "--disable-features=IsolateOrigins,site-per-process",
    ]

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        slow_mo=2000, headless=True, args=launch_args, proxy=proxy
    )
    context = await browser.new_context()
    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = await context.new_page()
    await install_online_guard_async(context, page)

    try:
        user_email = get_specific_derived_email(email=BASE_EMAIL, suffix=ISSUE_NUMBER)
        await sign_in_new_dot(page, email=user_email, password=PASSWORD)

        settings = page.locator('button[aria-label="My settings"]')
        await expect(settings).to_be_visible()
        await settings.click()

        # Switch to offline mode and navigate through the UI
        await context.set_offline(True)
        await page.locator('div[aria-label="Workspaces"]').click()
        await page.locator('button[aria-label="New workspace"]').first.click()
        await page.locator('text="More features"').click()
        await page.locator('button[aria-label="Add, update, and enforce rates."]').click()
        await page.locator('button[aria-label="Configure how spend is approved and paid."]').click()

        # Check that Distance rates & Workflows are visible
        distance_rates = page.get_by_test_id("WorkspaceInitialPage").get_by_label("Distance rates")
        await expect(distance_rates).to_be_visible()
        workflows = page.get_by_test_id("WorkspaceInitialPage").get_by_label("Workflows")
        await expect(workflows).to_be_visible()

        # Validate that Distance Rates section is grayed out
        await distance_rates.click()
        distance_rate_row = (
            page.locator("button").filter(has_text="0.6700 / mileEnabled").locator("xpath=../..")
        )
        is_dr_grayed_out = await page.evaluate(
            """
            (element) => {
                const computedStyle = window.getComputedStyle(element);
                return computedStyle.opacity === '0.5';
            }
            """,
            await distance_rate_row.element_handle(),
        )

        # Validate that Workflows section is grayed out
        await workflows.click()
        delay_submission_workflow = page.get_by_label(
            "Delay expense submissions based on a custom schedule, or keep this option disabled to maintain realtime spend visibility."
        ).locator("xpath=../../../..")
        is_wf_grayed_out = await page.evaluate(
            """
            (element) => {
                const computedStyle = window.getComputedStyle(element);
                return computedStyle.opacity === '0.5';
            }
            """,
            await delay_submission_workflow.element_handle(),
        )

        assert is_dr_grayed_out and is_wf_grayed_out, (
            "Distance route or workflow section is not grayed out after going offline."
        )

    finally:
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        await close_browser(context, page, playwright)
