import os
import re
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import expect
from utils.browser import close_browser, start_browser
from utils.login import NEW_DOT_URL, check_if_logged_in
from utils.online_guard import install_online_guard_async

EMAIL = "testexpensifyturing+006@gmail.com"
PASSWORD = "qqdd pegp kagr emax"


@pytest.mark.asyncio
async def test_issue_25901(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    # Start browser
    context, page, playwright = await start_browser(
        persistent=True,
        launch_args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins, site-per-process",
        ],
    )
    await install_online_guard_async(context, page)

    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    try:
        # Sign in
        # Not using the utility function here since we don't need a valid OTP when running
        # the test with the recorded flow. Any random OTP works.
        if not await check_if_logged_in(page=page, url=NEW_DOT_URL):
            await page.get_by_test_id("username").fill(EMAIL)
            await page.get_by_role("button", name="Continue").click()
            await page.get_by_test_id("validateCode").fill("123456")
            try:
                await page.get_by_role(
                    "button",
                    name=re.compile(r"^(continue|sign.?in)$", re.I),
                ).click(timeout=20000)
            except Exception:
                pass

        # Submit Expense
        await page.get_by_label("Start chat (Floating action)").click()
        await page.get_by_label("Track expense").click()

        # Got it button
        got_it_button = page.get_by_role("button", name="Got it")
        try:
            await got_it_button.wait_for(timeout=2000)
            await got_it_button.click()
        except Exception:
            # Ignore if the "Got it" button is not present
            pass

        # Click on Distance
        await page.get_by_label("Distance").click()

        # Add 1st Waypoint
        await page.get_by_test_id("IOURequestStartPage").get_by_role("button", name="Start").click()
        await (
            page.get_by_test_id("IOURequestStepWaypoint")
            .get_by_role("textbox")
            .fill("madison square")
        )
        await page.get_by_text("Madison Square Garden", exact=True).click()

        # Add 2nd Waypoint
        await page.get_by_role("button", name="Stop").click()
        await (
            page.get_by_test_id("IOURequestStepWaypoint")
            .get_by_role("textbox")
            .fill("segal rogerscasey")
        )
        await page.get_by_text("Segal Rogerscasey").first.click()

        # Add 3rd Waypoint
        await page.get_by_role("button", name="Add stop").click()
        await (
            page.get_by_test_id("IOURequestStepWaypoint")
            .get_by_role("textbox")
            .fill("frames bowling")
        )
        await page.get_by_text("Frames Bowling Lounge").first.click()

        # Check if "Delete waypoint" function works correctly
        await page.wait_for_timeout(3000)  # wait for component to render
        await page.get_by_role("button", name="Segal Rogerscasey").click()
        await page.get_by_label("More").click()
        await expect(page.get_by_label("Delete waypoint")).to_be_visible()
        await page.get_by_label("Delete waypoint").click()
        await expect(page.get_by_text("Are you sure you want to")).to_be_visible()
        await page.get_by_role("button", name="Delete").click()
        await expect(page.get_by_role("button", name="Segal Rogerscasey")).not_to_be_visible()

    finally:
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Ensure browser resources are cleaned up irrespective of test outcome
        await close_browser(context, page, playwright)
