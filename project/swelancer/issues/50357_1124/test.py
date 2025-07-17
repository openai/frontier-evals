import os
import re
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import logging

from playwright.async_api import expect
from utils.browser import close_browser, start_browser
from utils.login import NEW_DOT_URL, check_if_logged_in
from utils.online_guard import install_online_guard_async

logging.basicConfig(level=logging.INFO)

EMAIL = "brighturing926+50357@gmail.com"


@pytest.mark.asyncio
async def test_issue_50357(pytestconfig):
    """
    Port of original test with Playwright tracing support wrapped in try/finally.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    context = page = playwright = None
    newWorkspceName = "qbocredentialws262634"

    try:
        # Browser / context start
        context, page, playwright = await start_browser(
            launch_args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins, site-per-process",
            ]
        )
        await install_online_guard_async(context, page)

        # Start tracing if requested
        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Step 1: Sign in if not already logged in
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

        # Navigate to workspace
        await page.get_by_label("My settings").click()
        await page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").click()
        await page.wait_for_timeout(2000)
        await page.get_by_text(newWorkspceName, exact=True).last.click()

        # Accounting tab
        await page.get_by_test_id("WorkspaceInitialPage").get_by_text("Accounting").click()

        # Attempt a sync on Expensify Accounting with Quickbooks connection
        await page.get_by_test_id("PolicyAccountingPage").get_by_label("More").click()
        await page.wait_for_timeout(3000)
        enter_credentials = page.get_by_label("Enter your Credentials")

        await expect(enter_credentials).to_be_visible()

    finally:
        # Stop tracing and clean trace if enabled
        if trace_enabled and context is not None:
            try:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                await context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            except Exception as e:
                # Log tracing errors but do not mask original test failures
                logging.error(f"Failed to stop/clean trace: {e}")

        # Always attempt to close the browser
        if context and page and playwright:
            await close_browser(context, page, playwright)
