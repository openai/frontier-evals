import os
import re
import sys
import uuid

import pytest
from playwright.async_api import expect
from utils.browser import close_browser, start_browser
from utils.login import NEW_DOT_URL, check_if_logged_in
from utils.online_guard import install_online_guard_async
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

EMAIL = "brighturing926+49543@gmail.com"


@pytest.mark.asyncio
async def test_issue_49543(pytestconfig) -> None:
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    workspace_name = "multitagsws45"

    # Launch the browser/context
    context, page, playwright = await start_browser(
        launch_args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins, site-per-process",
        ]
    )
    await install_online_guard_async(context, page)

    # Start tracing if requested
    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    try:
        # Step 1: Sign in
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

        await page.get_by_label("My settings").click()
        await page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").click()
        await page.wait_for_timeout(2000)
        await page.get_by_text(workspace_name, exact=True).last.click()

        await page.wait_for_timeout(3000)
        await page.get_by_test_id("WorkspaceInitialPage").get_by_label("Tags").click()
        await page.get_by_label("State").click()
        await page.locator("#California").click()

        # Validate the absence of the "Tag rules" feature
        tag_rules_element = page.get_by_text("Tag rules")
        await expect(tag_rules_element).not_to_be_visible()

    finally:
        # Stop tracing and clean trace if enabled
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Close browser/context
        await close_browser(context, page, playwright)
