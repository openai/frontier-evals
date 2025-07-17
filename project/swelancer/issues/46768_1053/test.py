import os
import sys
import uuid

import pytest
from playwright.sync_api import BrowserContext
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright.async_api import (
    Browser,
    Page,
    PlaywrightContextManager,
    async_playwright,
    expect,
)
from utils.online_guard import install_online_guard_async

EMAIL = "expensify66+543209565435065432247@gmail.com"
FIRST_NAME = "Milan"
LAST_NAME = "Tonborn"


async def simulate_netsuite_fail(page: Page, workspace_id: str):
    js_code = """
    async (workspace_id) => {
        const net_suite = {
            lastSync: {
                errorDate: "2024-12-9T10:26:57+0000",
                errorMessage: "Unable to validate NetSuite tokens",
                isAuthenticationError: true,
                isConnected: false,
                isSuccessful: false,
                source: "NEWEXPENSIFY",
                successfulDate: "",
            },
            verified: false,
        };

        await Onyx.merge(`policy_${workspace_id}`, {
            connections: {
                netsuite: net_suite,
            },
        });
    };
    """
    await page.evaluate(js_code, workspace_id)


async def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool,
    first_name: str = FIRST_NAME,
    last_name: str = LAST_NAME,
) -> tuple[Browser, "BrowserContext", Page]:
    """
    Launch a browser, create a context (with tracing if enabled), navigate through the
    onboarding flow, and return the browser, context and page.
    """
    browser: Browser = await p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://127.0.0.1:8080"},
        slow_mo=500,
    )

    context = await browser.new_context()
    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = await context.new_page()
    await install_online_guard_async(context, page)

    await page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    await page.get_by_test_id("username").fill(EMAIL)
    await page.get_by_role("button", name="Continue").click()
    await page.get_by_role("button", name="Join").click()
    await page.locator("text=What do you want to do today?").wait_for(timeout=5000)
    await page.locator("text='Track and budget expenses'").click()
    await page.get_by_role("button", name="Continue").click()
    await page.locator('input[name="fname"]').fill(first_name)
    await page.locator('input[name="lname"]').fill(last_name)
    await page.get_by_role("button", name="Continue").last.click()
    await page.get_by_text("Get Started").click(timeout=3000)

    return browser, context, page


async def create_workspace_and_enable_netsuite(page: Page) -> str:
    """
    Create a new workspace and enable NetSuite.
    """
    await page.get_by_label("My settings").click()
    await page.get_by_text("Workspaces").last.click()
    await page.get_by_text("New workspace").first.click()
    await page.get_by_test_id("WorkspaceInitialPage").get_by_text("More features").click()
    workspace_id = re.search(r"settings/workspaces/([^/]+)/", page.url).group(1)
    await page.get_by_label("Sync your chart of accounts").click()
    await page.get_by_test_id("WorkspaceInitialPage").get_by_text("Accounting").click()
    await page.get_by_label("NetSuite").get_by_role("button", name="Connect").click()
    await page.get_by_role("button", name="Upgrade").click(timeout=3000)
    await page.get_by_role("button", name="Got it, thanks").click()
    await page.get_by_test_id("NetSuiteTokenInputPage").get_by_label("Back").click()
    return workspace_id


@pytest.mark.asyncio
async def test_first_issue_error_message(pytestconfig) -> None:
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None
    async with async_playwright() as p:
        try:
            # Login and get workspace ready
            browser, context, page = await login_user(p, trace_enabled, FIRST_NAME, LAST_NAME)

            workspace_id = await create_workspace_and_enable_netsuite(page)
            await simulate_netsuite_fail(page, workspace_id)

            await expect(
                page.get_by_test_id("PolicyAccountingPage").get_by_text(
                    "Can't connect to NetSuite"
                ),
                "Error message is not visible",
            ).to_be_visible(timeout=10000)
        finally:
            # Stop tracing and clean trace if enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                await context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close browser if it was started
            if browser is not None:
                await browser.close()
