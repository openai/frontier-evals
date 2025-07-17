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


async def check_if_wrapped(element):
    is_wrapped = await element.evaluate(
        """(element) => {
            const lineHeight = parseFloat(getComputedStyle(element).lineHeight);
            const lines = element.clientHeight / lineHeight;
            return lines > 1;
        }"""
    )
    return is_wrapped


@pytest.mark.asyncio
async def test_workspace_header(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    derived_email = "testuserexpensify+25215188@gmail.com"

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = await browser.new_context(
        ignore_https_errors=True, viewport={"width": 375, "height": 812}
    )

    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = await context.new_page()
    await install_online_guard_async(context, page)

    try:
        # Step 1: Login / Sign up
        await sign_up(page, derived_email)

        # Complete onboarding
        try:
            await complete_onboarding(page)
        except Exception:
            pass

        # Navigate to Workspaces and perform assertions
        await expect(page.get_by_label("Inbox")).to_be_visible()
        await page.mouse.click(0, 0)
        await page.locator('button[aria-label="My settings"]').click()
        await page.locator('div[aria-label="Workspaces"]').click()

        back_button_parent = page.locator("#backButton").locator("..")
        workspaces_div = back_button_parent.locator("div", has_text="Workspaces").last

        if await workspaces_div.count() == 0:
            print("No 'Workspaces' div found.")
            return

        is_wrapped = await check_if_wrapped(workspaces_div)
        assert not is_wrapped, "Workspaces header is wrapped into multiple lines!"

    finally:
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        await close_browser(context, page, playwright)
