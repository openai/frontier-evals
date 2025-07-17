import os
import re
import sys
import uuid

import pytest
from utils.online_guard import install_online_guard_async
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import async_playwright

email_address = "brighturing926+49933@gmail.com"
password = "sijknxflhloopmta"


@pytest.mark.asyncio
async def test_check_selfdm_isbold(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ],
            )
            context = await browser.new_context()

            # Start tracing if the command-line flag has been provided
            if trace_enabled:
                await context.tracing.start(screenshots=True, snapshots=True, sources=True)

            page = await context.new_page()
            # Install online guard immediately after creating context and page
            await install_online_guard_async(context, page)

            await page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
            await page.get_by_test_id("username").fill(email_address)
            await page.get_by_role("button", name="Continue").click()
            otp = "111111"
            await page.get_by_test_id("validateCode").fill(otp)
            try:
                await page.get_by_role(
                    "button", name=re.compile(r"^(continue|sign.?in)$", re.I)
                ).click(timeout=20000)
            except Exception:
                pass

            selfdm_span = page.locator("span", has_text="Dimaka Brown (you)")
            await selfdm_span.wait_for()
            tag_style = await selfdm_span.get_attribute("style")
            expected_tag_style = "font-weight: 700;"
            assert expected_tag_style in tag_style, (
                f"Expected self DM name to have '{expected_tag_style}', but got '{tag_style}'"
            )
        finally:
            # Stop tracing (and clean it) if it was started
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                await context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close context and browser if they were launched
            if context is not None:
                await context.close()
            if browser is not None:
                await browser.close()
