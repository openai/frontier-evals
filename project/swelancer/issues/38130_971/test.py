import base64
import io
import os
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import Page, async_playwright
from utils import login
from utils.online_guard import install_online_guard_async

PROXY_URL = os.getenv("PROXY_URL")
LOGIN_EMAIL = "testerfreelancer555+38130_rcrd@gmail.com"
EMAIL_PASSWORD = "snct ajbr yadx kwlq"

IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABLUlEQVR4nO3RQREAIAzAsIF/"
    "z0NGHjQKetczO4GuDvhdA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANw"
    "BqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQ"
    "BrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANY"
    "ArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFY"
    "A7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGY"
    "A3AGoA1AGsA9gDO4gH/iDJCTQAAAABJRU5ErkJggg=="
)


async def reproduce_scenario(page: Page):
    # Step 2: Create a new workspace
    await page.get_by_label("Workspaces").click()
    await page.get_by_label("New workspace").click()
    await page.get_by_label("Back").click()
    await page.get_by_test_id("selection-list").get_by_label("'s Workspace").first.click()
    # Workspace chat still not opening for some reason
    await page.get_by_label("Find something...").click()
    await page.get_by_test_id("selection-list-text-input").fill("Workspace")
    await page.get_by_test_id("selection-list").get_by_label("'s Workspace").first.click()

    # Step 4: Split an expense with uploaded receipt
    await page.get_by_role("button", name="Create").click()
    await page.get_by_label("Split expense").click()
    async with page.expect_file_chooser() as fc_info:
        await page.get_by_label("Choose file").click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(
        [
            {
                "name": "image.png",
                "mimeType": "image/png",
                "buffer": io.BytesIO(base64.b64decode(IMAGE_BASE64)).getvalue(),
            }
        ]
    )

    await page.get_by_role("button", name="Split expense").click()

    # Step 5: Open the expense while scanning and click split expense
    try:
        await page.get_by_text("Receipt", exact=True).click()
    except Exception:
        await page.get_by_text("Receipt", exact=True).nth(1).click()

    # Assertion
    split_expense_button = page.get_by_role("button", name="Split expense")
    # Trigger validation errors
    await split_expense_button.click()

    # Assert that the button's cursor is not set to 'not-allowed', indicating it's not disabled
    assert (
        await split_expense_button.evaluate("button => getComputedStyle(button).cursor")
        != "not-allowed"
    ), (
        "The 'Split expense' button is disabled based on cursor style; "
        "we don't disable buttons when there's an error on the form"
    )


@pytest.mark.asyncio
async def test_split_expense(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            slow_mo=300,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            proxy={"server": PROXY_URL} if PROXY_URL else None,
        )
        context = await browser.new_context(ignore_https_errors=True)
        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = await context.new_page()
            await install_online_guard_async(context, page)

            await login.sign_up(page, LOGIN_EMAIL)
            try:
                await login.complete_onboarding(page)
            except (AssertionError, TimeoutError):
                pass

            await reproduce_scenario(page)

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                await context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            await context.close()
            await browser.close()
