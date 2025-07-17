import os
import re
import sys
import uuid

import pytest
from utils.online_guard import install_online_guard_async
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import Page, async_playwright, expect

USER_NAME = "xecuycismfsga+42204"
USER_EMAIL = f"{USER_NAME}@gmail.com"
PASSWORD = "uwgb zzoc cofa kaaq"


def get_magic_code(user_email: str, password: str, current_timestamp: int, retries=5, delay=3):
    return "123456"


async def login(page: Page, user_email: str, password: str):
    now = 20241211
    await page.get_by_test_id("username").fill(user_email)
    await page.get_by_role("button", name="Continue").click()
    await page.wait_for_timeout(3000)
    magic_code = get_magic_code(user_email, password, now, retries=3, delay=3)
    await page.get_by_role("textbox").fill(magic_code)
    try:
        await page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass
    await page.get_by_label("Inbox").wait_for(state="visible")


async def close_button_if_present(page: Page):
    """
    Occasionally, there is a close button that prevents any clicks on the page as
    it covers most of the screen. This button cannot be seen visually.
    """
    close_button = page.locator('button[aria-label="Close"]')
    if await close_button.is_visible():
        await close_button.click()


async def leave_group_chat(page: Page):
    if await page.get_by_text(USER_NAME).first.is_visible():
        await page.get_by_text(USER_NAME).first.click()
        await page.locator(
            f'div[data-testid="DisplayNamesWithTooltip"]:has-text("{USER_NAME}")'
        ).last.click()
        await page.get_by_label("Leave").click()
        await page.get_by_label("Back").first.click()


@pytest.mark.asyncio
async def test_leave_group_chat(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    playwright = None
    browser = None
    context = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            slow_mo=500,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = await browser.new_context()
        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        page = await context.new_page()
        await install_online_guard_async(context, page)
        await page.goto("https://dev.new.expensify.com:8082/", timeout=60000)

        await login(page, USER_EMAIL, PASSWORD)
        await leave_group_chat(page)
        await close_button_if_present(page)

        # Start a new group chat and perform operations
        await page.get_by_label("Start chat (Floating action)").click()
        await page.get_by_label("Start chat", exact=True).click()
        await page.get_by_test_id("selection-list-text-input").fill(f"{USER_NAME}_0@gmail.com")
        await page.get_by_role("button", name="Add to group").click()
        await page.get_by_role("button", name="Next").click()
        await page.get_by_role("button", name="Start group").click()
        await page.get_by_role("textbox").fill("Hello World")
        await page.get_by_role("button", name="Send").click()

        # Remove member and leave group
        await page.locator(
            f'div[data-testid="DisplayNamesWithTooltip"]:has-text("{USER_NAME}_0@gmail.com")'
        ).last.click()
        await page.get_by_label("Members").click()
        await page.get_by_label(f"{USER_NAME}_0@gmail.com").last.click()
        await page.get_by_role("button", name="selected").click()
        await page.get_by_label("Remove members").click()
        await page.get_by_role("button", name="Remove").click()
        await page.get_by_test_id("ReportParticipantsPage").get_by_label("Back").click()
        await page.get_by_label("Leave").click()
        await expect(page.get_by_role("button", name="Leave")).to_be_visible()
        await page.get_by_role("button", name="Leave").click()

    finally:
        if trace_enabled and context is not None:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            try:
                await context.tracing.stop(path=trace_path)
            except Exception:
                # Ensure we don't fail the test because tracing.stop had issues
                pass
            else:
                trace_cleaner(trace_path)

        if context is not None:
            await context.close()
        if browser is not None:
            await browser.close()
        if playwright is not None:
            await playwright.stop()
