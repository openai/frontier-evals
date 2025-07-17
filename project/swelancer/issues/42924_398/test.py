import os
import re
import sys
import uuid

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import pytest
from playwright.async_api import Page, async_playwright, expect
from utils.browser import close_browser
from utils.login import NEW_DOT_URL, logging, sign_in
from utils.online_guard import install_online_guard_async
from utils.trace_cleaner import trace_cleaner

STATIC_PREFIX = 42924
EXPENSIFY_USERNAME_FIRSTNAME = "osaidtest"
EXPENSIFY_USERNAME_LASTNAME = "testaccount"
EXPENSIFY_USERNAME = f"{EXPENSIFY_USERNAME_FIRSTNAME} {EXPENSIFY_USERNAME_LASTNAME}"
EXPENSIFY_EMAIL_USERNAME = "osaidtest5"
EXPENSIFY_EMAIL = f"osaidtest5+{STATIC_PREFIX}@gmail.com"
EXPENSIFY_PASSWORD = "ntvd acej mmaa qqmd"
DATA_DIR = "browser_context"
WORKSPACE_NAME = "workspace_xero"
SLOW_MO = 1000
VIEW_PORT = {"width": 375, "height": 812}
TFA_CODE = "VGap9xc0"
NORMAL_RUN = False


async def start_browser(
    headless=True,
    persistent=False,
    data_dir=None,
    slow_mo=500,
    launch_args=None,
):
    """
    Start a browser instance with the given parameters.

    :param headless: Boolean to specify if the browser should run in headless mode.
    :param persistent: Boolean to specify if the browser context should be persistent.
    :param data_dir: Directory to store browser data for persistent context.
    :return: A tuple of (context, page, playwright).
    """
    if launch_args is None:
        launch_args = ["--ignore-certificate-errors"]
    playwright = await async_playwright().start()
    context, page = (None, None)
    if persistent:
        if data_dir is None:
            data_dir = "browser_context"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        context = await playwright.chromium.launch_persistent_context(
            data_dir,
            headless=headless,
            args=launch_args,
            slow_mo=slow_mo,
            timezone_id="Asia/Karachi",
            viewport=VIEW_PORT,
        )
        page = context.pages[0]
    else:
        browser = await playwright.chromium.launch(
            headless=headless, args=launch_args, slow_mo=slow_mo
        )
        context = await browser.new_context(
            ignore_https_errors=True, timezone_id="Asia/Karachi", viewport=VIEW_PORT
        )
        page = await context.new_page()
    return (context, page, playwright)


async def sign_in_recorded(page: Page, email: str):
    await page.get_by_test_id("username").fill(email)
    await page.get_by_role("button", name="Continue").click()
    await page.get_by_test_id("validateCode").fill("123456")
    try:
        await page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


async def enter_2fa_code(page: Page, code: str):
    await page.get_by_role("button", name="Use recovery code").click()
    await page.get_by_role("textbox").fill(code)
    await page.get_by_role("Button", name="Sign in").click()
    await page.get_by_text("Please enter your recovery code").wait_for(state="hidden")
    logging.info("2fa code entered successfully.")


@pytest.mark.asyncio
async def test_42924(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    context = page = playwright = None  # Predefine for use in finally if creation fails

    try:
        context, page, playwright = await start_browser(
            persistent=False,
            data_dir=DATA_DIR,
            headless=True,
            slow_mo=1000,
            launch_args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        await install_online_guard_async(context, page)

        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        if NORMAL_RUN:
            await sign_in(page, EXPENSIFY_EMAIL, password=EXPENSIFY_PASSWORD)
            await enter_2fa_code(page, TFA_CODE)
        else:
            await page.goto(NEW_DOT_URL, timeout=60000)
            await sign_in_recorded(page=page, email=EXPENSIFY_EMAIL)
            await enter_2fa_code(page, "lXgvKcUU")

        await page.get_by_role("button", name="My settings").click()
        await page.get_by_role("menuitem", name="Workspaces").click()
        await (
            page.get_by_test_id("WorkspacesListPage")
            .get_by_role("button", name="row")
            .filter(has=page.get_by_text("workspace_xero"))
            .click()
        )
        await (
            page.get_by_test_id("WorkspaceInitialPage")
            .get_by_role("menuitem", name="Accounting")
            .click()
        )
        await (
            page.get_by_test_id("PolicyAccountingPage").get_by_role("menuitem", name="test").click()
        )
        await page.get_by_test_id("selection-list").get_by_role("button", name="anotherOrg").click()
        await page.get_by_test_id("PolicyAccountingPage").get_by_role("button", name="Back").click()
        await page.get_by_test_id("WorkspaceInitialPage").get_by_role("button", name="Back").click()
        await expect(
            page.get_by_test_id("WorkspacesListPage").get_by_text("Workspaces")
        ).to_be_visible()

    finally:
        # Handle tracing and clean-up
        if context and trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            try:
                await context.tracing.stop(path=trace_path)
            except Exception:
                # If tracing was not started or already stopped, ignore silently
                pass
            else:
                trace_cleaner(trace_path)

        if context or playwright:
            await close_browser(context, page, playwright)
