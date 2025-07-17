import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

static_email = "testuserexpensify1+45@gmail.com"


def login_user(
    p: PlaywrightContextManager, first_name="Milan", last_name="T"
) -> tuple[Browser, Page, str]:
    """
    Helper to launch a browser, return a page logged-in as `static_email`.
    """
    browser = p.chromium.launch(
        headless=True,
        slow_mo=1000,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    # Explicitly create a new context for this page.
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = static_email
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()

    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        # Swallow exceptions – original script tolerated this case.
        pass

    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()

    return browser, page, user_email


def test_delay_submission_toggle(pytestconfig):
    """
    Ported test wrapped in try/finally with optional Playwright tracing.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser: Browser | None = None
        context = None  # Will be set once the page is created
        try:
            # --- Test setup / execution -------------------------------------------------
            browser, page, user_email = login_user(p)
            context = page.context  # fetch the implicitly created context

            # Start tracing if requested
            if trace_enabled and context:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # --- Original test steps ----------------------------------------------------
            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            plus_icon.click()

            new_workspace_button = page.locator('div[aria-label="New workspace"]')
            new_workspace_button.click()

            page_feature = page.locator('div[aria-label="More features"]')
            page_feature.click()

            workflows = page.locator(
                'button[aria-label="Configure how spend is approved and paid."]'
            )
            workflows.click()

            workflow_setting = page.locator('div[aria-label="Workflows"]')
            workflow_setting.click()

            delay_submission = page.locator(
                'button[aria-label="Delay expense submissions based on a custom schedule, or keep this option disabled to maintain realtime spend visibility."]'
            )
            delay_submission.click()

            submission_frequency = page.get_by_text("Weekly")
            submission_frequency.click()

            change_manual_freq = page.locator('button[aria-label="Manually"]')
            change_manual_freq.click()

            is_delay_toggle_off = page.locator(
                'button[aria-label="Delay expense submissions based on a custom schedule, or keep this option disabled to maintain realtime spend visibility."]'
            )
            aria_checked_value = is_delay_toggle_off.get_attribute("aria-checked")
            if aria_checked_value == "false":
                assert False, (
                    "Delay submissions turns off if the Submission frequency is set to Manually"
                )
            else:
                assert True, (
                    "Delay submissions does not turn off if the Submission frequency is set to Manually"
                )

            page.wait_for_timeout(2000)

        finally:
            # --- Trace teardown ---------------------------------------------------------
            if trace_enabled and context:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                try:
                    context.tracing.stop(path=trace_path)
                except Exception:
                    # Ensure that tracing issues do not mask test failures.
                    pass
                trace_cleaner(trace_path)

            # --- Browser & Context teardown --------------------------------------------
            if context:
                context.close()
            if browser:
                browser.close()
