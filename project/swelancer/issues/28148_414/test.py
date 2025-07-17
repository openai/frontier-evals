import os
import sys
import uuid

import pytest
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright._impl._browser_context import BrowserContext
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Ayush",
    last_name: str = "G",
    *,
    trace_enabled: bool = False,
) -> tuple[Page, BrowserContext]:
    """
    Signs a user into newdot and returns the page & context.

    If trace_enabled is True, tracing will be started on the newly-created context.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=500,
    )
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Installed the online guard immediately after page creation
    user_email = "zfhmkgsf+1733658447@gmail.com"

    # Begin login / onboarding flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.get_by_test_id("username").fill(user_email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()

    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
        page.get_by_text("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()
    except Exception:
        # The onboarding flow might have been completed already; ignore failures here.
        pass

    return page, context


def test_save_description_in_submitted_expense(pytestconfig: pytest.Config):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        context: BrowserContext | None = None
        try:
            # Login / create a context (with optional tracing)
            page, context = login_user(p, trace_enabled=trace_enabled)

            # Go offline and perform settings update
            context.set_offline(True)
            page.get_by_label("My settings").click()
            page.locator("div").filter(has_text=re.compile(r"^Ayush G$")).nth(2).click()
            page.get_by_role("textbox", name="First name").fill("Ayush1")
            save_button = page.locator('button div:text("Save")')
            save_button.click()

            change_display_name_text = page.locator(
                'div:text("Your display name is shown on your profile.")'
            )
            expect(change_display_name_text).to_be_hidden()

        finally:
            if context is not None:
                # Stop tracing and clean trace if required
                if trace_enabled:
                    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                    context.tracing.stop(path=trace_path)
                    trace_cleaner(trace_path)

                context.close()
