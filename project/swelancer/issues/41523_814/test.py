import os
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync


def login_user_if_not_logged_in(page: Page, first_name="Test", last_name="User"):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    email = "giwbpkqv+1735214762@gmail.com"
    page.get_by_test_id("username").fill(email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()
    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
        page.get_by_text("Track and budget expenses").click()
        page.get_by_role("button", name="Continue").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()
    except Exception:
        # If we are already past this point (user already logged in), just continue
        pass


def test_expensify_41523(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            proxy={"server": "http://localhost:8080"},
            slow_mo=500,
        )
        context = browser.new_context(timezone_id="Asia/Karachi")

        # Start tracing if the option is enabled
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            test_email = "ztzvcilj+1735214758@gmail.com"

            # Log in (or finish sign-up) if required
            login_user_if_not_logged_in(page, "Test", "User")

            # Submit two manual expenses
            for _ in range(2):
                page.get_by_label("Start chat (Floating action)").click()
                page.get_by_label("Submit expense").first.click()
                page.get_by_label("Manual").click()
                page.get_by_placeholder("0").fill("10")
                page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
                page.get_by_test_id("selection-list-text-input").fill(test_email)
                page.get_by_test_id("selection-list").get_by_label(test_email).click()
                page.get_by_role("button", name=re.compile("Submit .+")).click()

            # Open the newly created report and assert header visibility
            page.locator("button").filter(has_text=f"{test_email} owes: PKR 20.00").click()
            if page.get_by_label(test_email).nth(1).is_visible():
                print("Test Passed")
            else:
                pytest.fail("Chat header is not immediately visible")

        finally:
            if trace_enabled:
                # Stop tracing and clean the trace file
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
