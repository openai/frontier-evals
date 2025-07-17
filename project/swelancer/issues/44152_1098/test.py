import os
import re
import sys
import uuid

import pytest
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import sync_playwright

EMAIL = "t1533148@gmail.com"
PASSWORD = "logb hzkg pkfk kskg"


def signup_user(page, user_email, first_name, last_name) -> None:
    page.goto("https://127.0.0.1:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)
    otp = "123456"
    page.locator('input[name="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    page.wait_for_timeout(2000)


def run_test(page: Page) -> None:
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(1000)
    page.get_by_text("Start chat").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").fill("a@a.com")
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Add to group").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Next").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Start group").click()
    page.wait_for_timeout(1000)
    page.get_by_role("textbox").fill("Hello message 1")
    page.wait_for_timeout(3000)
    page.get_by_label("Send").click()
    page.wait_for_timeout(3000)
    page.get_by_role("textbox").fill("Hello message 2")
    page.wait_for_timeout(1000)
    page.get_by_label("Send").click()
    page.wait_for_timeout(2000)
    page.locator('button[aria-label*="Ak, Hidden"][role="button"]').nth(-1).click()
    page.wait_for_timeout(3000)
    page.get_by_label("Members").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Hidden").nth(-1).click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Remove from group").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Remove", exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("ReportParticipantsPage").get_by_label("Back").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Leave").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Leave").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Back", exact=True).click()
    page.wait_for_timeout(3000)
    is_main_page_visible = bool(page.get_by_label("Workspaces").is_visible())
    is_error_visible = page.locator("text=Hmm... it's not hereYou don't").is_visible()
    if is_error_visible:
        raise AssertionError("Error message was visible, test failed.")
    elif not is_main_page_visible:
        raise AssertionError("Screen is loading for too long.")
    else:
        print("Test passed: Error message was not visible.")


@pytest.mark.parametrize("first_name, last_name", [("AK", "Test")])
def test_expensify(first_name: str, last_name: str, pytestconfig) -> None:
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser: Browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            slow_mo=1000,
            proxy={"server": "http://127.0.0.1:8080/"},
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        # Browser context setup
        context_args = {
            "geolocation": {"latitude": 37.7749, "longitude": 122.4194},
            "permissions": ["geolocation"],
            "viewport": {"width": 375, "height": 550},
            "ignore_https_errors": True,
        }
        context = browser.new_context(**context_args)

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(
                context, page
            )  # Install the online guard immediately after page creation
            signup_user(page, EMAIL, first_name, last_name)
            run_test(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            browser.close()
