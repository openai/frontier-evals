import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright.sync_api import expect, sync_playwright

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1+89@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    page.locator('input[data-testid="validateCode"]').fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


def timezone(page, context):
    page.get_by_label("My settings").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Asia/Karachi").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Automatically determine your").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("TimezoneInitialPage").get_by_label("Asia/Karachi").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("TimezoneSelectPage").get_by_label("Back").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Automatically determine your").click()
    page.wait_for_timeout(1000)
    page1 = context.new_page()
    install_online_guard_sync(context, page1)
    page1.goto("https://dev.new.expensify.com:8082/settings/profile/timezone/select")
    page.wait_for_timeout(2000)
    page1.get_by_label("Asia/Kamchatka").click()
    page.wait_for_timeout(2000)
    page1.get_by_label("Back").click()
    page.wait_for_timeout(2000)
    expect(page1.get_by_test_id("TimezoneInitialPage").get_by_label("Asia/Karachi")).to_be_visible()
    page1.get_by_label("Back").click()
    page.wait_for_timeout(2000)
    expect(page1.get_by_text("Asia/Karachi").first).to_be_visible()


def test_expensify_0000(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            proxy={"server": "http://localhost:8080"},
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            slow_mo=None,
            channel="chromium",
        )

        # Create the browser context
        first_user_context = browser.new_context(
            ignore_https_errors=True, timezone_id="Asia/Karachi"
        )

        # Start tracing if enabled
        if trace_enabled:
            first_user_context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = first_user_context.new_page()
            install_online_guard_sync(first_user_context, page)
            page.wait_for_timeout(2000)
            new_dot_login(page, "pehlajrajput1@gmail.com")
            page.wait_for_timeout(2000)
            timezone(page, first_user_context)
        finally:
            # Handle trace export and cleaning if tracing was enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                first_user_context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the browser regardless of tracing
            first_user_context.close()
            browser.close()
