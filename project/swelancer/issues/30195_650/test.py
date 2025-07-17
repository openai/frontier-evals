import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re
import time

from playwright.sync_api import expect, sync_playwright

EMAIL = "t1533148+123@gmail.com"


def login_user(p, trace_enabled: bool = False):
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        proxy={"server": "http://127.0.0.1:8080/"},
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context_args = {
        "geolocation": {"latitude": 37.7749, "longitude": 122.4194},
        "permissions": ["geolocation"],
        "ignore_https_errors": True,
    }
    context = browser.new_context(**context_args)

    # Start tracing immediately after context creation if enabled
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = EMAIL
    page.goto("https://127.0.0.1:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    page.wait_for_timeout(5000)
    otp = "123456"
    page.locator('input[name="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    return browser, page, context


def test_field_validation(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as pw:
        browser, page, context = login_user(pw, trace_enabled=trace_enabled)

        try:
            page.wait_for_timeout(2000)
            page.get_by_label("My settings").click()
            page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").get_by_text(
                "Workspaces"
            ).click()
            page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").click()
            page.locator(
                'button[aria-label="New workspace"][role="button"][data-tag="pressable"]'
            ).nth(0).click()
            page.get_by_label("More features").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Configure how spend is").click()
            page.get_by_label("Workflows").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Connect bank account").click()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Update to USD").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Connect manually").click()
            page.get_by_role("textbox", name="Routing number").fill("071004200")
            page.wait_for_timeout(1000)
            page.get_by_role("textbox", name="Account number").fill("871698019")
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Next").click()
            page.wait_for_timeout(1000)
            page.get_by_role("textbox", name="Legal first name").fill("d")
            page.wait_for_timeout(1000)
            page.get_by_role("textbox", name="Legal last name").fill("c")
            page.get_by_role("button", name="Next").click()
            page.wait_for_timeout(1000)
            page.get_by_role("form").get_by_label("1", exact=True).click()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Next").click()
            page.wait_for_timeout(1000)
            page.get_by_role("textbox", name="Last 4 of SSN").fill("0988")
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Next").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Street address").get_by_role("textbox").click()
            page.wait_for_timeout(1000)
            page.get_by_role("textbox", name="City").click()
            page.wait_for_timeout(1000)
            page.get_by_role("menuitem", name="State").get_by_role("img").click()
            page.wait_for_timeout(1000)
            page.get_by_test_id("StateSelectionPage").get_by_label("Back").click()
            page.wait_for_timeout(1000)
            expect(
                page.get_by_test_id("PersonalInfo")
                .get_by_role("menuitem")
                .get_by_text("This field is required.")
            ).to_be_visible()
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
