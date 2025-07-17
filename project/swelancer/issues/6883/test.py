import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright.sync_api import expect, sync_playwright

user_email = "totherandomuser+005@gmail.com"
password = "umei djwi bfmp jntm"
EMAIL = "totherandomuser+005@gmail.com"
PASSWORD = "umei djwi bfmp jntm"
NEWDOT_URL = "https://dev.new.expensify.com:8082"


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    otp = "123456"
    page.locator('input[data-testid="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


def login(p, trace_enabled: bool = False):
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
        slow_mo=1000,
        channel="chromium",
    )
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)
    page.goto(NEWDOT_URL, timeout=60000)
    new_dot_login(page, user_email)
    return browser, page, context


def test_keyboard_shortcuts(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = page = context = None  # initialise for finally safety
        try:
            browser, page, context = login(p, trace_enabled=trace_enabled)

            page.wait_for_timeout(5000)
            page.get_by_label("Start chat (Floating action)").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Start chat", exact=True).click()
            page.wait_for_timeout(1000)
            page.get_by_test_id("selection-list-text-input").fill("testuseracc22@gmail.com")
            page.wait_for_timeout(2000)
            page.get_by_test_id("selection-list").get_by_label("testuseracc22@gmail.com").click()
            page.wait_for_timeout(3000)
            page.keyboard.press("ControlOrMeta+j")
            page.wait_for_timeout(1000)
            expect(
                page.get_by_text("Save time with these handy keyboard shortcuts:")
            ).to_be_visible()
        finally:
            if trace_enabled and context is not None:
                # Stop tracing and clean trace file
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close context and browser if they were created
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
