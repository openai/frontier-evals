import os
import re
import sys
import uuid

from playwright.sync_api import expect, sync_playwright
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from utils.online_guard import install_online_guard_sync

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
        page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass


def formatting(page):
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(2000)
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(2000)
    page.get_by_test_id("selection-list-text-input").click()
    page.wait_for_timeout(2000)
    page.get_by_test_id("selection-list-text-input").fill("pehlajrajput1+43@gmail.com")
    page.get_by_test_id("selection-list").get_by_label("pehlajrajput1+43@gmail.com").click()
    page.wait_for_timeout(2000)
    page.get_by_text("<html> <body> <h1>HTML").click(button="right")
    page.wait_for_timeout(1000)
    page.get_by_label("Copy to clipboard").click()
    page.wait_for_timeout(2000)
    page.get_by_test_id("report-screen-8198893088087890").locator("#composer")
    page.wait_for_timeout(2000)
    page.keyboard.press("ControlOrMeta+v")
    page.wait_for_timeout(2000)
    expect(page.get_by_text("<html><body> <h1>HTML")).to_be_visible()


def test_expensify_0000(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=False,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(ignore_https_errors=True)
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL)
            page.wait_for_timeout(2000)
            formatting(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                # Ensure the directory exists
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()
            browser.close()
