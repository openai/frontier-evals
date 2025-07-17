import os
import re
import uuid

from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


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


def compose_box(page):
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(2000)
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(2000)
    page.get_by_test_id("selection-list-text-input").click()
    page.wait_for_timeout(2000)
    page.get_by_test_id("selection-list-text-input").fill("agata.kosior@callstack.com")
    page.wait_for_timeout(2000)
    page.get_by_test_id("selection-list").get_by_label("Agata Kosior").click()
    page.wait_for_timeout(2000)
    composer_locator = page.locator('[contenteditable="true"]').last
    composer_locator.type("Hi")
    page.wait_for_timeout(1000)
    expect(composer_locator.get_by_text("Hi", exact=True)).to_be_visible()


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
            slow_mo=1000,
            channel="chromium",
        )
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.goto(NEWDOT_URL, timeout=60000)
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL)
            page.wait_for_timeout(2000)
            compose_box(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()
            browser.close()
