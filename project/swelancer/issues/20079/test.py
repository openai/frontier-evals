import os
import re
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import sync_playwright

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1+345@gmail.com"
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


def composer_box(page):
    page.get_by_label("Start chat (Floating action)").click()
    page.get_by_label("Start chat", exact=True).click()
    page.get_by_test_id("selection-list-text-input").click()
    page.get_by_test_id("selection-list-text-input").fill("tosomenewuser@gmail.com")
    page.get_by_label("tosomenewuser@gmail.com").click()
    page.wait_for_timeout(3000)
    composer_locator = page.locator('[contenteditable="true"]').last
    current_url = page.url
    match = re.search("/r/(\\d+)", current_url)
    if match:
        last_id = match.group(1)
        print(f"Extracted last ID: {last_id}")
        test_id = f"report-screen-{last_id}"
        page.get_by_test_id(test_id).get_by_text("Hi", exact=True).click()
    else:
        print("No matching ID found in the URL.")
    page.get_by_label("Menu").click()
    page.get_by_label("Copy to clipboard").click()
    page.wait_for_timeout(2000)
    assert composer_locator.evaluate("node => document.activeElement === node"), (
        "Composer is not focused after 'Copy to clipboard'"
    )
    page.get_by_test_id(test_id).get_by_text("Hi", exact=True).click()
    page.get_by_label("Menu").click()
    page.get_by_label("Copy link").first.click()
    page.wait_for_timeout(2000)
    assert composer_locator.evaluate("node => document.activeElement === node"), (
        "Composer is not focused after 'Copy link'"
    )
    composer_locator.click()
    page.wait_for_timeout(2000)
    page.get_by_test_id(test_id).get_by_text("Hi", exact=True).hover()
    page.get_by_label("Mark as unread").click()
    page.wait_for_timeout(3000)
    assert composer_locator.evaluate("node => document.activeElement === node"), (
        "Composer is not focused after 'Mark as unread'"
    )


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
        first_user_context = browser.new_context(ignore_https_errors=True)
        if trace_enabled:
            first_user_context.tracing.start(screenshots=True, snapshots=True, sources=True)
        try:
            page = first_user_context.new_page()
            install_online_guard_sync(first_user_context, page)  # Install online guard
            page.wait_for_timeout(2000)
            new_dot_login(page, "pehlajrajput1+89@gmail.com")
            page.wait_for_timeout(2000)
            composer_box(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                first_user_context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            first_user_context.close()
            browser.close()
