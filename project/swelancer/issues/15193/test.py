import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)
from playwright.sync_api import expect, sync_playwright

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
EMAIL_A = "expensify87+15193E@gmail.com"
EMAIL_B = "expensify87+12@gmail.com"


def new_dot_signup(page, email):
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.get_by_role("button", name="Join").nth(0).click()
    page.get_by_label("Track and budget expenses").click()
    page.get_by_role("textbox", name="First name").fill("Test")
    page.get_by_role("textbox", name="Last name").click()
    page.get_by_role("textbox", name="Last name").fill("Bud")
    page.get_by_role("button", name="Continue").click()


def test_issue_15193(pytestconfig):
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
            slow_mo=500,
            channel="chromium",
        )
        context = browser.new_context(ignore_https_errors=True)

        # Start tracing immediately after context creation if enabled
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard
            page.goto(NEWDOT_URL, timeout=60000)
            new_dot_signup(page, EMAIL_A)
            page.wait_for_timeout(2000)
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_text("Start chat").click()
            page.get_by_test_id("selection-list-text-input").fill(EMAIL_B)
            page.get_by_label(EMAIL_B).click()
            page.get_by_role("textbox").click()
            page.get_by_role("textbox").fill("# ```\n\ntest\n\n```")
            context.set_offline(True)
            page.get_by_role("button", name="Send").click()
            report_id = page.url.split("/r/")[-1]
            message = (
                page.get_by_test_id(f"report-screen-{report_id}")
                .get_by_test_id("pre")
                .get_by_text("test")
            )
            expect(message).to_have_css("font-weight", "400", timeout=2000)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()  # Ensuring context is closed before browser
            browser.close()
