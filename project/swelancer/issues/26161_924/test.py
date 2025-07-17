import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright


def test_console_error(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()

        # Start tracing if the flag is enabled
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Added guard installation
            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)

            page1 = context.new_page()
            install_online_guard_sync(
                context, page1
            )  # Added guard installation for the second page
            page1.goto("https://dev.new.expensify.com:8082/")

            phone_or_email_input = page1.locator('input[type="email"]')
            expect(phone_or_email_input).to_be_visible()
            phone_or_email_input.fill("jellystarfish99@gmail.com")
            page1.get_by_role("button", name="Continue").click()
            page1.wait_for_timeout(1000)

            expect(page.get_by_text("You've opened the login page")).to_be_visible()
            expect(page.get_by_text("Another login page is open.")).to_be_visible()

            page1.wait_for_timeout(1000)

        finally:
            # Stop tracing and clean up the trace if tracing was enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close context and browser
            context.close()
            browser.close()
