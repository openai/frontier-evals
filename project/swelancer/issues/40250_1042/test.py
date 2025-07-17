import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import sync_playwright


def test_offline_message_not_visible_when_online(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()

        if trace_enabled:
            # Start tracing for this context
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Install online guard

            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
            context.set_offline(False)
            page.wait_for_timeout(2000)

            offline_message = page.locator("text=You appear to be offline")
            assert offline_message.count() == 0, (
                "The offline message should not be visible when online"
            )

        finally:
            if trace_enabled:
                # Export and clean the trace
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()  # Close context before closing the browser
            browser.close()
