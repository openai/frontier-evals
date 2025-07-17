import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright


# --------------------------------------------------------------------------------------
# TESTS
# --------------------------------------------------------------------------------------
def test_anonymous_user(pytestconfig):
    """
    Verify that anonymous users cannot see the 2FA banner inside the security page.
    """
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

        # Start tracing (if enabled)
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(
                context, page
            )  # Install guard immediately after creating the page

            # Navigate to a public report link
            page.goto(
                "https://dev.new.expensify.com:8082/r/3295647304331553",
                timeout=60_000,
            )

            # Click on the security settings link
            page.get_by_text("https://dev.new.expensify.com:8082/settings/security").click()

            # Assert that the 2FA banner is not visible
            enable_2fa = page.get_by_text(
                "Enable two-factor authentication to keep your account safe."
            )
            expect(enable_2fa).not_to_be_visible()

        finally:
            # Stop tracing and clean up (if enabled)
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Always close the context and browser
            context.close()
            browser.close()
