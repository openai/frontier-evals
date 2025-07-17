import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright

SITE_URL = "https://dev.new.expensify.com:8082/"
PRIMARY_EMAIL = "expensifyolddot+1733831382@gmail.com"
FIRST_NAME = "Expenolddot"
LAST_NAME = "1733831382"


def login_user(
    page: Page,
    first_name="Asaad",
    last_name="I",
    email: str = "testuser@gmail.com",
    platform="chromium",
):
    page.goto(SITE_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.locator("button", has_text="Continue").click()
    page.locator("button", has_text="Join").click()
    page.wait_for_timeout(2000)
    if page.locator("text='Track and budget expenses'").is_visible():
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("form").get_by_role("button", name="Continue").click()
    page.reload()
    if platform.lower() in ["ios", "android"]:
        page.get_by_label("Last chat message preview").filter(
            has_text="task for Track an expense"
        ).wait_for()
        page.get_by_label("Last chat message preview").filter(
            has_text="task for Track an expense"
        ).first.click()
    page.get_by_label("guided-setup-track-personal-").wait_for()
    if platform.lower() in ["ios", "android"]:
        page.get_by_label("Back").click()
        page.get_by_label("Inbox").wait_for()


def verify_go_back_works_after_reload_on_flag_page(page: Page):
    page.goto("https://dev.new.expensify.com:8082/r/5624984165978443", timeout=60000)
    page.wait_for_timeout(5000)
    page.get_by_label("Chat message", exact=True).first.click(button="right")
    page.get_by_label("Flag as offensive").click()
    page.get_by_text("Choose a reason for flagging").wait_for()
    page.get_by_test_id("FlagCommentPage").get_by_label("Back").click()
    expect(page.get_by_label("Chat message", exact=True).first).to_be_visible()
    page.get_by_label("Chat message", exact=True).first.click(button="right")
    page.get_by_label("Flag as offensive").click()
    page.get_by_text("Choose a reason for flagging").wait_for()
    page.reload()
    page.get_by_test_id("FlagCommentPage").get_by_label("Back").click()
    expect(page.get_by_label("Chat message", exact=True).first).to_be_visible()


def _start_tracing_if_enabled(context, enabled: bool):
    """Starts playwright tracing if enabled flag is True."""
    if enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_tracing_if_enabled(context, enabled: bool):
    """Stops tracing and cleans the trace if the enabled flag is True."""
    if enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_go_back_after_reload_on_flag_page(pytestconfig):
    """
    Test to verify that the go back works after reload on flag page.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None

    with sync_playwright() as p:
        try:
            platform = "ios"
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--ignore-certificate-errors",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ],
                proxy={"server": "http://localhost:8080"},
                slow_mo=500,
            )

            device_config = p.devices["iPhone 14"]
            context_args = {"timezone_id": "Asia/Kathmandu"}
            merged_context_args = {**device_config, **context_args}
            context = browser.new_context(**merged_context_args)

            # Start tracing if requested
            _start_tracing_if_enabled(context, trace_enabled)

            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed guard

            login_user(page, FIRST_NAME, LAST_NAME, PRIMARY_EMAIL, platform)
            verify_go_back_works_after_reload_on_flag_page(page)

        finally:
            if context:
                _stop_tracing_if_enabled(context, trace_enabled)
                context.close()
            if browser:
                browser.close()
