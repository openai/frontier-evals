import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright


def launch_browser(playwright, platform="desktop"):
    mobile_context = {}
    if platform.lower() == "mobile":
        mobile_context = {"viewport": {"width": 540, "height": 864}, "is_mobile": True}
    browser = playwright.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=500,
    )
    context = browser.new_context(timezone_id="Asia/Karachi", **mobile_context)
    page = context.new_page()
    return context, page


def login_user(page: Page, first_name="Asaad", last_name="I"):
    page.bring_to_front()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    email = "lbrxifzk+1735219804@gmail.com"
    page.get_by_test_id("username").fill(email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()
    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
        page.get_by_text("Track and budget expenses").click()
        page.get_by_role("button", name="Continue").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()
    except Exception:
        pass


def handle_close_btn(page: Page):
    try:
        close_btn = page.get_by_label("Close")
        expect(close_btn).to_be_visible()
        close_btn.click()
    except Exception:
        pass


def _start_tracing_if_enabled(context, enabled: bool):
    if enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_tracing_and_clean(context, enabled: bool):
    if enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_qbo_connection_alignment_in_accounting_settings_narrow_layout(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as playwright:
        context, page = launch_browser(playwright, "mobile")
        install_online_guard_sync(context, page)  # Added
        _start_tracing_if_enabled(context, trace_enabled)

        try:
            login_user(page, "Asaad", "I")
            handle_close_btn(page)

            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_label("New workspace").click()
            page.get_by_label("More features").click()
            page.get_by_label("Sync your chart of accounts").click()
            page.get_by_label("Accounting").click()

            qbo_text_box = page.get_by_text("Quickbooks Online", exact=True).bounding_box()
            qbo_text_box_center_y = qbo_text_box["y"] + qbo_text_box["height"] / 2

            connect_button_box = (
                page.get_by_label("Quickbooks Online")
                .get_by_role("button", name="Connect")
                .bounding_box()
            )
            connect_button_center_y = connect_button_box["y"] + connect_button_box["height"] / 2

            tolerance = 1
            assert abs(qbo_text_box_center_y - connect_button_center_y) < tolerance, (
                "Text and button are not vertically center-aligned."
            )
        finally:
            _stop_tracing_and_clean(context, trace_enabled)
            context.close()
