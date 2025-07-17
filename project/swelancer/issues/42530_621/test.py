import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import sync_playwright

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "randomturing@gmail.com"
PASSWORD = "ryfo zezf tmlq xybs"
static_email = "randomturing+40121573@gmail.com"


def new_dot_login_join(page, email, first_name, last_name):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()
    page.locator("text='Track and budget expenses'").wait_for()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("form").get_by_role("button", name="Continue").click()


def verify_edit_message_with_only_emoji(page, chat_name):
    """
    Verify that editing a message containing only an emoji does not cut off
    the emoji in the composer.
    """
    page.get_by_label("Navigates to a chat").get_by_text(chat_name, exact=True).click()
    page.get_by_test_id("report-actions-list").get_by_text("Your space").wait_for()

    emoji_message = page.get_by_test_id("comment").get_by_text("😄")
    if emoji_message.count() == 0:
        page.get_by_role("textbox").fill("😄")
        page.get_by_role("textbox").press("Enter")

    emoji_message.first.click(button="right")
    page.get_by_label("Edit comment").click()

    emoji_box = page.get_by_test_id("report-actions-list").get_by_text("😄").bounding_box()
    input_box = page.locator("#messageEditInput").bounding_box()
    page.get_by_label("Save changes").click()

    assert (
        emoji_box["width"] <= input_box["width"] and emoji_box["height"] <= input_box["height"]
    ), "The emoji is cut off"


def test_expensify(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            slow_mo=1000,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(ignore_https_errors=True)

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard

            email = static_email
            new_dot_login_join(page, email, first_name="Tester", last_name="U")
            chat_name = "Tester U (you)"
            verify_edit_message_with_only_emoji(page, chat_name)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                # Stop tracing and save to the predefined location
                context.tracing.stop(path=trace_path)
                # Clean the trace
                trace_cleaner(trace_path)

            # Ensure resources are cleaned up
            context.close()
            browser.close()
