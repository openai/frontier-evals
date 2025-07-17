import os
import sys
import uuid
from datetime import datetime

from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

# Make repo root importable
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

TIMEOUT = 200000


def generate_future_date_string():
    future_date = datetime(datetime.now().year + 1, 1, 1)
    return future_date.strftime("%Y-%m-%d")


def login_user(
    p,
    first_name="Milan",
    last_name="T",
    user_email=None,
    trace_enabled: bool = False,
):
    """
    Logs in a user and returns the created browser, context, page and email.
    If `trace_enabled` is True a tracing session is started on the created context.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        slow_mo=1000,
    )

    context = browser.new_context(timezone_id="Asia/Dhaka")
    context.set_default_navigation_timeout(TIMEOUT)
    context.set_default_timeout(TIMEOUT)

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()

    try:
        something_else_button = page.locator('div[aria-label="Something else"]')
        something_else_button.click()
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
    except Exception:
        # If the user is already created and "Something else" isn't present
        pass

    return browser, context, page, user_email


def _stop_tracing_and_clean(context, trace_enabled: bool):
    """
    Helper to stop tracing on a context and clean the resulting file.
    """
    if not trace_enabled or context is None:
        return

    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_banned_user_banner_appears_in_chat_instead_of_textbox(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        # Initialise to None to ensure we can reference them in `finally`
        browser_user_one = None
        browser_user_two = None
        context_user_one = None
        context_user_two = None

        try:
            user_email_1 = "rgarciatammy4+1733074305@gmail.com"
            user_email_2 = "rgarciatammy4+1733074306@gmail.com"

            (
                browser_user_one,
                context_user_one,
                page_user_one,
                _,
            ) = login_user(
                p,
                first_name="User",
                last_name="One",
                user_email=user_email_1,
                trace_enabled=trace_enabled,
            )

            (
                browser_user_two,
                context_user_two,
                page_user_two,
                _,
            ) = login_user(
                p,
                first_name="User",
                last_name="Two",
                user_email=user_email_2,
                trace_enabled=trace_enabled,
            )

            # User One starts a chat with User Two
            start_chat_fab = page_user_one.get_by_role(
                "button", name="Start chat (Floating action)"
            )
            start_chat_fab.click()
            page_user_one.get_by_role("menuitem", name="Start chat").click()
            page_user_one.get_by_test_id("selection-list-text-input").fill(user_email_2)
            page_user_one.get_by_label("User Two").click()

            chat_textbox = page_user_one.get_by_role("textbox")
            text_content = (
                "This is a test message from an Expensify contributor! "
                "Please flag me as assault and block me."
            )
            chat_textbox.fill(text_content)
            chat_textbox.press("Enter")
            page_user_one.wait_for_timeout(2000)

            # User Two flags the message
            page_user_two.get_by_text("User Two").first.click()
            page_user_two.get_by_text("User One").first.click()
            page_user_two.get_by_text(text_content).hover()
            page_user_two.get_by_label("Menu").click()
            page_user_two.get_by_text("Flag as offensive").click()
            page_user_two.get_by_label("Assault").get_by_text(
                "Message removed from channel"
            ).click()

            # Simulate User One being banned from chat
            page_user_one.evaluate(
                """
                async (date) => {
                    await Onyx.set('nvp_private_blockedFromChat', date);
                }
                """,
                generate_future_date_string(),
            )
            page_user_one.wait_for_timeout(3000)

            # Assertion: the banned banner appears
            report_actions_view = page_user_one.get_by_test_id("report-actions-view-wrapper").last
            expect(report_actions_view).to_contain_text(
                "Note: You've been banned from chatting in this channel"
            )

        finally:
            # Stop tracing and clean traces
            _stop_tracing_and_clean(context_user_one, trace_enabled)
            _stop_tracing_and_clean(context_user_two, trace_enabled)

            # Close contexts
            if context_user_one:
                context_user_one.close()
            if context_user_two:
                context_user_two.close()

            # Close browsers
            if browser_user_one:
                browser_user_one.close()
            if browser_user_two:
                browser_user_two.close()
