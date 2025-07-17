import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._browser_context import BrowserContext
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

onboarding = False


def generate_random_email() -> str:
    task_number = "19398"
    return f"jellystarfish99+{task_number}@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool = False,
    first_name: str = "Milan",
    last_name: str = "T",
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launches the browser, (optionally) starts tracing, and performs sign-in.

    Returns
    -------
    (Browser, BrowserContext, Page, str):
        The browser instance, its context, the page object, and the user email.
    """
    global onboarding

    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    context = browser.new_context()

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = generate_random_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)

    # Occasionally an additional modal/button needs to be clicked.
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    # Handle onboarding flow if it appears
    if page.locator("text=What do you want to do today?").is_visible():
        onboarding = True
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(1000)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(1000)

    return browser, context, page, user_email


def test_add_reaction_to_chat_message_with_tab_key(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = context = page = None  # for safety in finally block
        try:
            browser, context, page, user_email = login_user(p, trace_enabled)

            # Determine the correct chat label based on onboarding state
            if onboarding:
                personal_chat = page.get_by_text("Milan T (you)")
            else:
                personal_chat = page.get_by_text("Jellystarfish99+19398@gmail.com (you)")

            expect(personal_chat).to_be_visible()
            personal_chat.click()
            page.wait_for_timeout(1000)

            # Send a message
            page.get_by_role("textbox").fill("My Message")
            page.get_by_role("textbox").press("Enter")
            page.wait_for_timeout(1000)

            message = page.locator("span", has_text="My Message")
            expect(message).to_be_visible()

            # Reply in thread
            message.click(button="right")
            page.wait_for_timeout(1000)
            reply_thread = page.get_by_text("Reply in thread")
            expect(reply_thread).to_be_visible()
            reply_thread.click()
            page.wait_for_timeout(1000)
            page.get_by_role("textbox").fill("My Reply")
            page.get_by_role("textbox").press("Enter")
            page.wait_for_timeout(1000)

            # Return to main chat and open message context menu
            if onboarding:
                personal_chat = page.get_by_text("Milan T (you)").nth(0)
            else:
                personal_chat = page.get_by_text("Jellystarfish99+19398@gmail.com (you)").nth(0)

            expect(personal_chat).to_be_visible()
            personal_chat.click()
            page.wait_for_timeout(1000)

            message = page.locator(".css-textHasAncestor-1jxf684", has_text="My Message")
            message.last.click(button="right")
            page.wait_for_timeout(1000)

            # Capture the x-position before and after editing comment
            pos_before = page.get_by_text("1 Reply").last.bounding_box()["x"]
            edit_comment = page.get_by_text("Edit comment")
            expect(edit_comment).to_be_visible()
            edit_comment.click()
            page.wait_for_timeout(1000)
            pos_after = page.get_by_text("1 Reply").last.bounding_box()["x"]

            assert pos_before == pos_after

        finally:
            # Stop tracing (if enabled) and clean trace
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close browser if it was created
            if browser is not None:
                if context is not None:
                    context.close()
                browser.close()
