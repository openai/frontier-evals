import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import re
import time
from datetime import datetime, timezone
from email.header import decode_header

from playwright._impl._browser import Browser
from playwright._impl._browser_context import BrowserContext
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

EMAIL_USERNAME = "aliceoglualice@gmail.com"
EMAIL_PASSWORD = "cfpy ofxm lcmq aeta"


def get_magic_code(user_email, password, since=None, retries=5, delay=10):
    if since is None:
        since = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user_email, password)
    for _ in range(retries):
        imap.select("inbox")
        status, messages = imap.search(None, '(UNSEEN SUBJECT "Expensify magic sign-in code:")')
        if status == "OK":
            email_ids = messages[0].split()
            if email_ids:
                latest_email_id = email_ids[-1]
                status, msg_data = imap.fetch(latest_email_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding or "utf-8")
                        match = re.search(r"Expensify magic sign-in code: (\d+)", subject)
                        email_date = email.utils.parsedate_to_datetime(msg["Date"])
                        if match and email_date >= since:
                            code = match.group(1)
                            imap.logout()
                            return code
                print("Email doesn't match conditions. Retrying...")
            else:
                print("No unread emails found with the subject. Retrying...")
        else:
            print("Failed to retrieve emails. Retrying...")
        time.sleep(delay)
    imap.logout()
    print("Max retries reached. Email not found.")
    return None


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool = False,
    first_name: str = "A",
    last_name: str = "Khan",
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launch the browser, (optionally) start tracing, and complete the login flow.

    Returns
    -------
    tuple
        (browser, context, page, user_email)
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--ignore-certificate-errors",
        ],
    )
    # Explicitly create a context so we can start tracing before any pages act.
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Guard installed immediately after page creation
    user_email = EMAIL_USERNAME

    # Begin actual login steps
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.get_by_test_id("username").fill(user_email)
    page.get_by_role("button", name="Continue").click()

    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=30000
        )
    except (AssertionError, TimeoutError):
        page.get_by_test_id("SignInPage").get_by_role("button", name="Join").click()
    else:
        otp_code = "123456"
        otp_input = page.locator('input[autocomplete="one-time-code"]')
        otp_input.fill(otp_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass

    try:
        expect(page.get_by_text("What do you want to do today?")).to_be_visible(timeout=30000)
    except (AssertionError, TimeoutError):
        pass
    else:
        page.get_by_label("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()

    return browser, context, page, user_email


def _export_and_clean_trace(context: BrowserContext):
    """
    Helper to export and clean a Playwright trace for the supplied context.
    """
    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_mention_logic_in_workspace_chat(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Login and obtain browser, context, and page handles
            browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)

            # Begin the original test steps
            account_settings = page.locator('button[aria-label="My settings"]')
            expect(account_settings).to_be_visible()
            account_settings.click()
            page.wait_for_timeout(1000)

            workspaces = page.locator('div[dir="auto"]:has-text("Workspaces")').last
            expect(workspaces).to_be_visible()
            workspaces.click()
            page.wait_for_timeout(1000)

            new_ws = page.locator('button[aria-label="New workspace"]').first
            expect(new_ws).to_be_visible()
            new_ws.click()
            page.wait_for_timeout(1000)

            locator = page.locator('div:has-text("invited to")').last
            text_content = locator.text_content()
            match = re.search(r"Khan's Workspace (\d+)", text_content or "")
            workspace_name = "A Khan's Workspace"
            if match:
                number = match.group(1)
                workspace_name = f"{workspace_name} {number}"

            back_arrow_button = page.locator('button[aria-label="Back"]')
            expect(back_arrow_button).to_be_visible()
            back_arrow_button.click()

            inbox_menu = page.locator('button[aria-label="Inbox"]')
            expect(inbox_menu).to_be_visible()
            inbox_menu.click()
            page.wait_for_timeout(1000)

            page.get_by_role("textbox").fill("#")
            page.wait_for_timeout(2000)
            page.get_by_text("#admins").click()

            expect(page.get_by_role("textbox").get_by_text("#", exact=True)).not_to_be_visible()
        finally:
            # Handle tracing export & clean-up if enabled
            if trace_enabled and context is not None:
                try:
                    _export_and_clean_trace(context)
                except Exception as e:
                    # Do not mask original test errors; just log if needed
                    print(f"Failed to export/clean trace: {e}")

            # Always close the context and browser if they were created
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
