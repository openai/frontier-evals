import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import random
import re
import time
from datetime import datetime, timezone
from email.header import decode_header

from playwright._impl._page import Page
from playwright.sync_api import TimeoutError, expect, sync_playwright

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


def launch_browser(trace_enabled: bool = False):
    """
    Launch a browser and return playwright, browser, context, page.
    If trace_enabled is True, tracing will start on the created context.
    """
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--ignore-certificate-errors",
        ],
    )
    # Explicitly create a context so we can manage tracing reliably
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    # Install online guard immediately after context and page creation
    install_online_guard_sync(context, page)

    return playwright, browser, context, page


def login_user_if_not_logged_in(page: Page, first_name="Asaad", last_name="I"):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(EMAIL_USERNAME)
    page.locator("button", has_text="Continue").click()
    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=5000
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
        expect(page.get_by_text("What do you want to do today?")).to_be_visible(timeout=5000)
    except (AssertionError, TimeoutError):
        pass
    else:
        page.get_by_label("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)


def verify_delete_message(page: Page, chat_name: str):
    """
    Verify that a message can be deleted
    """
    chat_button = page.get_by_label("Navigates to a chat")
    chat_button.get_by_text(chat_name, exact=True).first.click()
    message = f"Hello this is a test message {random.randint(0, 1000000)}"
    if page.get_by_label("Chat message", exact=True).filter(has_text=message).count() == 0:
        page.get_by_role("textbox").fill(message)
        page.get_by_role("textbox").press("Enter")
    chat_message = page.get_by_text("Hello this is a test message").nth(1)
    chat_message.click(button="right")
    page.get_by_label("Delete comment").click()
    page.get_by_role("button", name="Delete").click()
    expect(page.get_by_role("button", name="Delete")).to_have_count(0)
    expect(chat_message).to_have_count(0)


def test_delete_message(pytestconfig):
    """
    Test that a message can be deleted
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    playwright = browser = context = page = None  # predefine for finally scope
    try:
        playwright, browser, context, page = launch_browser(trace_enabled)
        login_user_if_not_logged_in(page, "Asaad", "I")
        verify_delete_message(page, chat_name="Asaad I (you)")
    finally:
        if trace_enabled and context is not None:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            # Stop tracing and export the trace
            context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Clean up the browser & playwright objects
        if context is not None:
            context.close()
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()
