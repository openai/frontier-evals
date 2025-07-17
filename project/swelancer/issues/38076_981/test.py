import os
import re
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import logging
import time

from playwright.sync_api import expect, sync_playwright

EMAIL_USERNAME = "ibukunsamuel999"
EMAIL_PASSWORD = "jkgz unyn rnwl slrp"
EXPENSIFY_URL = "https://dev.new.expensify.com:8082/"

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def clear_inbox(username, password):
    """
    Delete all the messages from the Inbox.
    """
    LOGGER.info("Deleting all the messages from the email inbox")
    with imaplib.IMAP4_SSL(host="imap.gmail.com") as imap:
        imap.login(username, password)
        imap.select("inbox")
        imap.store("1:*", "+FLAGS", "\\Deleted")
        imap.expunge()
        imap.close()


def get_otp_from_email(username, password, retries=12, delay=5):
    """
    Read the OTP email and return the OTP code.
    """
    LOGGER.info("Checking the OTP email")
    with imaplib.IMAP4_SSL(host="imap.gmail.com") as imap:
        imap.login(username, password)
        for _ in range(1, retries + 1):
            imap.select("inbox")
            status, messages = imap.search(None, "ALL")
            if status == "OK":
                for message_id in reversed(messages[0].split()):
                    status, data = imap.fetch(message_id, "(RFC822)")
                    if status == "OK":
                        email_message = email.message_from_bytes(data[0][1])
                        subject, encoding = email.header.decode_header(email_message["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding)
                        if subject.startswith("Expensify magic sign-in code:"):
                            otp_code = subject.split(":")[-1].strip()
                            LOGGER.info("Got the OTP %s", otp_code)
                            return otp_code
            time.sleep(delay)
        imap.close()
    raise AssertionError("Failed to read the OTP from the email")


def login_user(page, email_addr, first_name="John", last_name="Doe"):
    """
    Log into the Expensify app.
    """
    page.goto(EXPENSIFY_URL, timeout=60000)
    page.get_by_test_id("username").fill(email_addr)
    page.get_by_role("button", name="Continue").click()
    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=7000
        )
    except (AssertionError, TimeoutError):
        page.get_by_test_id("SignInPage").get_by_role("button", name="Join").click()
    else:
        otp_code = "123456"
        page.get_by_test_id("SignInPage").get_by_test_id("validateCode").fill(otp_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
        page.get_by_test_id("SignInPage").get_by_role("button", name="Sign in").click()
    try:
        expect(page.get_by_text("What do you want to do today?")).to_be_visible(timeout=10000)
    except (AssertionError, TimeoutError):
        pass
    else:
        page.get_by_label("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()
        try:
            page.get_by_role("button", name="Back").first.click(timeout=3000)
        except (AssertionError, TimeoutError):
            pass
    expect(page.get_by_test_id("BaseSidebarScreen")).to_be_visible(timeout=10000)


def generate_random_email():
    timestamp = int(time.time())
    return f"{EMAIL_USERNAME}+{timestamp}@gmail.com"


def test_check_cursor_position(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
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
            install_online_guard_sync(
                context, page
            )  # Install guard immediately after page creation
            user_email = "ibukunsamuel999+1733126918@gmail.com"
            login_user(page, user_email, first_name="Milan", last_name="T")

            page.get_by_role("textbox").fill("Hello @applausetester+mn03@applause.expensifail.com")
            time.sleep(1)
            page.get_by_role("paragraph").get_by_text("Hello").click()
            time.sleep(1)

            start_position = page.evaluate(
                "() => {"
                "const selection = window.getSelection();"
                "if (selection.rangeCount > 0) {"
                "   const range = selection.getRangeAt(0);"
                "   return range.startOffset;"
                "}"
                "return null;"
                "}"
            )

            for _ in range(5):
                page.get_by_role("textbox").press("ArrowRight")

            at_position = page.evaluate(
                "() => {"
                "const selection = window.getSelection();"
                "if (selection.rangeCount > 0) {"
                "   const range = selection.getRangeAt(0);"
                "   return range.startOffset;"
                "}"
                "return null;"
                "}"
            )

            for _ in range(5):
                page.get_by_role("textbox").press("ArrowLeft")

            final_position = page.evaluate(
                "() => {"
                "const selection = window.getSelection();"
                "if (selection.rangeCount > 0) {"
                "   const range = selection.getRangeAt(0);"
                "   return range.startOffset;"
                "}"
                "return null;"
                "}"
            )

            assert final_position == start_position, (
                f"Cursor did not move after the @: start {start_position}, @ position {at_position}, final {final_position}"
            )

            time.sleep(2)

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()  # Ensure context is properly closed
            browser.close()
