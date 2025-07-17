import os
import re
import sys
import uuid

import pytest
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import logging
import time

from playwright.sync_api import Page, expect, sync_playwright

EXPENSIFY_URL = "https://dev.new.expensify.com:8082/"

EMAIL_USERNAME = "turingsampleopen@gmail.com"
EMAIL_PASSWORD = "dmyi iqgn znbg kqii"

# Logger
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
    try:
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
                            subject, encoding = email.header.decode_header(
                                email_message["Subject"]
                            )[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding)
                            if subject.startswith("Expensify magic sign-in code:"):
                                otp_code = subject.split(":")[-1].strip()
                                LOGGER.info("Got the OTP %s", otp_code)
                                return otp_code
                time.sleep(delay)
            imap.close()
        raise AssertionError("Failed to read the OTP from the email")
    except (AssertionError, TimeoutError):
        otp_code = "123456"
        return otp_code


def login_user(page: Page, email=EMAIL_USERNAME, first_name="John", last_name="Doe"):
    """
    Log into the Expensify app.
    """
    # Open the Expensify app
    page.goto(EXPENSIFY_URL)
    # Login user
    page.get_by_test_id("username").fill(email)
    page.get_by_role("button", name="Continue").click()
    # Check if OTP is required for the login
    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=7000
        )
    except (AssertionError, TimeoutError):
        # If not required, expect the join button to appear and click the button
        page.get_by_test_id("SignInPage").get_by_role("button", name="Join").click()
    else:
        # Get the OTP and complete verification
        otp_code = "123456"  # or call get_otp_from_email(...)
        page.get_by_test_id("SignInPage").get_by_test_id("validateCode").fill(otp_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass

    if page.get_by_text("Track and budget expenses").is_visible():
        # Complete the onboarding
        page.get_by_label("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()
        # Exit the Concierge chat, if opened by default
        try:
            page.get_by_role("button", name="Back").first.click(timeout=15000)
        except (AssertionError, TimeoutError):
            pass


@pytest.mark.parametrize("viewport", [(1280, 720)])
def test_export_onyx_state(pytestconfig, viewport):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    width, height = viewport

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features-IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(viewport={"width": width, "height": height})

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)

            # Test steps begin
            login_user(page)

            page.locator('button[aria-label="My settings"]').click()
            page.locator('div[aria-label="Troubleshoot"]').click()

            troubleshoot_div = page.locator('div[data-testid="TroubleshootPage"]')
            export_onyx_state_div = troubleshoot_div.locator(
                'div:text("Export Onyx State")', has_text="Export Onyx State"
            )

            assert export_onyx_state_div.is_visible(), "Export Onyx State div is not visible"

            with page.expect_download() as download_info:
                export_onyx_state_div.click()

            download = download_info.value
            download_path = download.path()

            assert download_path is not None, "No file was downloaded"

            file_size = os.path.getsize(download_path)

            assert file_size != 0, "Empty file download for authentication codes!"
            # Test steps end

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
