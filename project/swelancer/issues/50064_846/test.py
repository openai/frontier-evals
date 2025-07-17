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
import random
import time

from playwright.sync_api import expect, sync_playwright

# Constants for URLs and credentials
NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "ikhanle001+34@gmail.com"
PASSWORD = "dzcr knad otxp bsuz"


class IMAPOTPExtractor:
    def __init__(self, email_address, password):
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password

        random_suffix = random.randint(1000, 9999)
        current_timestamp = int(time.time())
        random_suffix = f"{current_timestamp}{random_suffix}"

    def _validate_email(self, email_address):
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email_address))

    def set_new_email(self, email):
        self.new_email = email

    def check_for_otp(self):
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.original_email, self.password)
            mail.select("inbox")

            status, messages = mail.search(None, "UNSEEN")
            if status != "OK":
                return None

            email_ids = messages[0].split()
            email_ids = email_ids[::-1]

            for email_id in email_ids:
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status != "OK":
                    continue

                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])

                        if self.new_email in msg["To"]:
                            otp_code = self._extract_otp_from_email(msg)
                            if otp_code:
                                mail.store(email_id, "+FLAGS", "\\Deleted")
                                mail.expunge()
                                mail.close()
                                mail.logout()
                                return otp_code

            mail.close()
            mail.logout()
            return None

        except imaplib.IMAP4.error:
            return None

    def _extract_otp_from_email(self, msg):
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_body = part.get_payload(decode=True).decode()
                    otp_code = self._extract_otp(email_body)
                    if otp_code:
                        return otp_code
        else:
            email_body = msg.get_payload(decode=True).decode()
            otp_code = self._extract_otp(email_body)
            if otp_code:
                return otp_code
        return None

    def _extract_otp(self, text):
        match = re.search(r"\b\d{6}\b", text)
        return match.group(0) if match else None


@pytest.mark.parametrize("headless", [True])
def test_50064(pytestconfig, headless):
    """
    Test that verifies the onboarding flow when clicking
    'Track and budget expenses' after logging in.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        # Create a fresh context
        context = browser.new_context(ignore_https_errors=True)

        # If tracing is enabled, start the trace recording
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            # Page 1 – perform login
            page1 = context.new_page()
            install_online_guard_sync(context, page1)
            page1.goto(NEWDOT_URL)
            page1.get_by_test_id("username").fill(EMAIL)
            page1.get_by_test_id("username").press("Enter")

            # Use static OTP for now (replace with extractor if needed)
            otp = "123456"
            page1.locator('input[data-testid="validateCode"]').fill(otp)
            try:
                page1.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                    timeout=20000
                )
            except Exception:
                pass
            page1.wait_for_timeout(4000)

            # Page 2 – verify onboarding redirect
            page2 = context.new_page()
            install_online_guard_sync(context, page2)
            page2.goto(NEWDOT_URL)
            page1.wait_for_timeout(4000)
            page2.get_by_label("Track and budget expenses").click()

            expect(page2).to_have_url(re.compile(r".*/onboarding/personal-details.*"))

        finally:
            # Stop and export trace if it was enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Always close the browser
            context.close()
            browser.close()
