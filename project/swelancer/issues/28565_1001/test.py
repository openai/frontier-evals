import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import random
import re
import time

from playwright.sync_api import sync_playwright
from utils.online_guard import install_online_guard_sync

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "turingvidhant+109@gmail.com"
PASSWORD = "uotx fzkt kfeg dehz"


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
        return bool(re.match("[^@]+@[^@]+\\.[^@]+", email_address))

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
            email_ids = messages[0].split()[::-1]
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
        match = re.search("\\b\\d{6}\\b", text)
        return match.group(0) if match else None


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(3000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(3000)
    page.locator('input[data-testid="validateCode"]').fill("123456")
    try:
        page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass
    try:
        page.get_by_role("button", name="Sign In").click()
    except Exception:
        # The Sign In button may not always be present (already logged in etc.)
        pass


def launch_browser(pw, headless=True, device=None, geolocation=None):
    """
    Launch the browser.
    """
    browser = pw.chromium.launch(headless=headless)
    context_args = {}
    if device:
        context_args.update(pw.devices[device])
    if geolocation:
        context_args["geolocation"] = geolocation
        context_args["permissions"] = ["geolocation"]
    context = browser.new_context(**context_args)
    page = context.new_page()
    install_online_guard_sync(context, page)
    return (browser, context, page)


def _start_tracing_if_enabled(context, enabled: bool):
    """
    Helper to start tracing with the required options if tracing is enabled.
    """
    if enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_trace_and_clean(context, enabled: bool):
    """
    Helper to stop tracing, save to unique path and clean it.
    """
    if enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        # Stop tracing and export
        context.tracing.stop(path=trace_path)
        # Clean the trace
        trace_cleaner(trace_path)


def test_expensify(pytestconfig):
    """
    Ported test with tracing integration and try/finally cleanup.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        first_user_context = browser.new_context(ignore_https_errors=True)

        # Start tracing if the flag is enabled
        _start_tracing_if_enabled(first_user_context, trace_enabled)

        try:
            page = first_user_context.new_page()
            install_online_guard_sync(first_user_context, page)
            page.wait_for_timeout(3000)

            # Existing test steps
            new_dot_login(page, EMAIL)
            page.wait_for_timeout(3000)

            page.locator("button[aria-label='My settings']").click()
            page.locator("button[aria-label='Share code']").click()

            copy_url_button = page.locator("div[aria-label='Copy URL']")
            cursor_style = copy_url_button.evaluate(
                "element => window.getComputedStyle(element).cursor"
            )
            assert cursor_style == "pointer"
            copy_url_button.click()

            copied_code = page.locator("div[aria-label='Copied!']")
            cursor_style = copied_code.evaluate(
                "element => window.getComputedStyle(element).cursor"
            )
            assert cursor_style == "default"

            page.wait_for_timeout(1500)

        finally:
            # Stop tracing and clean up if it was enabled
            _stop_trace_and_clean(first_user_context, trace_enabled)

            # Ensure pages/contexts/browsers are closed even on failure
            try:
                page.close()
            except Exception:
                # Page might already be closed; ignore safely
                pass
            browser.close()
