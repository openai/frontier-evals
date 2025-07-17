import email
import imaplib
import os
import random
import re
import sys
import time
import uuid

from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

# --------------------------------------------------------------------------- #
#  Path set-up so local utils can be imported regardless of current location  #
# --------------------------------------------------------------------------- #
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

# --------------------------------------------------------------------------- #
#                                  CONSTANTS                                  #
# --------------------------------------------------------------------------- #
NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "turingvidhant+109@gmail.com"
PASSWORD = "uotx fzkt kfeg dehz"


# --------------------------------------------------------------------------- #
#                              Helper / Utility Code                          #
# --------------------------------------------------------------------------- #
class IMAPOTPExtractor:
    def __init__(self, email_address, password):
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password
        # The random suffix is calculated but never used in original code.
        random_suffix = random.randint(1000, 9999)
        current_timestamp = int(time.time())

    @staticmethod
    def _validate_email(email_address) -> bool:
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
            email_ids = messages[0].split()[::-1]  # latest first
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

    @staticmethod
    def _extract_otp(text) -> str | None:
        match = re.search(r"\b\d{6}\b", text)
        return match.group(0) if match else None


def new_dot_login(page, email):
    """
    Logs into NewDot using a dummy OTP flow.
    """
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(3000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(3000)
    page.locator('input[data-testid="validateCode"]').fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    # Existing logic retained after mandated try/except
    try:
        page.get_by_role("button", name="Sign In").click()
    except Exception:
        # Occasionally the Sign In button is not present if already logged in.
        pass


def launch_browser(pw, headless=True, device=None, geolocation=None):
    """
    Utility launcher used elsewhere; kept for parity.
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
    return browser, context, page


# --------------------------------------------------------------------------- #
#                                   TESTS                                     #
# --------------------------------------------------------------------------- #
def test_expensify(pytestconfig):
    """
    Simple smoke test that validates DM preview is not visible
    on a fresh chat screen.
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
        context = browser.new_context(ignore_https_errors=True)

        # ------------------------------------------------------------------- #
        #                           TRACE START (if enabled)                  #
        # ------------------------------------------------------------------- #
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.wait_for_timeout(3000)

            new_dot_login(page, EMAIL)
            page.wait_for_timeout(3000)

            dm_preview = page.locator(
                'div[aria-label="Last chat message preview"]',
                has_text="This is the beginning of your chat",
            )
            expect(dm_preview).not_to_be_visible()

            page.wait_for_timeout(1500)
            page.close()

        finally:
            # ----------------------------- TRACE STOP ----------------------- #
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # ----------------------------- CLEAN-UP ------------------------- #
            context.close()
            browser.close()
