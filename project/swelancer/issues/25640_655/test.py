import os
import random
import re
import sys
import time
import uuid

import pytest

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "ikhanle001+23@gmail.com"
PASSWORD = "dzcr knad otxp bsuz"


class IMAPOTPExtractor:
    def __init__(self, email_address, password):
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password
        random_suffix = random.randint(1000, 9999)
        current_timestamp = int(time.time())
        # NOTE: random_suffix is generated but unused just like in the original code.
        random_suffix = f"{current_timestamp}{random_suffix}"

    def _validate_email(self, email_address):
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email_address))

    def set_new_email(self, email):
        self.new_email = email

    def check_for_otp(self):
        try:
            import email as email_lib
            import imaplib

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
                        msg = email_lib.message_from_bytes(response_part[1])
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


def login_user(page: Page, email: str):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.locator('button[tabindex="0"]').click()
    time.sleep(1)
    otp = "123456"
    page.locator('input[data-testid="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


@pytest.mark.parametrize("viewport", [{"width": 1280, "height": 720}])
def test_expensify_group_split(viewport, pytestconfig):
    """
    Ported test with tracing support controlled by --user-tool-trace.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    width = viewport["width"]
    height = viewport["height"]

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(
            viewport={"width": width, "height": height},
            timezone_id="Europe/Bucharest",
        )

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)

            # Original test actions
            email = EMAIL
            login_user(page, email)

            # Start chat
            page.locator('button[aria-label="Start chat (Floating action)"][tabindex="0"]').click()
            page.locator('div[aria-label="Start chat"][role="menuitem"]').click()

            # Add three new users to group chat
            for i in range(3):
                email = f"example_{i}_{int(time.time())}@gmail.com"
                input_field = page.locator('input[data-testid="selection-list-text-input"]')
                input_field.fill(email)
                time.sleep(1)
                page.get_by_role("button", name="Add to group").click()

            page.wait_for_timeout(3000)
            page.keyboard.press("Enter")
            page.wait_for_timeout(3000)
            page.keyboard.press("Enter")

            # Create manual split
            page.wait_for_timeout(4000)
            page.locator('button[aria-label="Create"]').last.click()
            page.locator('div[aria-label="Split expense"]').click()
            page.locator('button[aria-label="Manual"]').last.click()
            page.locator('input[role="presentation"]').fill("1000")
            page.wait_for_timeout(3000)
            page.keyboard.press("Enter")
            page.wait_for_timeout(3000)
            page.keyboard.press("Enter")

            split_text = page.locator('div[aria-label="Split"]').inner_text()
            print(split_text)
            # Ensuring "(none)" is not present and last generated test user not in split result
            assert "(none)" not in split_text and f"testuser{i}@gmail.com" not in split_text

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
