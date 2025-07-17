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

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright

EMAIL = "mancity9271+1005@gmail.com"
APP_PASSWORD = "aplp neek apqv cjbr"
NEWDOT_URL = "https://dev.new.expensify.com:8082/"


class IMAPOTPExtractor:
    def __init__(self, email_address, password):
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password

    def _validate_email(self, email_address):
        return bool(re.match("[^@]+@[^@]+\\.[^@]+", email_address))

    def set_new_email(self, email):
        self.new_email = email

    def check_for_otp(self):
        otp = "123456"
        return otp
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
                    print("Failed to fetch email ID:", email_id)
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
        except imaplib.IMAP4.error as e:
            print(f"IMAP error: {e}")
            return None

    def _extract_otp_from_email(self, msg):
        email_body = None
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_body = part.get_payload(decode=True).decode()
                    break
        else:
            email_body = msg.get_payload(decode=True).decode()
        if email_body:
            otp_code = self._extract_otp(email_body)
            return otp_code
        return None

    def _extract_otp(self, text):
        match = re.search("\\b\\d{6}\\b", text)
        return match.group(0) if match else None


class UserLogin:
    def new_dot_login(self, page, email):
        page.goto(NEWDOT_URL, timeout=60000)
        page.locator('input[type="email"]').fill(email)
        page.wait_for_timeout(2000)
        page.get_by_role("button", name="Continue").nth(0).click()
        page.wait_for_timeout(10000)
        otp = "123456"
        if otp:
            page.locator('input[data-testid="validateCode"]').fill(otp)
            try:
                page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                    timeout=20000
                )
            except Exception:
                pass
        else:
            print("OTP not received.")


def test_invoice_report_amount(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser: Browser = p.chromium.launch(
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
            page: Page = context.new_page()
            install_online_guard_sync(context, page)  # Added guard installation

            user = UserLogin()
            user.new_dot_login(page, EMAIL)
            page.wait_for_timeout(2000)
            page.get_by_label("My settings").click()
            page.locator(
                "div:nth-child(2) > div:nth-child(2) > div > div > div > div > div > div > div > div:nth-child(2) > .css-view-175oi2r"
            ).click()
            page.wait_for_timeout(1000)
            page.get_by_label("New workspace").first.click()
            page.locator(
                "div:nth-child(2) > div:nth-child(5) > div > div > div > div > div > div > div > div > div:nth-child(2) > div"
            ).click()
            page.get_by_test_id("selection-list-text-input").click()
            page.get_by_test_id("selection-list-text-input").press("CapsLock")
            page.wait_for_timeout(1000)
            page.get_by_test_id("selection-list-text-input").fill("K")
            page.get_by_test_id("selection-list-text-input").press("CapsLock")
            page.wait_for_timeout(1000)
            page.get_by_test_id("selection-list-text-input").fill("Kes")
            page.get_by_test_id("selection-list").get_by_label("KES - KSh").click()
            page.get_by_text("Members").click()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Invite member").click()
            page.get_by_test_id("selection-list-text-input").fill("test@gmail.com")
            page.wait_for_timeout(2000)
            page.locator('[id="\\31 78516"]').get_by_label("Joanna Cheng").click()
            page.get_by_role("button", name="Next").click()
            page.wait_for_timeout(1000)
            page.get_by_test_id("WorkspaceInviteMessagePage").get_by_role(
                "button", name="Invite"
            ).click()
            page.get_by_text("More features").click()
            page.get_by_label("Send and receive invoices.").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Back").click()
            page.get_by_label("Inbox").click()
            page.get_by_label("Start chat (Floating action)").click()
            page.wait_for_timeout(1000)
            page.get_by_text("Send invoice").click()
            page.get_by_label("Select a currency").click()
            page.get_by_test_id("selection-list-text-input").press("CapsLock")
            page.wait_for_timeout(1000)
            page.get_by_test_id("selection-list-text-input").fill("K")
            page.get_by_test_id("selection-list-text-input").press("CapsLock")
            page.wait_for_timeout(1000)
            page.get_by_test_id("selection-list-text-input").fill("Kes ")
            page.get_by_label("KES - KSh").click()
            page.get_by_placeholder("0").fill("100")
            page.get_by_role("button", name="Next").click()
            page.wait_for_timeout(1000)
            page.get_by_test_id("selection-list-text-input").fill("test@gmail.com")
            page.wait_for_timeout(2000)
            page.get_by_test_id("selection-list").get_by_label("Joanna Cheng").click()
            page.get_by_role("button", name="Next").click()
            page.get_by_role("textbox", name="Your company name").press("CapsLock")
            page.wait_for_timeout(1000)
            page.get_by_role("textbox", name="Your company name").fill("JTL")
            page.get_by_role("textbox", name="Your company website").click()
            page.get_by_role("textbox", name="Your company website").fill(
                "https://qa-trainer-jschool.web.app/"
            )
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="Send KES 100.00 invoice").click()
            page.wait_for_timeout(2000)
            page.get_by_label("View details").click()
            page.locator(
                "div:nth-child(2) > div > div:nth-child(3) > div > div > div > div > div > div > div > div > div > div > .css-view-175oi2r"
            ).click()
            page.get_by_label("Select a currency").click()
            page.get_by_test_id("selection-list-text-input").fill("USD")
            page.wait_for_timeout(2000)
            page.get_by_label("USD - $").click()
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="Save").click()
            page.wait_for_timeout(3000)
            expect(page.get_by_text("TotalKES 0.00Out-of-pocket")).not_to_be_visible()
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
