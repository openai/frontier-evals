import email
import imaplib
import os
import random
import re
import sys
import time
import uuid

import pytest
from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

email_address = "t56826807+10@gmail.com"
password = "wxck tsaw ocna tkfd"


class EmailHandler:
    def __init__(self, email_address=email_address, password=password):
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password
        random_suffix = random.randint(1000, 9999)
        current_timestamp = int(time.time())
        random_suffix = f"{current_timestamp}{random_suffix}"
        self.new_email = self._generate_new_email(email_address, random_suffix)

    def _validate_email(self, email_address):
        return bool(re.match("[^@]+@[^@]+\\.[^@]+", email_address))

    def _generate_new_email(self, email_address, suffix):
        username, domain = email_address.split("@")
        # Keeping the original email intact as per original implementation.
        return email_address

    def get_email_address(self):
        return self.new_email

    def check_for_otp(self, recipient: str | None = None, retries=5, delay=5):
        """
        Check for OTP in the Gmail inbox.
        """
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.original_email, self.password)
            for _ in range(retries):
                mail.select("inbox")
                status, messages = mail.search(
                    None, '(UNSEEN SUBJECT "Expensify magic sign-in code:")'
                )
                if status == "OK":
                    email_ids = messages[0].split()
                    if email_ids:
                        latest_email_id = email_ids[-1]
                        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
                        for response_part in msg_data:
                            if isinstance(response_part, tuple):
                                msg = email.message_from_bytes(response_part[1])
                                if recipient and msg["To"] != recipient:
                                    continue
                                otp_code = self._extract_otp_from_email(msg)
                                if otp_code:
                                    mail.store(latest_email_id, "+FLAGS", "\\Deleted")
                                    mail.expunge()
                                    mail.close()
                                    mail.logout()
                                    return otp_code
            mail.close()
            mail.logout()
            print("Max retries reached. No OTP found.")
            return None
        except imaplib.IMAP4.error as e:
            print(f"Failed to connect to Gmail: {e}")
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


def login_user(page: Page, email: str):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.locator('button[tabindex="0"]').click()
    page.locator('input[name="validateCode"]').fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


def complete_onboarding(page: Page, name: str):
    page.locator("text='Track and budget expenses'").click()
    page.locator('input[name="fname"]').fill(name)
    page.locator('input[name="lname"]').fill("")
    page.get_by_role("button", name="Continue").last.click()


@pytest.mark.parametrize("viewport", [(1280, 720)])
def test_expensify_manage_tags(viewport, pytestconfig):
    """
    Reproduces the original behaviour of the `test` function while adding
    Playwright tracing support behind the `--user-tool-trace` flag.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as playwright:
        # ------------------ Browser / Context setup ------------------ #
        browser = playwright.chromium.launch(
            headless=True,
            slow_mo=1000,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]})

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            # ------------------ Test steps ------------------ #
            page = context.new_page()
            install_online_guard_sync(
                context, page
            )  # Install the online guard immediately after page creation

            client = EmailHandler()
            email_to_use = client.get_email_address()

            login_user(page, email_to_use)

            page.locator('button[aria-label="Workspaces"]').click()
            page.get_by_test_id("WorkspaceSwitcherPage").get_by_role(
                "button", name="New workspace"
            ).click()
            page.locator('div[aria-label="More features"]').last.click()
            page.locator(
                'button[aria-label="Classify costs and track billable expenses."]'
            ).last.click()
            page.locator('div[aria-label="Tags"]').last.click()
            page.locator("button", has_text="Add tag").last.click()
            page.locator('input[aria-label="Name"]').last.fill("abc")
            page.locator('button[data-listener="Enter"]', has_text="Save").last.click()
            page.locator('div[aria-label="Select all"]').last.click()
            page.locator("button", has_text="1 selected").last.click()
            page.locator('div[aria-label="Delete tag"]').last.click()
            page.locator('button[data-listener="Enter"]', has_text="Delete").last.click()

            visible = page.locator(
                'div[dir="auto"]', has_text="You haven't created any tags"
            ).is_visible()
            assert visible, "Blank screen after deleting all tags!"

        finally:
            # ------------------ Tracing teardown ------------------ #
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # ------------------ Browser teardown ------------------ #
            context.close()
            browser.close()
