import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import random
import re
import time
from typing import Any

from playwright.sync_api import Browser, Page, Playwright, sync_playwright

master_email_address = "brighturing926+49492@gmail.com"
master_app_password = "sijknxflhloopmta"
dev_base_url = "https://dev.new.expensify.com:8082"


class EmailHandler:
    def __init__(self, email_address, password) -> Any:
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password
        random_suffix = random.randint(1000, 9999)
        current_timestamp = int(time.time())
        random_suffix = f"{current_timestamp}{random_suffix}"
        self.new_email = email_address

    def _validate_email(self, email_address) -> Any:
        return bool(re.match("[^@]+@[^@]+\\.[^@]+", email_address))

    def get_email_address(self) -> Any:
        return self.new_email

    def check_for_otp(self) -> Any:
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.original_email, self.password)
            mail.select("inbox")
            status, messages = mail.search(None, "UNSEEN")
            if status != "OK":
                print("No unread emails found.")
                return None
            email_ids = messages[0].split()
            email_ids = email_ids[::-1]
            for email_id in email_ids:
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                if status != "OK":
                    print("Error fetching email.")
                    continue
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        if msg["To"] == self.new_email:
                            otp_code = self._extract_otp_from_email(msg)
                            if otp_code:
                                mail.store(email_id, "+FLAGS", "\\Deleted")
                                mail.expunge()
                                mail.close()
                                mail.logout()
                                return otp_code
            mail.close()
            mail.logout()
            print("No OTP found in unread emails.")
            return None
        except imaplib.IMAP4.error:
            print("Failed to connect to Gmail. Please check your email address or password.")
            return None

    def _extract_otp_from_email(self, msg) -> Any:
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

    def _extract_otp(self, text) -> Any:
        match = re.search("\\b\\d{6}\\b", text)
        return match.group(0) if match else None


def login_user(p: Playwright, user_email: str) -> tuple[Browser, BrowserContext, Page]:
    """
    Launches a browser, creates a context and logs the user in.
    Returns the browser, context and the created page.
    """
    browser: Browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=500,
    )
    context: BrowserContext = browser.new_context()
    page: Page = context.new_page()

    # Install online guard immediately after context and page creation
    install_online_guard_sync(context, page)

    page.goto(dev_base_url, timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(5000)

    # Fill OTP (dummy for tests)
    my_otp_code = "111111"
    page.get_by_test_id("validateCode").fill(my_otp_code)
    try:
        page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass
    try:
        page.get_by_role("button", name="Sign in").click()
    except Exception:
        # Sometimes the button is not present if auto-logged in
        pass

    return browser, context, page


def test_enter_key_to_delete(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser, context, page = login_user(p, master_email_address)

        # Start tracing immediately after context creation if enabled
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page.get_by_test_id("CustomBottomTabNavigator").get_by_label("Search").click()
            page.wait_for_timeout(1000)

            page.get_by_role("button", name="Filters").click()
            page.wait_for_timeout(1000)

            page.get_by_test_id("SearchAdvancedFiltersPage").get_by_role(
                "menuitem", name="Category"
            ).click()
            page.wait_for_timeout(1000)

            page.locator("#Equipment").click()
            page.get_by_role("button", name="Save").click()
            page.wait_for_timeout(1000)

            page.get_by_test_id("SearchAdvancedFiltersPage").get_by_role(
                "menuitem", name="Currency"
            ).click()
            page.wait_for_timeout(1000)

            page.get_by_test_id("selection-list-text-input").fill("usd")
            page.locator('[id="USD\\ -\\ $"]').click()
            page.wait_for_timeout(1000)

            page.get_by_role("button", name="Save").click()
            page.get_by_role("button", name="Save search").click()
            page.wait_for_timeout(1000)

            page.get_by_label("More").first.click()
            page.get_by_label("Delete").click()
            page.wait_for_timeout(1000)

            page.locator("body").press("Enter")
            page.wait_for_timeout(1000)

            delete_model = page.get_by_role("button", name="Delete")
            assert not delete_model.is_visible(), (
                "Enter key not working, Delete modal is still visible"
            )

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            browser.close()
