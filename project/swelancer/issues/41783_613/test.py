import os
import sys
import uuid

import pytest
from utils.online_guard import install_online_guard_async
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import logging
import re
import time
from email.header import decode_header

from playwright.async_api import Page, async_playwright, expect

NEW_DOT_URL = "https://dev.new.expensify.com:8082"
EXPENSIFY_EMAIL = os.getenv("USER_EMAIL", "testingq186+41783@gmail.com")
EXPENSIFY_PASSWORD = os.getenv("PASSWORD", "kkfr hqiv yuxq rbso")
FIRST_NAME = "Testing"
LAST_NAME = "41783"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@pytest.mark.asyncio
async def test_41783(pytestconfig):
    """
    Ported version of the original test with optional Playwright tracing support.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            proxy={"server": "http://localhost:8080"},
            slow_mo=500,
        )
        context = await browser.new_context(ignore_https_errors=True)
        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = await context.new_page()
            await install_online_guard_async(context, page)

            class EmailHandler:
                """
                A class to handle email operations such as cleaning the inbox,
                marking all unread emails as read, and reading OTP codes.
                """

                def __init__(self, user_email, password, imap_server: str = "imap.gmail.com"):
                    self.user_email = user_email
                    self.password = password
                    self.imap_server = imap_server
                    self.imap = None

                def __enter__(self):
                    self.imap = imaplib.IMAP4_SSL(self.imap_server)
                    try:
                        self.imap.login(self.user_email, self.password)
                        logging.info("Logged into IMAP server.")
                    except Exception as e:
                        logging.error(f"Failed to login to IMAP server: {e}")
                        raise
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    if self.imap:
                        self.imap.logout()
                        logging.info("Logged out from IMAP server.")

                # ---------- helper email methods ----------
                def clean_inbox(self):
                    logging.warning("Deleting all emails in the inbox.")
                    status, _ = self.imap.select("INBOX")
                    if status != "OK":
                        logging.error("Failed to select INBOX.")
                        return
                    status, messages = self.imap.search(None, "ALL")
                    if status != "OK":
                        logging.error("Failed to retrieve emails.")
                        return
                    email_ids = messages[0].split()
                    if not email_ids:
                        logging.info("No emails to delete.")
                        return
                    for email_id in email_ids:
                        self.imap.store(email_id, "+FLAGS", "\\Deleted")
                    self.imap.expunge()
                    logging.info("All emails deleted from the inbox.")

                def mark_all_unread_as_read(self):
                    logging.info("Marking all unread emails as read.")
                    status, _ = self.imap.select("INBOX")
                    if status != "OK":
                        logging.error("Failed to select INBOX.")
                        return
                    status, messages = self.imap.search(None, "(UNSEEN)")
                    if status != "OK":
                        logging.error("Failed to retrieve unread emails.")
                        return
                    email_ids = messages[0].split()
                    if not email_ids:
                        logging.info("No unread emails to mark as read.")
                        return
                    for email_id in email_ids:
                        self.imap.store(email_id, "+FLAGS", "\\Seen")
                    logging.info("All unread emails marked as read.")

                def read_otp_code(self, retries: int = 5, delay: int = 6):
                    logging.info("Attempting to read OTP code from emails.")
                    for i in range(retries):
                        self.imap.select("inbox")
                        status, messages = self.imap.search(
                            None, '(UNSEEN SUBJECT "Expensify magic sign-in code:")'
                        )
                        if status != "OK":
                            logging.error(
                                f"Failed to search for emails. Retrying {i + 1}/{retries}..."
                            )
                            time.sleep(delay)
                            continue
                        email_ids = messages[0].split()
                        if not email_ids:
                            logging.info(
                                f"Failed to retrieve emails. Retrying {i + 1}/{retries}..."
                            )
                            time.sleep(delay)
                            continue
                        latest_email_id = email_ids[-1]
                        status, msg_data = self.imap.fetch(latest_email_id, "(RFC822)")
                        for response_part in msg_data:
                            if isinstance(response_part, tuple):
                                msg = email.message_from_bytes(response_part[1])
                                subject, encoding = decode_header(msg["Subject"])[0]
                                if isinstance(subject, bytes):
                                    subject = subject.decode(encoding or "utf-8")
                                match = re.search(r"Expensify magic sign-in code: (\d+)", subject)
                                if match:
                                    return match.group(1)
                        logging.info(f"No matching emails found. Retrying {i + 1}/{retries}...")
                        time.sleep(delay)
                    logging.warning("Max retries reached. OTP code not found.")
                    return None

            def read_otp_from_email(email_addr: str, password: str):
                with EmailHandler(email_addr, password) as email_handler:
                    email_handler.clean_inbox()
                    return email_handler.read_otp_code()

            async def complete_onboarding():
                await page.locator("text=Track and budget expenses").click()
                await page.locator('input[name="fname"]').fill(FIRST_NAME)
                await page.locator('input[name="lname"]').fill(LAST_NAME)
                await page.get_by_role("button", name="Continue").last.click()

            async def sign_in_new_dot(page: Page, email_addr: str, password: str):
                await page.goto(NEW_DOT_URL, timeout=60000)
                await page.get_by_test_id("username").fill(email_addr)
                await page.get_by_role("button", name="Continue").click()
                try:
                    await expect(page.get_by_test_id("validateCode")).to_be_visible()
                    await page.get_by_test_id("validateCode").fill("111111")
                    try:
                        await page.get_by_role(
                            "button",
                            name=re.compile(r"^(continue|sign.?in)$", re.I),
                        ).click(timeout=20000)
                    except Exception:
                        pass
                except Exception:
                    await page.get_by_role("button", name="Join").click()
                try:
                    await expect(page.get_by_text("What do you want to do today?")).to_be_visible()
                    await complete_onboarding()
                except Exception:
                    pass

            async def test_app_returns_to_confirmation_page_when_tapping_back_button_on_category_page(
                page: Page,
            ):
                await page.get_by_label("Start chat (Floating action)").click()
                await page.get_by_label("Track expense").first.click()
                await page.get_by_role("button", name="Got it").click()
                await page.get_by_label("Manual").click()
                await page.get_by_placeholder("0").fill("1000")
                await (
                    page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
                )
                await page.get_by_role("button", name="Track expense").click()
                await page.get_by_role("button", name="Categorize it").first.click()
                await page.get_by_label("Back").click()
                await expect(
                    page.get_by_test_id("IOURequestStepConfirmation").get_by_text("Categorize")
                ).not_to_be_visible()

            # ----- actual test steps -----
            await sign_in_new_dot(page, EXPENSIFY_EMAIL, EXPENSIFY_PASSWORD)
            await test_app_returns_to_confirmation_page_when_tapping_back_button_on_category_page(
                page
            )

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                await context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            await browser.close()
