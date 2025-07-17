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
import re

from playwright.async_api import async_playwright, expect

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
EMAIL = "gallahad.420@gmail.com"
PASSWORD = "ndbq rbvo honh lkke"


class IMAPOTPExtractor:
    def __init__(self, email_address, password):
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password
        self.new_email = None

    def _validate_email(self, email_address):
        return bool(re.match("[^@]+@[^@]+\\.[^@]+", email_address))

    def set_new_email(self, email):
        self.new_email = email

    async def check_for_otp(self):
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.original_email, self.password)
            mail.select("inbox")
            status, messages = mail.search(None, "UNSEEN")
            if status != "OK":
                return "123456"
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
            return "123456"
        except imaplib.IMAP4.error:
            return "123456"

    def _extract_otp_from_email(self, msg):
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    email_body = part.get_payload(decode=True).decode()
                    return self._extract_otp(email_body)
        else:
            email_body = msg.get_payload(decode=True).decode()
            return self._extract_otp(email_body)
        return None

    def _extract_otp(self, text):
        match = re.search("\\b\\d{6}\\b", text)
        return match.group(0) if match else None


async def new_dot_login(page, email):
    await page.goto(NEWDOT_URL, timeout=60000)
    await page.locator('input[type="email"]').fill(email)
    await page.wait_for_timeout(2000)
    await page.get_by_role("button", name="Continue").nth(0).click()
    await page.wait_for_timeout(10000)
    otp = "123456"
    await page.locator('input[data-testid="validateCode"]').fill(otp)
    try:
        await page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass
    try:
        await page.get_by_role("button", name="Sign In").click()
    except Exception:
        pass


async def emoji_check(page, first_name="Test", last_name="User"):
    try:
        await page.locator('button[tabindex="0"]').click()
        await page.wait_for_timeout(1000)
    except Exception:
        pass
    new_user_login = True
    try:
        await expect(page.locator("text=What do you want to do today?")).to_be_visible()
    except Exception:
        new_user_login = False
    if new_user_login:
        await page.locator("text='Track and budget expenses'").click()
        await page.get_by_role("button", name="Continue").click()
        await page.wait_for_timeout(1000)
        await page.locator('input[name="fname"]').fill(first_name)
        await page.locator('input[name="lname"]').fill(last_name)
        await page.get_by_role("button", name="Continue").last.click()
        await page.wait_for_timeout(1000)
    await page.locator('button[aria-label="Start chat (Floating action)"][tabindex="0"]').click()
    await page.locator('div[aria-label="Start chat"][role="menuitem"]').click()
    email = "unilogics+test@gmail.com"
    input_field = page.locator('input[data-testid="selection-list-text-input"]')
    await input_field.fill(email)
    await page.wait_for_timeout(1000)
    await page.locator('button[aria-label="Dev H"][tabindex="0"]').last.click()
    await page.wait_for_timeout(2000)
    await page.locator('div[aria-label="Emoji"][data-tag="pressable"]').last.click()
    await page.wait_for_timeout(2000)
    message = page.locator(
        'button[aria-label="gallahad.420@gmail.com"][tabindex="0"]',
        has_text="Test User",
    ).last
    await message.click(button="right")
    await page.wait_for_timeout(2000)
    check_for_box = page.locator('div[aria-label="Reply in thread"][role="menuitem"]').last
    await expect(check_for_box).to_be_visible()


@pytest.mark.asyncio
async def test_expensify(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = await browser.new_context(ignore_https_errors=True)

        if trace_enabled:
            # Start tracing
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = await context.new_page()
            await install_online_guard_async(context, page)
            await new_dot_login(page, EMAIL)
            await emoji_check(page)
        finally:
            # Stop tracing, export, and clean if enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                await context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            await context.close()
            await browser.close()
