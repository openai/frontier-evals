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
import time
from email.header import decode_header

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright

password = "ywvy nytt kjig mscy"
user_email = "jellystarfish99@gmail.com"
magic_code = "111111"


def get_magic_code(user_email, password, retries=5, delay=10):
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user_email, password)
    for _attempt in range(retries):
        imap.select("inbox")
        status, messages = imap.search(None, '(UNSEEN SUBJECT "Expensify magic sign-in code:")')
        if status == "OK":
            email_ids = messages[0].split()
            if email_ids:
                latest_email_id = email_ids[-1]
                status, msg_data = imap.fetch(latest_email_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding or "utf-8")
                        match = re.search(r"Expensify magic sign-in code: (\d+)", subject)
                        if match:
                            code = match.group(1)
                            imap.logout()
                            return code
            else:
                print("No unread emails found with the subject. Retrying...")
        else:
            print("Failed to retrieve emails. Retrying...")
        time.sleep(delay)
    imap.logout()
    print("Max retries reached. Email not found.")
    return None


def create_user(page: Page, firstname: str):
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    validate_code_input = page.locator('input[data-testid="validateCode"]')
    expect(validate_code_input).to_be_visible()
    validate_code_input.fill(magic_code)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    page.wait_for_timeout(2000)
    if page.locator("text=What do you want to do today?").is_visible():
        page.get_by_text("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(firstname)
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Continue").click()


def close_button_if_present(page: Page):
    """
    Occasionally, there is a close button that prevents any clicks on the page as
    it covers most of the screen. This button cannot be seen visually.
    """
    close_button = page.locator('button[aria-label="Close"]')
    if close_button.is_visible():
        close_button.click()


def test_newline_indicator(pytestconfig):
    """
    Verifies the presence of the 'New message line indicator' after marking
    a parent message as unread.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)

            create_user(page, "TestUser")
            page.wait_for_timeout(1000)

            close_button_if_present(page)

            green_button = page.locator(
                'button[aria-label="Start chat (Floating action)"][tabindex="0"]'
            )
            green_button.click()
            page.wait_for_timeout(1000)

            start_chat = page.locator('div[aria-label="Start chat"][role="menuitem"]')
            start_chat.click()
            page.wait_for_timeout(1000)

            input_field = page.locator('input[data-testid="selection-list-text-input"]')
            input_field.fill(user_email)
            page.wait_for_timeout(1000)

            page.get_by_text("Test (you)").click()
            page.get_by_role("textbox").fill("Parent")
            page.keyboard.press("Enter")
            page.wait_for_timeout(1000)

            page.get_by_test_id("report-actions-list").get_by_text("Parent").last.click(
                button="right"
            )
            page.wait_for_timeout(1000)

            page.get_by_label("Reply in thread").last.click()
            page.wait_for_timeout(1000)

            page.get_by_role("textbox").fill("Thread 1")
            page.wait_for_timeout(1000)
            page.keyboard.press("Enter")
            page.wait_for_timeout(1000)

            page.get_by_text("Parent").last.click(button="right")
            page.wait_for_timeout(1000)

            page.get_by_label("Mark as unread").last.click()
            page.wait_for_timeout(1000)

            expect(page.get_by_label("New message line indicator").last).to_be_visible()

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close resources regardless of test outcome
            context.close()
            browser.close()
