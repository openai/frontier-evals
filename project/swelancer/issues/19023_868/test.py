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
import re
import time

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

USER_EMAIL = "playwright2user+1@gmail.com"
PASSWORD = "zehr mglm gizg gjcc"


def fetch_verification_code_from_email(user_email, password, retries=10, delay=10):
    """
    Fetch the OTP code from the latest email.
    """
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user_email, password)
    for _attempt in range(retries):
        imap.select("inbox")
        status, messages = imap.search(None, '(UNSEEN SUBJECT "Expensify magic sign-in code")')
        if status == "OK":
            email_ids = messages[0].split()
            if email_ids:
                latest_email_id = email_ids[-1]
                status, msg_data = imap.fetch(latest_email_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                if content_type == "text/plain":
                                    body = part.get_payload(decode=True).decode()
                                    match = re.search(r"\b\d{6}\b", body)
                                    if match:
                                        otp_code = match.group(0)
                                        imap.logout()
                                        return otp_code
                        else:
                            body = msg.get_payload(decode=True).decode()
                            match = re.search(r"\b\d{6}\b", body)
                            if match:
                                otp_code = match.group(0)
                                imap.logout()
                                return otp_code
            else:
                print("No new emails found. Retrying...")
                otp_code = "123456"
                return otp_code
        else:
            print("Failed to retrieve emails. Retrying...")
        time.sleep(delay)
    imap.logout()
    raise Exception("Max retries reached. No magic code email found.")


def generate_random_email():
    timestamp = int(time.time())
    return f"testuser{timestamp}@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Ayush",
    last_name: str = "G",
    trace_enabled: bool = False,
) -> tuple[Browser, "BrowserContext", Page, str]:
    """
    Launches the browser, performs OTP login and optionally starts tracing.
    Returns the browser instance, context, page and the user email used.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(ignore_https_errors=True)
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    # Install online guard immediately after page and context creation
    install_online_guard_sync(context, page)

    user_email = USER_EMAIL
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    time.sleep(1)
    verification_code = "123456"
    otp_input = page.locator("[autocomplete='one-time-code']")
    expect(otp_input).to_be_visible()
    otp_input.fill(verification_code)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    try:
        page.locator('button[tabindex="0"]').click()
        time.sleep(1)
    except Exception:
        pass
    return browser, context, page, user_email


def test_save_description_in_submitted_expense(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    context = None
    with sync_playwright() as p:
        try:
            browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)

            page.wait_for_timeout(5000)
            page.locator('span:text("Test (you)")').first.click()

            plus_create_icon = page.locator('button[aria-label="Create"]').last
            expect(plus_create_icon).to_be_visible()
            plus_create_icon.click()

            page.get_by_text("Assign task").click()
            page.get_by_role("textbox", name="Title").fill("sample title")
            page.get_by_role("textbox", name="Title").press("Tab")
            page.get_by_role("textbox", name="Description (optional)").fill("sample description")
            page.get_by_role("button", name="Next").click()
            page.locator('button div:text("Confirm task")').nth(0).click()

            page.wait_for_timeout(5000)

            task_title_div = page.locator(
                'div[data-testid="comment"] div:has-text("sample title")'
            ).last
            task_title_div.click()

            expanded_task_title_div = page.locator('div[aria-label="sample title"]').first
            expanded_task_title_div.click()

            page.get_by_role("textbox", name="Title").fill("")
            page.get_by_role("button", name="Save").click()

            page.wait_for_timeout(10000)

            title_missing_warning = page.locator('div:text("Please enter a title")')
            expect(title_missing_warning).to_be_visible()

            page.wait_for_timeout(10000)

            field_required_warning = page.locator('div:text("This field is required.")')
            expect(field_required_warning).to_be_hidden()

        finally:
            # Stop tracing and clean up
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                try:
                    context.tracing.stop(path=trace_path)
                except Exception:
                    # Ensure we don't mask original failures due to tracing stop issues
                    pass
                trace_cleaner(trace_path)

            if context is not None:
                try:
                    context.close()
                except Exception:
                    pass

            if browser is not None:
                browser.close()
