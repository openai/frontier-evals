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
from datetime import datetime, timezone
from email.header import decode_header

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

EMAIL_USERNAME = "aliceoglualice@gmail.com"
EMAIL_PASSWORD = "cfpy ofxm lcmq aeta"


def get_magic_code(user_email, password, since=None, retries=5, delay=10):
    if since is None:
        since = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user_email, password)
    for _ in range(retries):
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
                        email_date = email.utils.parsedate_to_datetime(msg["Date"])
                        if match and email_date >= since:
                            code = match.group(1)
                            imap.logout()
                            return code
                print("Email doesn't match conditions. Retrying...")
            else:
                print("No unread emails found with the subject. Retrying...")
        else:
            print("Failed to retrieve emails. Retrying...")
        time.sleep(delay)
    imap.logout()
    print("Max retries reached. Email not found.")
    return None


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool = False,
    first_name: str = "Milan",
    last_name: str = "T",
) -> tuple[Browser, "BrowserContext", Page, str]:
    """
    Launches a browser, optionally starts tracing (if trace_enabled=True),
    and logs the user into the application.

    Returns
    -------
    tuple
        (browser, context, page, user_email)
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--ignore-certificate-errors",
        ],
    )
    # Use an explicit context so we can control tracing
    context = browser.new_context(ignore_https_errors=True)
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Installed online guard immediately after page creation
    user_email = EMAIL_USERNAME

    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.get_by_test_id("username").fill(user_email)
    page.get_by_role("button", name="Continue").click()

    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=30000
        )
    except (AssertionError, TimeoutError):
        # The user is not yet registered – go through sign-up flow
        page.get_by_test_id("SignInPage").get_by_role("button", name="Join").click()
    else:
        # User exists, fill dummy OTP
        otp_code = "123456"
        otp_input = page.locator('input[autocomplete="one-time-code"]')
        otp_input.fill(otp_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass

    try:
        expect(page.get_by_text("What do you want to do today?")).to_be_visible(timeout=30000)
    except (AssertionError, TimeoutError):
        # Already past onboarding
        pass
    else:
        # Complete onboarding flow
        page.get_by_label("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()

    return browser, context, page, user_email


def test_edit_box_and_compose_box_alignment(pytestconfig):
    """
    Verifies that the edit box and the compose box are vertically aligned.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        # Launch browser and (optionally) start tracing
        browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)

        try:
            personal_chat = page.get_by_text("Milan T (you)").first
            expect(personal_chat).to_be_visible(timeout=30000)
            personal_chat.click()

            # Send three messages
            page.get_by_role("textbox").fill("My Message")
            page.get_by_role("textbox").press("Enter")
            page.wait_for_timeout(500)

            page.get_by_role("textbox").fill("My Second Message")
            page.get_by_role("textbox").press("Enter")
            page.wait_for_timeout(500)

            page.get_by_role("textbox").fill("My Third Message")
            page.get_by_role("textbox").press("Enter")
            page.wait_for_timeout(500)

            # Edit the third message
            message_three = page.locator("span", has_text="My Third Message")
            message_three.click(button="right")

            option_menu = page.locator('div[aria-label="Edit comment"]', has_text="Edit comment")
            option_menu.click()
            page.wait_for_timeout(2000)

            element1 = page.locator("button[aria-label='Close']")
            element2 = page.locator("button[aria-label='Create']").nth(1)

            element1_x = element1.bounding_box()["x"]
            element2_x = element2.bounding_box()["x"]

            assert element1_x == element2_x, (
                f"Elements are not vertically aligned. Element1 x: {element1_x}, Element2 x: {element2_x}"
            )

        finally:
            # Stop tracing (if enabled) and clean traces
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Always close the context and the browser
            context.close()
            browser.close()
