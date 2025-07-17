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

from playwright.sync_api import expect, sync_playwright

user_email = "totherandomuser@gmail.com"
password = "umei djwi bfmp jntm"


def get_magic_code(user_email, password, retries=5, delay=10):
    # Connect to the server
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

                        # Search for the magic code in the subject
                        match = re.search(r"Expensify magic sign-in code: (\d+)", subject)
                        if match:
                            code = match.group(1)
                            imap.logout()
                            return code
            else:
                print("No unread emails found with the subject. Retrying...")
        else:
            print("Failed to retrieve emails. Retrying...")

        # Wait for the specified delay before trying again
        time.sleep(delay)

    imap.logout()
    print("Max retries reached. Email not found.")
    return None


def login(p, if_phone=False, first_name="John", last_name="Cena"):
    """
    Launches a browser, creates a context (mobile if `if_phone` is True) and
    performs the sign-in flow up to filling the magic code.

    Returns
    -------
    Tuple(browser, context, page)
    """
    permissions = ["clipboard-read", "clipboard-write"]
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    if if_phone:
        phone = p.devices["iPhone 12 Pro"]
        context = browser.new_context(
            **phone, permissions=permissions, reduced_motion="no-preference"
        )
    else:
        context = browser.new_context()

    page = context.new_page()
    install_online_guard_sync(context, page)  # Installed online guard
    page.goto("https://dev.new.expensify.com:8082/")

    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()
    phone_or_email_input.fill(user_email)

    continue_button = page.locator('button[tabindex="0"]')
    expect(continue_button).to_be_visible()
    continue_button.click()

    page.wait_for_timeout(1000)

    magic_code = "123456"
    validate_code_input = page.locator('input[name="validateCode"]')
    expect(validate_code_input).to_be_visible()
    validate_code_input.fill(magic_code)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass

    page.wait_for_timeout(1000)

    return browser, context, page


def _start_tracing_if_enabled(context, enabled: bool):
    if enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_tracing_if_enabled(context, enabled: bool):
    """
    Stops tracing for the given context if enabled and cleans the resulting zip.
    """
    if enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def _close_browser_safely(browser):
    try:
        browser.close()
    except Exception:
        # Ignore errors while closing; test outcome should already be decided.
        pass


def test_unsubscribe_button_visibility(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = context = page = None  # Ensure variables exist for finally block
        try:
            # Step 1: Login
            browser, context, page = login(p, False)
            _start_tracing_if_enabled(context, trace_enabled)

            page.wait_for_timeout(5000)

            # Locate the div containing the specific text and click it
            group_chat = page.get_by_text("Totherandomuser's Workspace 2").first
            expect(group_chat).to_be_visible()
            group_chat.click()

            page.wait_for_timeout(1000)

            page.get_by_role("textbox").fill("My Message")
            page.get_by_role("textbox").press("Enter")
            page.wait_for_timeout(1000)

            page.get_by_label("My settings").click()
            page.get_by_label("Preferences").click()
            page.get_by_label("Force offline").click()
            page.locator("#overLayBottomButton").click()
            page.wait_for_timeout(1000)

            message = page.locator("span", has_text="My Message").last
            expect(message).to_be_visible()
            message.click(button="right")
            page.wait_for_timeout(1000)

            # Locate the element using aria-label and click it
            delete_button = page.locator('[aria-label="Delete comment"]').first
            delete_button.click()
            page.get_by_role("button", name="Delete").click()

            page.wait_for_timeout(1000)
            message = page.locator("span", has_text="My Message").last
            expect(message).to_be_visible()
            message.click(button="right")
            page.wait_for_timeout(1000)

            # Locate the div with the data attribute and the specific text
            unsubscribe_div = page.locator(
                'div[data-selection-scrapper-hidden-element="false"].css-text-146c3p1',
                has_text="Unsubscribe from thread",
            )
            is_unsubscribe_present = unsubscribe_div.is_visible()

            print(is_unsubscribe_present)

            assert is_unsubscribe_present is False, (
                "Unsubscribe from thread Button should not be visible."
            )

        finally:
            if context:
                _stop_tracing_if_enabled(context, trace_enabled)
            if browser:
                _close_browser_safely(browser)
