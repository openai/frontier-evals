import os
import sys
import uuid

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)
import email
import imaplib
import os
import re
import time
from email.header import decode_header

import pytest
from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

password = "ywvy nytt kjig mscy"
user_email = "jellystarfish99@gmail.com"


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
                        match = re.search("Expensify magic sign-in code: (\\d+)", subject)
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


def login(p, if_phone=False, first_name="John", last_name="Cena"):
    permissions = ["clipboard-read", "clipboard-write"]
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
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
    install_online_guard_sync(context, page)
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()
    phone_or_email_input.fill(user_email)
    continue_button = page.locator('button[tabindex="0"]')
    expect(continue_button).to_be_visible()
    continue_button.click()
    page.wait_for_timeout(5000)
    magic_code = "111111"
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
    return (browser, page)


def test_duplicate_contact_methods(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser, page = login(p, False)
        context = page.context

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page.get_by_label("My settings").click()
            page.get_by_label("jellystarfish99@gmail.com").click()
            page.get_by_role("button", name="New contact method").click()
            page.get_by_role("textbox", name="Email/Phone number").fill("+919920002597")
            page.get_by_role("button", name="Add").click()
            page.wait_for_timeout(5000)
            magic_code1 = "111111"
            page.get_by_test_id("ValidateCodeActionModal").get_by_role("textbox").fill(magic_code1)
            try:
                page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                    timeout=20000
                )
            except Exception:
                pass
            duplicate_added = False
            try:
                page.wait_for_timeout(5000)
                page.get_by_role("button", name="New contact method").click()
                page.get_by_role("textbox", name="Email/Phone number").fill("+919920002597")
                page.get_by_role("button", name="Add").click()
                page.wait_for_timeout(5000)
                magic_code2 = "111111"
                page.get_by_test_id("ValidateCodeActionModal").get_by_role("textbox").fill(
                    magic_code2
                )
                try:
                    page.get_by_role(
                        "button", name=re.compile(r"^(continue|sign.?in)$", re.I)
                    ).click(timeout=20000)
                except Exception:
                    pass
                duplicate_added = True
            except Exception:
                pass
            if duplicate_added:
                page.get_by_test_id("ContactMethodsPage").get_by_label("2597").nth(1).click()
                page.get_by_text("Remove").click()
                page.get_by_role("button", name="Yes, continue").click()
                pytest.fail("Test Failed able to add duplicate phone numbers")
            else:
                page.get_by_test_id("NewContactMethodPage").get_by_label("Back").click()
            page.get_by_test_id("ContactMethodsPage").get_by_label("2597").click()
            page.get_by_text("Remove").click()
            page.get_by_role("button", name="Yes, continue").click()
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()
            browser.close()
