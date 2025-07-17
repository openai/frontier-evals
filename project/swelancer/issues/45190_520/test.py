import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import io
import re
import time
from email.header import decode_header

import numpy as np
from PIL import Image
from playwright.sync_api import expect, sync_playwright

GMAIL_NAME = "testerbro7077"
GMAIL_APP_PASSWORD = "xqxz mmcb tvkn lpgp"


def get_test_user_info(seed=None, first_name=None, last_name=None):
    if first_name is None:
        first_name = "Yagan"
    if last_name is None:
        last_name = "Sai"
    email_addr = f"{GMAIL_NAME}+{seed}@gmail.com" if seed is not None else f"{GMAIL_NAME}@gmail.com"
    return {
        "email": email_addr,
        "password": GMAIL_APP_PASSWORD,
        "first_name": first_name,
        "last_name": last_name,
    }


def get_magic_code(user_email, password, retries=5, delay=5):
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user_email, password)
    for _ in range(retries):
        imap.select("inbox")
        status, messages = imap.search(None, '(UNSEEN SUBJECT "Expensify magic sign-in code:")')
        if status == "OK":
            email_ids = messages[0].split()
            print(email_ids)
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


def select_activity(page, first_name, last_name, activity_text):
    expect(page.get_by_text("What do you want to do today?")).to_be_visible()
    page.get_by_label(activity_text).click()
    page.get_by_role("textbox", name="First name").fill(first_name)
    page.get_by_role("textbox", name="Last name").fill(last_name)
    page.get_by_role("button", name="Continue").last.click()


def login_user(page, user_info, activity_text="Track and budget expenses"):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.wait_for_load_state("load")
    try:
        expect(page.get_by_label("Inbox")).to_be_visible(timeout=3000)
        return
    except Exception:
        pass
    page.get_by_test_id("username").fill(user_info["email"])
    page.get_by_role("button", name="Continue").click()
    join_button = page.get_by_role("button", name="Join")
    validate_code_input = page.locator('input[data-testid="validateCode"]')
    expect(join_button.or_(validate_code_input)).to_be_visible()
    if join_button.is_visible():
        join_button.click(timeout=3000)
    else:
        magic_code = "123456"
        print(f"Magic code: {magic_code}")
        validate_code_input.fill(magic_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
    page.wait_for_timeout(3000)
    select_activity_dialog = page.get_by_text("What do you want to do today?")
    if select_activity_dialog.count() > 0:
        select_activity(page, user_info["first_name"], user_info["last_name"], activity_text)


def chech_bg_color(page):
    page.get_by_label("My settings").click()
    page.wait_for_timeout(1000)
    page.get_by_text("Preferences").click()
    page.wait_for_timeout(1000)
    if not page.get_by_text("Dark").is_visible():
        page.get_by_text("Theme").click()
        page.get_by_label("Dark").click()
    page.get_by_test_id("CustomBottomTabNavigator").get_by_label("Search").click()
    page.get_by_role("button", name="Outstanding").click()
    checkbox = page.get_by_role("checkbox").nth(3)
    aria_checked = checkbox.get_attribute("aria-checked")
    page.wait_for_timeout(500)
    if aria_checked == "false":
        checkbox.click()
    else:
        checkbox.click()
        page.wait_for_timeout(500)
        checkbox.click()
    page.wait_for_timeout(1000)
    page.mouse.click(0.5, 0.5)
    page.wait_for_timeout(1000)
    view_button = page.get_by_role("button", name="View", exact=True).nth(2)
    container = view_button.locator("..").locator("..").locator("..").locator("..")
    background_color = container.evaluate("element => getComputedStyle(element).backgroundColor")
    expected_background_color = "rgb(26, 61, 50)"
    assert background_color == expected_background_color, (
        "Expected background UI change did not occur"
    )


def _ensure_dir(path: str):
    """Utility to create directories safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _export_and_clean_trace(context, enabled: bool):
    """
    Helper to save playwright trace for a given context and clean it.
    """
    if not enabled:
        return
    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    _ensure_dir(trace_path)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_login(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    browser = None
    contextA = None
    pageA = None

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--ignore-certificate-errors",
                ],
            )

            userA_info = get_test_user_info(seed=1, first_name="Yagna")
            contextA = browser.new_context()

            if trace_enabled:
                contextA.tracing.start(screenshots=True, snapshots=True, sources=True)

            pageA = contextA.new_page()
            install_online_guard_sync(contextA, pageA)
            login_user(pageA, userA_info)
            chech_bg_color(pageA)

        finally:
            # Export and clean traces if needed
            if contextA is not None:
                _export_and_clean_trace(contextA, trace_enabled)
            # Cleanup playwright resources
            if pageA is not None:
                try:
                    pageA.close()
                except Exception:
                    pass
            if contextA is not None:
                try:
                    contextA.close()
                except Exception:
                    pass
            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    pass
