import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import random
import re
import string
from email.header import decode_header
from imaplib import IMAP4_SSL

from playwright.sync_api import Browser, Page, expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

URL = "https://dev.new.expensify.com:8082/"
IMAP_SERVER = "imap.gmail.com"
MAX_TRIES = 2
USERNAME = "doorsqueaky@gmail.com"
PASSWORD = "eabcahqraatorpvt"


def login_to_email(email: str = USERNAME, password: str = PASSWORD):
    imap = IMAP4_SSL(IMAP_SERVER)
    imap.LOGIN(email, password)
    return imap


def logout_email(imap: IMAP4_SSL):
    imap.logout()


def clear_inbox(imap: IMAP4_SSL):
    imap.select("inbox")
    status, messages = imap.search(None, "ALL")
    email_ids = messages[0].split()
    for email_id in email_ids:
        imap.store(email_id, "+FLAGS", "\\Deleted")
    imap.expunge()


def generate_random_string(length: int) -> str:
    random_string = "".join(random.choice(string.ascii_letters) for _ in range(length))
    return random_string


def get_otp(imap: IMAP4_SSL):
    imap.select("inbox")
    status, messages = imap.search(None, "ALL")
    email_ids = messages[0].split()
    for email_id in email_ids:
        status, msg_data = imap.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")
                r = re.search("Expensify magic sign-in code:.*", subject, re.IGNORECASE)
                if not r:
                    return None
                temp = subject.split(":")
                otp_code = temp[-1].strip()
                return otp_code
    return None


def launch_app(
    pw: PlaywrightContextManager, headless: bool = False
) -> tuple[Browser, BrowserContext, Page]:
    """
    Launch the Expensify app and return browser, context, and page.
    """
    browser = pw.chromium.launch(
        headless=headless,
        args=["--disable-web-security", "--disable-features=IsolateOrigins,site-per-process"],
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)
    return browser, context, page


def login(page: Page, imap: IMAP4_SSL, email: str = USERNAME):
    """
    Login to the Expensify app and complete the onboarding.
    """
    page.goto(URL, timeout=60000)
    page.get_by_role("textbox", name="Phone or email").fill(email)
    page.locator("div:nth-child(3) > div:nth-child(2) > div > div").first.click()
    otp = "123456"
    page.locator('input[name="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    try:
        page.locator("div").filter(has_text=re.compile("^Sign in$")).nth(1).click(timeout=15000)
    except Exception:
        pass


def cleanup(browser: Browser, page: Page) -> None:
    page.wait_for_timeout(1000)
    browser.close()


def create_workspace(page: Page) -> str:
    page.locator('button[aria-label="My settings"]').click()
    page.locator('div[aria-label="Workspaces"][role="menuitem"]').click()
    page.locator('button[aria-label="New workspace"]').first.click()
    page.locator('div[role="menuitem"]', has_text="Name").last.click()
    input_field = page.locator('input[aria-label="Name"]')
    input_field.clear()
    workspace_name = generate_random_string(3) + " Workspace"
    input_field.fill(workspace_name)
    page.get_by_role("button", name="Save").click()
    page.locator('button[aria-label="Back"]').click()
    page.locator('button[aria-label="Inbox"]').click()
    return workspace_name


def switch_workspace(page: Page, workspace_name: str):
    page.locator('button[aria-label="Workspaces"]').click()
    page.locator(f'button[aria-label="{workspace_name}"]').last.click()
    title_inbox = page.get_by_text("Inbox").first
    inbox_parent = title_inbox.locator("..")
    title_workspace = inbox_parent.filter(has_text=f"{workspace_name}")
    expect(title_workspace).to_be_visible()


def check_title_post_reload(page: Page, workspace_name: str):
    page.reload()
    title_inbox = page.get_by_text("Inbox").first
    inbox_parent = title_inbox.locator("..")
    title_workspace = inbox_parent.filter(has_text=f"{workspace_name}")
    expect(title_workspace).to_be_visible()


def perform_login_steps(page: Page):
    """
    Originally inline in the test; extracted for reuse in try block.
    """
    page.goto(URL, timeout=60000)
    page.get_by_role("textbox", name="Phone or email").fill(USERNAME)
    page.locator("div:nth-child(3) > div:nth-child(2) > div > div").first.click()
    otp = "123456"
    page.locator('input[name="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    try:
        page.locator("div").filter(has_text=re.compile("^Sign in$")).nth(1).click(timeout=15000)
    except Exception:
        pass


def _start_tracing_if_enabled(trace_enabled: bool, context: BrowserContext):
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_tracing_if_enabled(trace_enabled: bool, context: BrowserContext):
    if trace_enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_workspace_switch_on_reload(pytestconfig):
    """
    Demonstrate the Expensify app email login verification.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as pw:
        browser, context, page = launch_app(pw)
        _start_tracing_if_enabled(trace_enabled, context)

        try:
            perform_login_steps(page)
            workspace_name = create_workspace(page)
            switch_workspace(page, workspace_name)
            check_title_post_reload(page, workspace_name)
        finally:
            _stop_tracing_if_enabled(trace_enabled, context)
            cleanup(browser, page)
