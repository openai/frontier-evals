import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import random
import re
import time

from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
EMAIL = "dev091197@gmail.com"
task_id = 48694
EMAIL1 = f"dev091197+{task_id}@gmail.com"
PASSWORD = "ptul fwbd fvze yzva"


class IMAPOTPExtractor:
    def __init__(self, email_address, password):
        if not self._validate_email(email_address):
            raise ValueError("Invalid email address format.")
        self.original_email = email_address
        self.password = password
        random_suffix = random.randint(1000, 9999)
        current_timestamp = int(time.time())
        random_suffix = f"{current_timestamp}{random_suffix}"

    def _validate_email(self, email_address):
        return bool(re.match("[^@]+@[^@]+\\.[^@]+", email_address))

    def set_new_email(self, email):
        self.new_email = email

    def check_for_otp(self):
        default_otp_code = "123456"
        return default_otp_code

    def _extract_otp_from_email(self, msg):
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

    def _extract_otp(self, text):
        match = re.search("\\b\\d{6}\\b", text)
        return match.group(0) if match else None


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    otp = "123456"
    page.locator('input[data-testid="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    try:
        page.get_by_role("button", name="Sign In").click()
    except Exception:
        pass


def task_check(page):
    random_int_1 = 13
    random_int_2 = 21
    workspace_name = f"WS-{random_int_1}{random_int_2}"
    room_name = f"r-{random_int_1}{random_int_2}"
    page.get_by_label("My settings").click()
    page.wait_for_timeout(2000)
    page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").click()
    page.get_by_role("button", name="New workspace").first.click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("WorkspacePageWithSections").get_by_text("Name").click()
    page.wait_for_timeout(1000)
    page.get_by_role("textbox", name="Name").press("ControlOrMeta+a")
    page.wait_for_timeout(1000)
    page.get_by_role("textbox", name="Name").fill(workspace_name)
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Save").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Back").click()
    page.get_by_label("Inbox").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Start chat (Floating action)").click()
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_label("Room").first.click()
    page.get_by_placeholder("social").fill(room_name)
    page.wait_for_timeout(1000)
    page.get_by_test_id("WorkspaceNewRoomPage").get_by_label("WS-").get_by_text("Workspace").click()
    page.get_by_test_id("ValueSelectorModal").get_by_label(workspace_name).click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Create room").click()
    page.get_by_label(f"#{room_name}").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("ReportDetailsPage").get_by_label("Settings").click()
    page.get_by_text("Notify me about new messages").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list").get_by_label("Daily").click()
    page.get_by_test_id("ReportSettingsPage").get_by_label("Back").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Leave").click()
    page.wait_for_timeout(1000)
    page.locator("#composer").last.click()
    page.locator("#composer").last.fill(f"message-{random_int_1}{random_int_2}")
    page.wait_for_timeout(1000)
    page.get_by_label("Send").last.click()
    page.get_by_label(f"#{room_name}").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("ReportDetailsPage").get_by_label("Settings").click()
    page.wait_for_timeout(2000)
    expect(page.get_by_text("Daily")).to_be_visible()


def test_expensify_0000(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
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

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL1)
            page.wait_for_timeout(2000)
            task_check(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()
            browser.close()
