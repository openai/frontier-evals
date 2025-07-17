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
from playwright.sync_api._context_manager import PlaywrightContextManager

GMAIL_NAME = "nomandevtur+50163"
GMAIL_APP_PASSWORD = "tpye xlsj axyq nltq"


def get_test_user_info(seed=None, first_name=None, last_name=None):
    if first_name is None:
        first_name = GMAIL_NAME
    if last_name is None:
        last_name = ""
    email_addr = f"{GMAIL_NAME}+{seed}@gmail.com" if seed is not None else f"{GMAIL_NAME}@gmail.com"
    return {
        "email": email_addr,
        "password": GMAIL_APP_PASSWORD,
        "first_name": first_name,
        "last_name": last_name,
    }


def delete_email_inbox(user_email, password, retries=5, delay=10):
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user_email, password)
    imap.select("inbox")
    status, all_emails = imap.search(None, "ALL")
    if status == "OK":
        for email_id in all_emails[0].split():
            imap.store(email_id, "+FLAGS", "\\Deleted")
        imap.expunge()


def get_magic_code(user_email, password, retries=5, delay=5):
    code = "167554"
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
                        if match:
                            code = match.group(1)
                            imap.logout()
                            return code
        time.sleep(delay)
    imap.logout()
    return code


def select_activity(page, first_name, last_name, activity_text):
    expect(page.get_by_text("What do you want to do today?")).to_be_visible()
    page.get_by_label(activity_text).click()
    page.get_by_role("textbox", name="First name").fill(first_name)
    page.get_by_role("textbox", name="Last name").fill(last_name)
    page.get_by_role("button", name="Continue").last.click()


def login_user(page, user_info, activity_text="Track and budget expenses"):
    page.context.clear_cookies()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.wait_for_load_state("load")
    try:
        expect(page.get_by_label("Inbox")).to_be_visible(timeout=10000)
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
        magic_code = "167554"
        validate_code_input.fill(magic_code)
        try:
            page.get_by_role(
                "button",
                name=re.compile(r"^(continue|sign.?in)$", re.I),
            ).click(timeout=20000)
        except Exception:
            pass

    page.wait_for_timeout(3000)
    select_activity_dialog = page.get_by_text("What do you want to do today?")
    if select_activity_dialog.count() > 0:
        select_activity(page, user_info["first_name"], user_info["last_name"], activity_text)


def launch_app(pw: PlaywrightContextManager, headless=True, device=None, geolocation=None):
    browser = pw.chromium.launch(
        headless=headless,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context_args = {"viewport": {"width": 1024, "height": 640}}
    if device:
        context_args.update(pw.devices[device])
    if geolocation:
        context_args["geolocation"] = geolocation
        context_args["permissions"] = ["geolocation"]
    context = browser.new_context(**context_args)
    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard
    return browser, context, page


def test_clear_btn_on_saved_search_name_field(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser, context, page = None, None, None
        try:
            user_info = get_test_user_info()
            browser, context, page = launch_app(p)

            if trace_enabled:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            login_user(page, user_info)
            page.get_by_test_id("CustomBottomTabNavigator").get_by_label("Search").click()
            page.get_by_label("More").click()
            page.get_by_label("Rename").click()
            page.wait_for_timeout(3000)

            clear_button = page.locator("button[aria-label='Clear']")
            assert clear_button.is_visible(), (
                "The button with aria-label 'Clear' is not present on the page."
            )
            print("The button with aria-label 'Clear' is present on the page.")
            page.wait_for_timeout(1000)
        finally:
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                try:
                    context.tracing.stop(path=trace_path)
                except Exception:
                    # In case tracing wasn't started or already stopped
                    pass
                else:
                    trace_cleaner(trace_path)

            # Close resources safely
            try:
                if page is not None:
                    page.close()
            finally:
                if context is not None:
                    context.close()
                if browser is not None:
                    browser.close()
