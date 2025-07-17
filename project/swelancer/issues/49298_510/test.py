import email
import imaplib
import logging
import os
import re
import sys
import time
import uuid

from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

EXPENSIFY_URL = "https://dev.new.expensify.com:8082/"
EMAIL_USERNAME = "naturesv057@gmail.com"
EMAIL_PASSWORD = "hyjk ilxi pnom oret"
EMAIL_USERNAME_POSTFIX = "49298_4"

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def generate_user_email(user_id=None):
    """
    Generate an email address for a user login.
    """
    temp = EMAIL_USERNAME.rsplit("@", 1)
    username = temp[0].strip()
    domain = temp[1].strip()
    return f"{username}+{EMAIL_USERNAME_POSTFIX}@{domain}"


def clear_inbox(username, password):
    """
    Delete all existing messages from the Inbox.
    """
    LOGGER.info("Deleting all existing messages from the email inbox")
    with imaplib.IMAP4_SSL(host="imap.gmail.com") as imap:
        imap.login(username, password)
        imap.select("inbox")
        imap.store("1:*", "+FLAGS", "\\Deleted")
        imap.expunge()
        imap.close()


def get_otp_from_email(username, password, retries=2, delay=2):
    """
    Read the OTP email and return the OTP code.
    """
    LOGGER.info("Reading the OTP email")
    with imaplib.IMAP4_SSL(host="imap.gmail.com") as imap:
        imap.login(username, password)
        for _i in range(1, retries + 1):
            imap.select("inbox")
            status, messages = imap.search(None, "ALL")
            if status == "OK":
                for message_id in reversed(messages[0].split()):
                    status, data = imap.fetch(message_id, "(RFC822)")
                    if status == "OK":
                        email_message = email.message_from_bytes(data[0][1])
                        subject, encoding = email.header.decode_header(email_message["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding)
                        if subject.startswith("Expensify magic sign-in code:"):
                            otp_code = subject.split(":")[-1].strip()
                            LOGGER.info("Got the OTP %s", otp_code)
                            return otp_code
            time.sleep(delay)
        imap.close()
    return "123456"


def launch_browser(pw, headless=True, device=None, geolocation=None):
    """
    Launch the browser.
    """
    browser = pw.chromium.launch(
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context_args = {}
    if device:
        context_args.update(pw.devices[device])
    if geolocation:
        context_args["geolocation"] = geolocation
        context_args["permissions"] = ["geolocation"]
    context = browser.new_context(**context_args)
    page = context.new_page()
    install_online_guard_sync(context, page)
    return browser, context, page


def login_user(page: Page, email, first_name="John", last_name="Doe"):
    """
    Login to the Expensify app and complete the onboarding.
    """
    page.goto(EXPENSIFY_URL, timeout=60000)
    page.get_by_test_id("username").fill(email)
    page.get_by_role("button", name="Continue").click()
    otp_code = "123456"
    page.get_by_test_id("validateCode").fill(otp_code)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


def _start_tracing_if_enabled(trace_enabled: bool, context):
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_tracing_if_enabled(trace_enabled: bool, context):
    """
    Stop tracing, save the trace file, and clean it using trace_cleaner.
    """
    if trace_enabled and context is not None:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_private_notes_scroll(pytestconfig):
    """
    Verify that the introductory text on the private notes list scrolls
    with the page rather than staying fixed.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = None
        context = None
        try:
            user_email = generate_user_email()
            LOGGER.info("User email %s", user_email)

            browser, context, page = launch_browser(p)
            _start_tracing_if_enabled(trace_enabled, context)

            login_user(page, user_email)
            page.wait_for_timeout(5000)
            LOGGER.info("User logged in successfully")

            # Start a chat
            page.locator('button[aria-label="Start chat (Floating action)"][tabindex="0"]').click()
            page.locator('div[aria-label="Start chat"][role="menuitem"]').click()
            page.locator('input[aria-label="Name, email, or phone number"]').fill(
                "t56826807+13@gmail.com"
            )

            # Try selecting the user
            try:
                page.locator('button[aria-label="Employee"]').last.click(timeout=15000)
            except Exception:
                page.get_by_label("t56826807+13@gmail.com").click()
            page.locator('button[aria-label="Employee"]').last.click()

            # Open private notes and add many lines to enable scrolling
            page.locator('div[aria-label="Private notes"]').last.click()
            page.locator('div[aria-label="Private notes"][role="textbox"]').fill("new line\n" * 100)
            page.locator('button[data-listener="Enter"]', has_text="Save").last.click()

            # Re-open private notes list
            page.locator('div[aria-label="Private notes"]').last.click()
            page.wait_for_selector('div[data-testid="PrivateNotesListPage"]')

            parent_div = page.locator('div[data-testid="PrivateNotesListPage"]')
            child_with_scroll_y = parent_div.evaluate_handle(
                """
                parent => {
                    const children = parent.querySelectorAll("div");
                    return Array.from(children).find(child => {
                        const style = getComputedStyle(child);
                        return style.overflowY === "scroll" || style.overflowY === "auto";
                    });
                }
                """
            )

            if child_with_scroll_y:
                text_content = child_with_scroll_y.evaluate("element => element.innerText")
            else:
                raise Exception("No child div with overflow-y set to scroll or auto found.")

            assert (
                "Keep notes about this chat here. You're the only person who can add, edit, or view these notes."
                in text_content
            ), (
                "The intro text on private notes list stays fixed when it should scroll with the page"
            )

        finally:
            _stop_tracing_if_enabled(trace_enabled, context)
            if context:
                context.close()
            if browser:
                browser.close()
