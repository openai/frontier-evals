import email
import imaplib
import logging
import os
import re
import sys
import time
import uuid

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import TimeoutError, expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

EMAIL_USERNAME = "expensifytesting1@gmail.com"
EMAIL_PASSWORD = "redx tbmr hzpc rbzr"
EMAIL_USERNAME_POSTFIX = "42525"
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def generate_user_email() -> str:
    """
    Generate an email address for a user login.
    """
    temp = EMAIL_USERNAME.rsplit("@", 1)
    username = temp[0].strip()
    domain = temp[1].strip()
    return f"{username}+{EMAIL_USERNAME_POSTFIX}@{domain}"


def clear_inbox(username: str, password: str) -> None:
    """
    Delete all the messages from the Inbox.
    """
    LOGGER.info("Clear email inbox")
    with imaplib.IMAP4_SSL(host="imap.gmail.com") as imap:
        imap.login(username, password)
        imap.select("inbox")
        imap.store("1:*", "+FLAGS", "\\Deleted")
        imap.expunge()
        imap.close()


def get_otp_from_email(username: str, password: str, retries: int = 12, delay: int = 5) -> str:
    """
    Read the OTP email and return the OTP code.
    Currently returns a dummy code for faster execution.
    """
    return "123456"  # Dummy implementation for the sake of the test logic.
    # The code below is kept for reference if real email reading is required.
    LOGGER.info("Waiting for the OTP email")
    with imaplib.IMAP4_SSL(host="imap.gmail.com") as imap:
        imap.login(username, password)
        for _ in range(1, retries + 1):
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
                            LOGGER.info("Got the OTP: %s", otp_code)
                            return otp_code
            time.sleep(delay)
        imap.close()
    raise AssertionError("Failed to read the OTP from the email")


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
):
    """
    Launches a browser, signs in (or signs up) a user, and optionally starts tracing.
    Returns the browser, page, and generated user email.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    # Each call to Browser.new_page() creates a fresh context under the hood.
    # We'll capture that context to manage tracing.
    page = browser.new_page()
    context = page.context

    # Install the online guard immediately after creating the context and page
    install_online_guard_sync(context, page)

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    user_email = generate_user_email()
    LOGGER.info("User email: %s", user_email)
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()

    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=5000
        )
    except AssertionError:
        page.get_by_test_id("SignInPage").get_by_role("button", name="Join").click()
    else:
        otp_code = "123456"  # Normally fetched via get_otp_from_email
        page.get_by_test_id("SignInPage").get_by_test_id("validateCode").fill(otp_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
        try:
            page.get_by_test_id("SignInPage").get_by_role("button", name="Sign in").click(
                timeout=2000
            )
        except (AssertionError, TimeoutError):
            pass

    try:
        expect(page.locator("text=What do you want to do today?")).to_be_visible()
    except AssertionError:
        pass
    else:
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()

    return browser, page, user_email


def submit_expense_in_workspace_chat(
    browser: Browser,
    page: Page,
    user_email: str,
    workspace_name: str,
    amount: str = "1000",
) -> None:
    page.get_by_test_id("BaseSidebarScreen").get_by_text(workspace_name, exact=True).last.click()
    page.locator('button[aria-label="Create"]').last.click()
    page.locator('div[aria-label="Submit expense"]').click()
    page.locator('button[aria-label="Manual"]').click()
    page.locator('input[role="presentation"]').fill(amount)
    page.locator('button[data-listener="Enter"]', has_text="Next").first.click()
    page.locator('div[role="menuitem"]', has_text="Merchant").click()
    page.locator('input[aria-label="Merchant"]').fill("Test Merchant")
    page.locator("button", has_text="Save").click()
    page.locator('button[data-listener="Enter"]', has_text="Submit").click()


def test_save_description_in_submitted_expense(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        try:
            browser, page, user_email = login_user(p, trace_enabled=trace_enabled)

            workspace_name = "Workspace 42525"
            page.get_by_role("button", name="My settings").click()
            page.get_by_role("menuitem", name="Workspaces").click()

            existing_workspaces = [
                item.split("\n")[0].strip()
                for item in page.get_by_test_id("WorkspacesListPage")
                .get_by_label("row", exact=True)
                .all_inner_texts()
            ]

            if workspace_name not in existing_workspaces:
                page.get_by_role("button", name="New workspace").first.click()
                page.get_by_test_id("WorkspacePageWithSections").get_by_text(
                    "Name", exact=True
                ).click()
                page.get_by_test_id("WorkspaceNamePage").get_by_role("textbox").fill(workspace_name)
                page.get_by_test_id("WorkspaceNamePage").get_by_role("button", name="Save").click()
                LOGGER.info("Created workspace '%s'", workspace_name)
                page.locator('button[aria-label="Back"]').click()
            else:
                LOGGER.info("Workspace '%s' already exists", workspace_name)

            page.wait_for_timeout(2000)
            page.locator('button[aria-label="Inbox"]').click()

            # Submit two expenses
            submit_expense_in_workspace_chat(browser, page, user_email, workspace_name)
            submit_expense_in_workspace_chat(browser, page, user_email, workspace_name)

            page.wait_for_timeout(2000)
            page.get_by_test_id("BaseSidebarScreen").get_by_text(
                workspace_name, exact=True
            ).last.click()
            page.locator('button[aria-label="View details"]').last.click()
            page.locator('button[aria-label^="Expense Report #"]').last.click()
            page.locator('div[aria-label^="Expense Report #"]').click()

            new_report_title = ""  # Intentionally empty to trigger validation
            page.locator('input[name="text_title"]').fill(new_report_title)
            page.locator('button[data-listener="Enter"]', has_text="Save").first.click()

            error_field = page.locator("text=This field is required.")
            if error_field.is_visible():
                assert True, "Empty Report title cannot be saved."
            else:
                assert False, "Empty Report title has been saved."

        finally:
            # Stop tracing if it was started and clean the trace file
            if trace_enabled:
                try:
                    context = page.context
                    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                    context.tracing.stop(path=trace_path)
                    trace_cleaner(trace_path)
                except Exception as e:
                    LOGGER.error("Error while stopping trace or cleaning it: %s", e)

            # Ensure the browser is closed even if assertions fail
            try:
                browser.close()
            except Exception:
                pass
