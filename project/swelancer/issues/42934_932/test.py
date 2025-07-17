import email
import imaplib
import os
import re
import time
import uuid
from email.header import decode_header

from playwright.sync_api import sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

# --------------------------------------------------------------------------- #
# Helper / Utility functions                                                  #
# --------------------------------------------------------------------------- #

USERNAME = "testemail.ameen"
PASSWORD = "yjsp jfez uvmy gkgs"  # Gmail app-password
URL = "https://dev.new.expensify.com:8082/"


def generate_random_email(username: str = USERNAME) -> str:
    """
    Generate a pseudo-random email address.
    """
    timestamp = int(time.time())
    return f"{username}{timestamp}@gmail.com"


def get_magic_code(page, user_email: str, password: str, retries: int = 5, delay: int = 8000):
    """
    Poll the inbox for an Expensify magic code.
    """
    page.wait_for_timeout(delay)
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
            else:
                print("No unread emails found with the subject. Retrying...")
        else:
            print("Failed to retrieve emails. Retrying...")

        page.wait_for_timeout(delay)

    imap.logout()
    print("Max retries reached. Email not found.")
    return None


def click_element(selectors_or_locator, page):
    """
    Click a Playwright locator or selector; supports chained selectors.
    """
    # Handle chained locators (if provided as a list)
    if isinstance(selectors_or_locator, list):
        element = page.locator(selectors_or_locator[0])
        for selector in selectors_or_locator[1:]:
            element = element.locator(selector)
    # Handle single selector string
    elif isinstance(selectors_or_locator, str):
        element = page.locator(selectors_or_locator)
    # Handle pre-located element
    else:
        element = selectors_or_locator

    element.wait_for(state="visible", timeout=10000)
    element.click()


def navigate_with_tabs(page, tab_count: int, enter: bool = True):
    """
    Press <Tab> `tab_count` times and optionally press <Enter>.
    """
    page.wait_for_timeout(1000)
    page.bring_to_front()

    for _ in range(tab_count):
        page.keyboard.press("Tab")
    if enter:
        page.keyboard.press("Enter")


def login_account(page, email_id: str, first_name: str = "Test", last_name: str = "User"):
    """
    Perform the Expensify login flow for a given email address.
    """
    # Step 1: Login ‑ enter email and continue
    page.locator('input[type="email"]').fill(email_id)
    page.locator('button[tabindex="0"]').click(timeout=10000)

    # If OTP is required
    try:
        page.wait_for_selector("text=Please enter the magic code sent to", timeout=5000)
        expensify_otp = get_magic_code(page, email_id, PASSWORD)  # Get OTP
        if expensify_otp:
            page.locator('input[inputmode="numeric"]').fill(expensify_otp)
            try:
                page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                    timeout=20000
                )
            except Exception:
                pass
            page.wait_for_selector('button[tabindex="0"]:has-text("Sign in")', timeout=15000)
    except Exception:
        # Sign-up / Join flow
        try:
            page.wait_for_selector('button[tabindex="0"]:has-text("Join")', timeout=15000)
            page.locator('button[tabindex="0"]:has-text("Join")').click()

            # Additional onboarding screens (best-effort: ignore if not present)
            page.locator("text='Track and budget expenses'").click()
            page.get_by_role("button", name="Continue").click()

            page.locator('input[name="fname"]').wait_for(state="visible", timeout=10000)
            page.locator('input[name="fname"]').fill(first_name)
            page.locator('input[name="lname"]').fill(last_name)
            page.get_by_role("button", name="Continue").click()
        except Exception:
            # If any step is not present, silently continue
            pass


def start_group_chat(page):
    """
    Create/start a new group chat.
    """
    click_element('button[aria-label="Start chat (Floating action)"]', page)
    click_element('div[aria-label="Start chat"][role="menuitem"]', page)

    page.locator('input[aria-label="Name, email, or phone number"]').fill("testuser@gmail.com")
    navigate_with_tabs(page, 2)  # Select first user
    click_element('button:has-text("Next")', page)
    click_element('button:has-text("Start group")', page)


def create_task(page):
    """
    Create a task inside the chat.
    """
    click_element(page.locator('button[aria-label="Create"]').nth(1), page)
    click_element(page.get_by_text("Assign task"), page)

    page.locator('input[aria-label="Title"]').fill("Test Task")
    click_element(page.locator("button", has_text="Next"), page)
    click_element(page.get_by_role("button", name="Confirm task"), page)


# --------------------------------------------------------------------------- #
# Test                                                                        #
# --------------------------------------------------------------------------- #


def test_task_checkbox_center_vertically_with_title(pytestconfig):
    """
    Ensure that the task checkbox is vertically centred/aligned with the title.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    email_id = generate_random_email()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()
        page = context.new_page()
        install_online_guard_sync(context, page)
        page.goto(URL, timeout=60000)

        # ------------------------------------------------------------------- #
        # Tracing start (if enabled)                                          #
        # ------------------------------------------------------------------- #
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            # ---------------------------------------------------------------- #
            # Authentication                                                   #
            # ---------------------------------------------------------------- #
            login_account(page, email_id)

            # ---------------------------------------------------------------- #
            # Chat & Task creation                                             #
            # ---------------------------------------------------------------- #
            start_group_chat(page)
            create_task(page)

            # ---------------------------------------------------------------- #
            # Assertion                                                        #
            # ---------------------------------------------------------------- #
            page.wait_for_timeout(2000)

            task_checkbox_bb = page.get_by_role("checkbox", name="Task").bounding_box()
            task_title_bb = page.get_by_role("button", name="Task").bounding_box()

            checkbox_height = task_checkbox_bb["height"]
            title_height = task_title_bb["height"]

            # Handle potential DOM differences
            if title_height == 24:
                title_height = (
                    page.get_by_role("button", name="Task")
                    .locator("div")
                    .nth(1)
                    .bounding_box()["height"]
                )

            assert checkbox_height == title_height, (
                f"Task title and checkbox heights differ: {checkbox_height} != {title_height}"
            )

        finally:
            # ----------------------------------------------------------------- #
            # Tracing stop / cleanup                                           #
            # ----------------------------------------------------------------- #
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
