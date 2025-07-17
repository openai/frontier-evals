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
from datetime import datetime, timezone
from email.header import decode_header

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

EMAIL_USERNAME = "aliceoglualice@gmail.com"
EMAIL_PASSWORD = "cfpy ofxm lcmq aeta"
EMAIL_USERNAME_POSTFIX = "29900.178212611614"


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


def generate_user_email(user_id: int) -> str:
    """
    Generate an email address for a user login.
    """
    temp = EMAIL_USERNAME.rsplit("@", 1)
    username = temp[0].strip()
    domain = temp[1].strip()
    return f"{username}+{EMAIL_USERNAME_POSTFIX}{user_id}@{domain}"


def login_user(
    p: PlaywrightContextManager,
    first_name: str,
    last_name: str,
    user_id: int,
    trace_enabled: bool = False,
):
    """
    Log a user into Expensify and return the browser, context, page and
    generated email used for the login.
    """
    browser: Browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--ignore-certificate-errors",
        ],
    )
    context = browser.new_context(ignore_https_errors=True)

    # Start tracing if required
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = generate_user_email(user_id)
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.get_by_test_id("username").fill(user_email)
    page.get_by_role("button", name="Continue").click()

    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=5000
        )
    except (AssertionError, TimeoutError):
        # New account – press "Join"
        page.get_by_test_id("SignInPage").get_by_role("button", name="Join").click()
    else:
        # Existing account – use magic code
        otp_code = get_magic_code(EMAIL_USERNAME, EMAIL_PASSWORD)
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
        expect(page.get_by_text("What do you want to do today?")).to_be_visible(timeout=5000)
    except (AssertionError, TimeoutError):
        pass
    else:
        page.get_by_label("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()

    # Return everything needed to close / stop trace later
    return browser, context, page, user_email


def test_invite_member_name_visibility_on_hover(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    # Initialise variables so they exist in the finally block even if setup fails
    browser = browser2 = None
    context = context2 = None
    with sync_playwright() as p:
        try:
            # Login first user (Bob)
            browser2, context2, page2, user_email2 = login_user(
                p, "Bob", "B", 2, trace_enabled=trace_enabled
            )

            # Login second user (Alice)
            browser, context, page, user_email = login_user(
                p, "Alice", "A", 1, trace_enabled=trace_enabled
            )

            # Begin original test steps
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_text("Start chat").click()
            page.get_by_test_id("selection-list-text-input").fill(user_email2)
            page.wait_for_timeout(1000)
            page.locator(f'div[dir="auto"]:has-text("{user_email2}")').last.click()
            page.get_by_role("textbox").fill("Hello")
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_text("New workspace").click()
            page.get_by_text("Members").click()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Invite member").click()
            page.locator('input[data-testid="selection-list-text-input"]').fill(user_email2)
            page.locator(f'div[dir="auto"]:has-text("{user_email2}")').last.click()
            next_button = page.get_by_role("button", name="Next")
            next_button.click()
            page.get_by_test_id("WorkspaceInviteMessagePage").get_by_role(
                "button", name="Invite"
            ).click()
            page.wait_for_timeout(1000)
            avatar_count = page.locator('[data-testid="SvgFallbackAvatar Icon"]').count()
            for i in range(avatar_count):
                avatar = page.locator('[data-testid="SvgFallbackAvatar Icon"]').nth(i)
                if avatar.is_visible() and avatar.is_enabled():
                    try:
                        avatar.hover(timeout=500)
                        print(f"Hovered over avatar {i}")
                    except Exception as e:
                        print(f"Failed to hover over avatar {i}: {e}")
                page.wait_for_timeout(100)
            hover_container = page.locator('div.css-text-146c3p1:has-text("Bob B")').nth(3)
            hover_container.wait_for(timeout=3000)
            assert hover_container.is_visible(), "Hover element containing 'Bob B' is not visible"
        finally:
            # Stop tracing and clean traces
            if trace_enabled:
                for ctx in [context2, context]:
                    if ctx is not None:
                        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                        try:
                            ctx.tracing.stop(path=trace_path)
                        except Exception as e:
                            # Tracing might already be stopped if context/browsers crashed.
                            print(f"Failed to stop tracing: {e}")
                        else:
                            trace_cleaner(trace_path)

            # Close browsers if they were created
            for br in [browser2, browser]:
                try:
                    if br is not None:
                        br.close()
                except Exception as e:
                    print(f"Failed to close browser: {e}")
