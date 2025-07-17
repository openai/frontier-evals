Below you will be shown a pytest playwright file that tests certain
functionality of an expensify application.

Your task is to clean up the contents of the file by removing any undesired
comments. Comments are considered undesired if any of the following is true:

1. They are relative to one or more import statements.
2. They are commented out code.
3. They are disabling some linting, formatting or type checking rule.

For a worked example, consider the following input file:

<EXAMPLE_INPUT_1>
import os
import sys
import uuid

import pytest

# Adjust the import path as necessary
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

"""
The test verifies if the offline indicator is correctly shown at the bottom of the screen.
"""
from playwright.async_api import Browser, Page, async_playwright, expect  # noqa: F401
from utils.browser import close_browser
from utils.email_handler import get_specific_derived_email
from utils.login import NEW_DOT_URL, complete_onboarding, sign_up  # noqa: F401

# Email
BASE_EMAIL = "totherandomuser@gmail.com"
PASSWORD = "umeidjwibfmpjntm"  # noqa: F401


# Set up browser
async def start_browser_with_proxy(mobile_device: str):
    playwright = await async_playwright().start()
    device_profile = playwright.devices[mobile_device] if mobile_device else {}
    browser = await playwright.chromium.launch(
        slow_mo=600,
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = await browser.new_context(ignore_https_errors=True, **device_profile)
    page = await context.new_page()
    return context, page, playwright


@pytest.mark.asyncio
async def test_offline_indicator(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    context = page = playwright = None  # Ensure availability in finally
    try:
        context, page, playwright = await start_browser_with_proxy(mobile_device="iPhone 12 Pro")

        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Step 1: Login / Sign up with a new account
        derived_email = get_specific_derived_email(email=BASE_EMAIL, suffix="128937")
        await sign_up(page, derived_email)

        # Complete onboarding if modal is visible
        try:
            await complete_onboarding(page, first_name="Deepak", last_name="Dhaka")
        except Exception:
            # Onboarding modal not present; proceed
            pass

        # Handle unexpected concierge modals
        await page.wait_for_timeout(2000)
        if await page.locator('button:has-text("Get started")').is_visible():
            await page.locator('button:has-text("Get started")').click()
            if await page.get_by_label("Back").is_visible():
                await page.get_by_label("Back").click()

        # Ensure Inbox is visible
        await expect(page.get_by_label("Inbox")).to_be_visible()

        # Step 2: Navigate to Workspace settings
        await page.get_by_label("My settings").click()
        await (
            page.get_by_test_id("InitialSettingsPage")
            .get_by_label("Workspaces")
            .get_by_text("Workspaces")
            .click()
        )
        await page.locator('button:has-text("New workspace")').first.click()
        await page.get_by_text("More features").click()

        # Locate and toggle the switch
        toggle_button = page.get_by_role("switch", name="Gain insights and control over spend")
        is_on = await toggle_button.get_attribute("aria-checked") == "true"
        if not is_on:
            await toggle_button.click()
            print("Button was off, now turned on.")
        else:
            print("Button is already on; clicking twice!")
            await toggle_button.click()
            await toggle_button.click()

        # Step 3: Navigate deeper into settings
        await page.wait_for_timeout(1000)
        await page.get_by_test_id("WorkspaceInitialPage").get_by_text("Expensify Card").click()
        await page.get_by_role("button", name="Issue card").click()
        await page.get_by_test_id("selection-list").get_by_label("Deepak Dhaka").click()

        # Simulate offline mode and verify indicator position
        text_to_find = "You appear to be offline."
        elements = page.locator(f"text={text_to_find}")
        await context.set_offline(True)

        y_pos = []
        count = await elements.count()
        if count > 0:
            for i in range(count):
                bounding_box = await elements.nth(i).bounding_box()
                if bounding_box:
                    print(f"Element {i + 1} position: {bounding_box}")
                    y_pos.append(bounding_box["y"])

        assert 361 not in y_pos, '"You appear to be offline" has been found at an unusual location.'

    finally:
        # Stop tracing and clean the trace if tracing was enabled
        if trace_enabled and context:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Close browser resources
        if context and page and playwright:
            await close_browser(context, page, playwright)
</EXAMPLE_INPUT_1>

Which should be ported to the following output file:

<EXAMPLE_OUTPUT_1>
import os
import sys
import uuid

import pytest

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import Browser, Page, async_playwright, expect
from utils.browser import close_browser
from utils.email_handler import get_specific_derived_email
from utils.login import NEW_DOT_URL, complete_onboarding, sign_up

# Email
BASE_EMAIL = "totherandomuser@gmail.com"
PASSWORD = "umeidjwibfmpjntm"


# Set up browser
async def start_browser_with_proxy(mobile_device: str):
    playwright = await async_playwright().start()
    device_profile = playwright.devices[mobile_device] if mobile_device else {}
    browser = await playwright.chromium.launch(
        slow_mo=600,
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = await browser.new_context(ignore_https_errors=True, **device_profile)
    page = await context.new_page()
    return context, page, playwright


@pytest.mark.asyncio
async def test_offline_indicator(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    context = page = playwright = None  # Ensure availability in finally
    try:
        context, page, playwright = await start_browser_with_proxy(mobile_device="iPhone 12 Pro")

        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Step 1: Login / Sign up with a new account
        derived_email = get_specific_derived_email(email=BASE_EMAIL, suffix="128937")
        await sign_up(page, derived_email)

        # Complete onboarding if modal is visible
        try:
            await complete_onboarding(page, first_name="Deepak", last_name="Dhaka")
        except Exception:
            # Onboarding modal not present; proceed
            pass

        # Handle unexpected concierge modals
        await page.wait_for_timeout(2000)
        if await page.locator('button:has-text("Get started")').is_visible():
            await page.locator('button:has-text("Get started")').click()
            if await page.get_by_label("Back").is_visible():
                await page.get_by_label("Back").click()

        # Ensure Inbox is visible
        await expect(page.get_by_label("Inbox")).to_be_visible()

        # Step 2: Navigate to Workspace settings
        await page.get_by_label("My settings").click()
        await (
            page.get_by_test_id("InitialSettingsPage")
            .get_by_label("Workspaces")
            .get_by_text("Workspaces")
            .click()
        )
        await page.locator('button:has-text("New workspace")').first.click()
        await page.get_by_text("More features").click()

        # Locate and toggle the switch
        toggle_button = page.get_by_role("switch", name="Gain insights and control over spend")
        is_on = await toggle_button.get_attribute("aria-checked") == "true"
        if not is_on:
            await toggle_button.click()
            print("Button was off, now turned on.")
        else:
            print("Button is already on; clicking twice!")
            await toggle_button.click()
            await toggle_button.click()

        # Step 3: Navigate deeper into settings
        await page.wait_for_timeout(1000)
        await page.get_by_test_id("WorkspaceInitialPage").get_by_text("Expensify Card").click()
        await page.get_by_role("button", name="Issue card").click()
        await page.get_by_test_id("selection-list").get_by_label("Deepak Dhaka").click()

        # Simulate offline mode and verify indicator position
        text_to_find = "You appear to be offline."
        elements = page.locator(f"text={text_to_find}")
        await context.set_offline(True)

        y_pos = []
        count = await elements.count()
        if count > 0:
            for i in range(count):
                bounding_box = await elements.nth(i).bounding_box()
                if bounding_box:
                    print(f"Element {i + 1} position: {bounding_box}")
                    y_pos.append(bounding_box["y"])

        assert 361 not in y_pos, '"You appear to be offline" has been found at an unusual location.'

    finally:
        # Stop tracing and clean the trace if tracing was enabled
        if trace_enabled and context:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Close browser resources
        if context and page and playwright:
            await close_browser(context, page, playwright)
</EXAMPLE_OUTPUT_1>

And another example

<EXAMPLE_INPUT_2>
import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner  # adjust the import path if necessary

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import re

from playwright.sync_api import expect, sync_playwright

EMAIL = "velmoner+960@gmail.com"
PASSWORD = "aory ezrb qkmi qkas"
NEWDOT_URL = "https://dev.new.expensify.com:8082/"


class OTPUtil:
    @staticmethod
    def _parse_otp(text) -> str | None:
        match = re.search(r"\b\d{6}\b", text)
        return match.group(0) if match else None

    @staticmethod
    def _fetch_otp(msg) -> str:
        otp = None
        if not msg.is_multipart():
            body = msg.get_payload(decode=True).decode()
            otp = OTPUtil._parse_otp(body)
            if otp:
                return otp

        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                _body = part.get_payload(decode=True).decode()
                otp = OTPUtil._parse_otp(_body)
        if otp:
            return otp

    @staticmethod
    def get_otp(email_id: str, password: str) -> str:
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(email_id, password)
            mail.select("inbox")

            status, messages = mail.search(None, "UNSEEN")
            if status != "OK":
                return None

            emails = messages[0].split()[::-1]

            for id in emails:
                status, msg_data = mail.fetch(id, "(RFC822)")
                if status != "OK":
                    continue

                for resp in msg_data:
                    if isinstance(resp, tuple):
                        msg = email.message_from_bytes(resp[1])

                        if email_id in msg["To"]:
                            otp_code = OTPUtil._fetch_otp(msg)
                            if otp_code:
                                mail.store(id, "+FLAGS", "\\Deleted")
                                mail.expunge()
                                mail.close()
                                mail.logout()
                                return otp_code

            mail.close()
            mail.logout()
            return None

        except imaplib.IMAP4.error:
            return None


def user_login(page: str, email_id: str):
    page.goto(NEWDOT_URL)

    page.locator('input[type="email"]').fill(email_id)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)

    # otp = OTPUtil.get_otp(email_id=EMAIL, password=PASSWORD)
    otp = "123456"
    page.locator('input[data-testid="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name="Sign In").click()
    except Exception:
        # If the button isn't present or another error occurs, continue silently
        pass


def test_payment_text(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()

            user_login(page=page, email_id=EMAIL)

            # Step 2: create task to assign
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_label("Assign task").click()

            # Step 3: Enter the test task
            page.get_by_role("textbox", name="Title").fill("test task")
            page.get_by_role("textbox", name="Title").press("Enter")
            page.wait_for_timeout(2000)

            # Step 4: Select assignee and check if the test assign to me is visible
            page.get_by_role("menuitem", name="Assignee").click()
            page.wait_for_timeout(2000)
            expect(page.get_by_text("Assign to me")).to_be_visible()

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            browser.close()
</EXAMPLE_INPUT_2>

Which should be ported to the following output file:

<EXAMPLE_OUTPUT_2>
import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import re

from playwright.sync_api import expect, sync_playwright

EMAIL = "velmoner+960@gmail.com"
PASSWORD = "aory ezrb qkmi qkas"
NEWDOT_URL = "https://dev.new.expensify.com:8082/"


class OTPUtil:
    @staticmethod
    def _parse_otp(text) -> str | None:
        match = re.search(r"\b\d{6}\b", text)
        return match.group(0) if match else None

    @staticmethod
    def _fetch_otp(msg) -> str:
        otp = None
        if not msg.is_multipart():
            body = msg.get_payload(decode=True).decode()
            otp = OTPUtil._parse_otp(body)
            if otp:
                return otp

        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                _body = part.get_payload(decode=True).decode()
                otp = OTPUtil._parse_otp(_body)
        if otp:
            return otp

    @staticmethod
    def get_otp(email_id: str, password: str) -> str:
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(email_id, password)
            mail.select("inbox")

            status, messages = mail.search(None, "UNSEEN")
            if status != "OK":
                return None

            emails = messages[0].split()[::-1]

            for id in emails:
                status, msg_data = mail.fetch(id, "(RFC822)")
                if status != "OK":
                    continue

                for resp in msg_data:
                    if isinstance(resp, tuple):
                        msg = email.message_from_bytes(resp[1])

                        if email_id in msg["To"]:
                            otp_code = OTPUtil._fetch_otp(msg)
                            if otp_code:
                                mail.store(id, "+FLAGS", "\\Deleted")
                                mail.expunge()
                                mail.close()
                                mail.logout()
                                return otp_code

            mail.close()
            mail.logout()
            return None

        except imaplib.IMAP4.error:
            return None


def user_login(page: str, email_id: str):
    page.goto(NEWDOT_URL)

    page.locator('input[type="email"]').fill(email_id)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)

    otp = "123456"
    page.locator('input[data-testid="validateCode"]').fill(otp)
    try:
        page.get_by_role("button", name="Sign In").click()
    except Exception:
        # If the button isn't present or another error occurs, continue silently
        pass


def test_payment_text(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()

            user_login(page=page, email_id=EMAIL)

            # Step 2: create task to assign
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_label("Assign task").click()

            # Step 3: Enter the test task
            page.get_by_role("textbox", name="Title").fill("test task")
            page.get_by_role("textbox", name="Title").press("Enter")
            page.wait_for_timeout(2000)

            # Step 4: Select assignee and check if the test assign to me is visible
            page.get_by_role("menuitem", name="Assignee").click()
            page.wait_for_timeout(2000)
            expect(page.get_by_text("Assign to me")).to_be_visible()

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            browser.close()
</EXAMPLE_OUTPUT_2>

As you can see, we have removed the comments related to import statements,
commented out code, and any comments that disable linting, formatting, or type
checking rules. Other comments are maintained.

Now, based on the instructions and examples above, you should port the following
test file:

<INPUT>
§input_file
</INPUT>

You should wrap the code in your response in <OUTPUT></OUTPUT> tags. If you find
that no porting is needed, you should respond with <NO-OP></NO-OP> tags,
containing an explanation of why the porting is not possible.

Note: You absolutely must not change anything other than the comments as
instructed. Functionality must remain identical to the original file. Do not
add more comments or veer from the instructions provided above.

Please proceed.
