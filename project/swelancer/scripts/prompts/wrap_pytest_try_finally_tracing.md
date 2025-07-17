Below you will be shown a pytest playwright file that tests certain
functionality of an expensify application.

Your task is to port this file such that the tests are wrapped in a try/finally
block. Additionally, in your port you should ensure that, if the
`--user-tool-trace` option is set to `True`, the following is true:

1. any browser context creation is followed by the start of a tracing session,
   via `context.tracing.start(screenshots=True, snapshots=True, sources=True)`
2. In the `finally` block, `context.tracing.stop()` is called for any context in
   the test, exporting the trace(s) to `/app/expensify/user_tool/output_<unique>.zip`
3. Still in the finally block, the `trace_cleaner(trace_path)` function is
   called on the path to the trace.zip file(s). This function can be imported
   with `from utils.trace_cleaner import trace_cleaner`.

If the `--user-tool-trace` option has not been set, the tracing should not
occur, but the try/finally block will still be there (and supposedly will not
really do anything; we do not wish to catch exceptions, the tests should behave
as normal).

You should assume that the `--user-tool-trace` option is a boolean that has been
configured in `conftest.py`.

For a worked example of a port, consider the following input file:

<EXAMPLE_INPUT_1>
import sys, os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)
from playwright.sync_api import sync_playwright, expect
import re, time
import os
import imaplib
import email
import re
import random
import string

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


def new_dot_login(page, email_id):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email_id)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    page.locator('input[data-testid="validateCode"]').fill("1234556")


def emoji_modal(page):
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").fill("tosomenewuser@gmail.com")
    page.wait_for_timeout(1000)
    page.get_by_label("tosomenewuser@gmail.com").click()
    page.wait_for_timeout(1000)
    composer_locator = page.locator('[contenteditable="true"]').last
    page.wait_for_timeout(1000)
    composer_locator.click()
    composer_locator.type(":smile:")
    page.wait_for_timeout(1000)
    expect(composer_locator.get_by_text(":smile:")).not_to_be_visible()
    expect(composer_locator.get_by_text("😄")).to_be_visible()


def test_expensify_0000():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            slow_mo=1000,
            channel="chromium",
        )
        context = browser.new_context()
        page = context.new_page()
        page.wait_for_timeout(2000)
        new_dot_login(page, "pehlajrajput1@gmail.com")
        page.wait_for_timeout(2000)
        emoji_modal(page)
        browser.close()
</EXAMPLE_INPUT_1>

Which should be ported to the following output file (or similar):

<EXAMPLE_OUTPUT_1>
import sys, os
import uuid
import pytest
from trace_cleaner import trace_cleaner  # adjust import path as needed

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)
from playwright.sync_api import sync_playwright, expect
import re, time
import os
import imaplib
import email
import re
import random
import string

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


def new_dot_login(page, email_id):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email_id)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    page.locator('input[data-testid="validateCode"]').fill("1234556")


def emoji_modal(page):
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").fill("tosomenewuser@gmail.com")
    page.wait_for_timeout(1000)
    page.get_by_label("tosomenewuser@gmail.com").click()
    page.wait_for_timeout(1000)
    composer_locator = page.locator('[contenteditable="true"]').last
    page.wait_for_timeout(1000)
    composer_locator.click()
    composer_locator.type(":smile:")
    page.wait_for_timeout(1000)
    expect(composer_locator.get_by_text(":smile:")).not_to_be_visible()
    expect(composer_locator.get_by_text("😄")).to_be_visible()


def test_expensify_0000(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            slow_mo=1000,
            channel="chromium",
        )
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
        try:
            page = context.new_page()
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL)
            page.wait_for_timeout(2000)
            emoji_modal(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            browser.close()
</EXAMPLE_OUTPUT_1>

Note that we kept the exact same functionality of the original test. It is
important that your porting concerns itself on converting the test into a
fixture-based format, and does not change the functionality of the test itself,
with the overall goal of enabling tracing.

Here we show an additional example:

<EXAMPLE_INPUT_2>
import sys, os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import pytest
from playwright.async_api import expect, async_playwright, Browser, Page
from utils.login import sign_in, NEW_DOT_URL, check_if_logged_in
from utils.browser import close_browser


USER_EMAILA = "doorsqueaky+3182311@gmail.com"
USERNAME_A = "John Doe"
PASSWORD_A = "eabcahqraatorpvt"
USER_EMAILB = "windowsqueaky+3919211@gmail.com"
PASSWORD_B = "jeaqzqklzxxdetsh"
USERNAME_B = "Jane Doe"


async def start_browser_with_proxy():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        slow_mo=500,
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = await browser.new_context(ignore_https_errors=True)
    page = await context.new_page()
    return context, page, playwright


async def send_message(page: Page, recepient_email: str, message: str):
    # step 1: click on floating action button and start a chat
    await page.locator('button[aria-label="Start chat (Floating action)"]').click()
    await page.locator('div[aria-label="Start chat"][role="menuitem"]').click()

    # step 2: enter user email and start chat
    await page.locator('input[aria-label="Name, email, or phone number"]').fill(
        recepient_email
    )
    await (
        page.get_by_test_id("selection-list")
        .locator(f'button:has-text("{recepient_email}")')
        .click()
    )
    await page.wait_for_timeout(1000)

    # step 3: Write the message and send
    message_editor = page.locator(
        'div[contenteditable="true"][placeholder="Write something..."]'
    ).last
    await message_editor.fill(message)
    await message_editor.press("Enter")


async def create_draft_reply_in_thread(
    page: Page, username: str, sender_email: str, message: str, reply: str
):
    # step 1: Navigate to the chat
    if await page.locator(
        'button[aria-label="Navigates to a chat"]', has_text=sender_email
    ).is_visible():
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=sender_email
        ).click()
    else:
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=username
        ).click()

    # step 2: Recognize the message and start reply in thread
    await page.locator('div[aria-label="Chat message"]', has_text=message).click(
        button="right"
    )
    await page.locator('div[aria-label="Reply in thread"][role="menuitem"]').click()

    await page.wait_for_timeout(1000)

    # step 3: Write the message and send
    message_editor = page.locator(
        'div[contenteditable="true"][placeholder="Write something..."]'
    ).last
    await message_editor.fill(reply)


async def navigate_away_check_draft(page: Page, username: str, sender_email: str):
    # step 1: navigate to main chat away from reply thread
    if await page.locator(
        'button[aria-label="Navigates to a chat"]', has_text=sender_email
    ).is_visible():
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=sender_email
        ).click()
    else:
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=username
        ).click()

    # few seconds of timeout before final assertion
    await page.wait_for_timeout(2000)

    # step 2: check draft report exists
    draft_reply_LHN_btn = page.locator(
        'button[aria-label="Navigates to a chat"]', has_text="No activity yet"
    )
    await expect(draft_reply_LHN_btn).to_be_visible()


async def enter_dummy_otp_if_not_logged_in(page: Page, email: str):
    if not await check_if_logged_in(page=page, url=NEW_DOT_URL):
        await page.get_by_test_id("username").fill(email)
        await page.get_by_role("button", name="Continue").click()
        await page.get_by_test_id("validateCode").fill("123456")


# Test function
@pytest.mark.asyncio
async def test_workspace_switch_on_reload():
    try:
        contextA, pageA, playwrightA = await start_browser_with_proxy()
        # Executing the following while replaying
        await enter_dummy_otp_if_not_logged_in(pageA, USER_EMAILA)
        # Executing the following while recording on an already signed up account
        # await sign_in(pageA, email=USER_EMAILA, password=PASSWORD_A)
        await expect(pageA.get_by_label("Inbox")).to_be_visible()

        message = "Hello How are you"
        await send_message(pageA, USER_EMAILB, message)

        contextB, pageB, playwrightB = await start_browser_with_proxy()
        # Executing the following while replaying
        await enter_dummy_otp_if_not_logged_in(pageB, USER_EMAILB)
        # Executing the following while recording on an already signed up account
        # await sign_in(pageB, email=USER_EMAILB, password=PASSWORD_B)
        await expect(pageA.get_by_label("Inbox")).to_be_visible()

        reply = "Hey I am good. What about you?"
        await create_draft_reply_in_thread(
            pageB, USERNAME_A, USER_EMAILA, message, reply
        )
        await navigate_away_check_draft(pageB, USERNAME_A, USER_EMAILA)

    finally:
        await close_browser(contextA, pageA, playwrightA)
        await close_browser(contextB, pageB, playwrightB)
</EXAMPLE_INPUT_2>

Which we port as such

<EXAMPLE_OUTPUT_2>
import sys, os
import uuid
import pytest
from trace_cleaner import trace_cleaner  # adjust import path as needed

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import expect, async_playwright, Browser, Page
from utils.login import sign_in, NEW_DOT_URL, check_if_logged_in
from utils.browser import close_browser

USER_EMAILA = "doorsqueaky+3182311@gmail.com"
USERNAME_A = "John Doe"
PASSWORD_A = "eabcahqraatorpvt"
USER_EMAILB = "windowsqueaky+3919211@gmail.com"
PASSWORD_B = "jeaqzqklzxxdetsh"
USERNAME_B = "Jane Doe"


async def start_browser_with_proxy():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        slow_mo=500,
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = await browser.new_context(ignore_https_errors=True)
    page = await context.new_page()
    return context, page, playwright


async def send_message(page: Page, recepient_email: str, message: str):
    await page.locator('button[aria-label="Start chat (Floating action)"]').click()
    await page.locator('div[aria-label="Start chat"][role="menuitem"]').click()
    await page.locator('input[aria-label="Name, email, or phone number"]').fill(recepient_email)
    await (
        page.get_by_test_id("selection-list")
        .locator(f'button:has-text("{recepient_email}")')
        .click()
    )
    await page.wait_for_timeout(1000)
    message_editor = page.locator(
        'div[contenteditable="true"][placeholder="Write something..."]'
    ).last
    await message_editor.fill(message)
    await message_editor.press("Enter")


async def create_draft_reply_in_thread(
    page: Page, username: str, sender_email: str, message: str, reply: str
):
    if await page.locator(
        'button[aria-label="Navigates to a chat"]', has_text=sender_email
    ).is_visible():
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=sender_email
        ).click()
    else:
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=username
        ).click()

    await page.locator('div[aria-label="Chat message"]', has_text=message).click(
        button="right"
    )
    await page.locator('div[aria-label="Reply in thread"][role="menuitem"]').click()
    await page.wait_for_timeout(1000)

    message_editor = page.locator(
        'div[contenteditable="true"][placeholder="Write something..."]'
    ).last
    await message_editor.fill(reply)


async def navigate_away_check_draft(page: Page, username: str, sender_email: str):
    if await page.locator(
        'button[aria-label="Navigates to a chat"]', has_text=sender_email
    ).is_visible():
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=sender_email
        ).click()
    else:
        await page.locator(
            'button[aria-label="Navigates to a chat"]', has_text=username
        ).click()

    await page.wait_for_timeout(2000)
    draft_reply_LHN_btn = page.locator(
        'button[aria-label="Navigates to a chat"]', has_text="No activity yet"
    )
    await expect(draft_reply_LHN_btn).to_be_visible()


async def enter_dummy_otp_if_not_logged_in(page: Page, email: str):
    if not await check_if_logged_in(page=page, url=NEW_DOT_URL):
        await page.get_by_test_id("username").fill(email)
        await page.get_by_role("button", name="Continue").click()
        await page.get_by_test_id("validateCode").fill("123456")


@pytest.mark.asyncio
async def test_workspace_switch_on_reload(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    try:
        contextA, pageA, playwrightA = await start_browser_with_proxy()
        if trace_enabled:
            await contextA.tracing.start(screenshots=True, snapshots=True, sources=True)

        await enter_dummy_otp_if_not_logged_in(pageA, USER_EMAILA)
        await expect(pageA.get_by_label("Inbox")).to_be_visible()

        message = "Hello How are you"
        await send_message(pageA, USER_EMAILB, message)

        contextB, pageB, playwrightB = await start_browser_with_proxy()
        if trace_enabled:
            await contextB.tracing.start(screenshots=True, snapshots=True, sources=True)

        await enter_dummy_otp_if_not_logged_in(pageB, USER_EMAILB)
        await expect(pageA.get_by_label("Inbox")).to_be_visible()

        reply = "Hey I am good. What about you?"
        await create_draft_reply_in_thread(
            pageB, USERNAME_A, USER_EMAILA, message, reply
        )
        await navigate_away_check_draft(pageB, USERNAME_A, USER_EMAILA)

    finally:
        if trace_enabled:
            trace_pathA = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_pathA), exist_ok=True)
            await contextA.tracing.stop(path=trace_pathA)
            trace_cleaner(trace_pathA)

            trace_pathB = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_pathB), exist_ok=True)
            await contextB.tracing.stop(path=trace_pathB)
            trace_cleaner(trace_pathB)

        await close_browser(contextA, pageA, playwrightA)
        await close_browser(contextB, pageB, playwrightB)
</EXAMPLE_OUTPUT_2>

Now, based on the instructions and examples above, you should port the following
test file:

<INPUT>
§input_file
</INPUT>

You should wrap the code in your response in <OUTPUT></OUTPUT> tags. If you find
that it is not possible to perform the porting, you should respond with
<ERROR></ERROR> tags, containing an explanation of why the porting is not
possible.

Note: You absolutely must not change anything unrelated to the integration of the user tool. Functionality when tracing is disabled must remain identical to the original script; make sure not to change any selectors, strings being passed into functions, etc. 

Please proceed.
