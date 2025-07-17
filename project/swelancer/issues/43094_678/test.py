import os
import re
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

PROXY = False
PROXY_OTP = True

import time

from playwright.async_api import Page, async_playwright, expect
from utils.browser import close_browser, start_browser
from utils.email_handler import (
    EmailHandler,
    get_specific_derived_email,
)
from utils.login import sign_in
from utils.online_guard import install_online_guard_async

BASE_EMAIL = "testerfreelancer555@gmail.com"
PASSWORD = "snct ajbr yadx kwlq"
ISSUE_NUMBER = "43094"
SUFFIX_1 = f"{ISSUE_NUMBER}1"
SUFFIX_2 = f"{ISSUE_NUMBER}2"


async def create_a_new_workspace(page):
    await page.get_by_label("My settings").click()
    await (
        page.get_by_test_id("InitialSettingsPage")
        .get_by_label("Workspaces")
        .get_by_text("Workspaces")
        .click()
    )
    await page.wait_for_timeout(3000)
    try:
        await page.get_by_label("New workspace").click()
    except Exception:
        await page.get_by_label("New workspace").first.click()
    await page.get_by_test_id("WorkspacePageWithSections").get_by_text("Name").click()
    random_name = "1733413785"
    await page.get_by_role("textbox", name="Name").fill(random_name)
    await page.wait_for_timeout(2000)
    await page.get_by_role("button", name="Save").click()
    return random_name


async def create_room_in_workspace(page: Page, room_name: str, workspace_name: str):
    plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
    await expect(plus_icon).to_be_visible()
    await plus_icon.click()
    start_chat = page.locator('div[aria-label="Start chat"]')
    await expect(start_chat).to_be_visible()
    await start_chat.click()
    room_button = page.locator('button[aria-label="Room"]')
    await expect(room_button).to_be_visible()
    await room_button.click()
    await page.locator('input[aria-label="Room name"]').fill(room_name)
    await page.get_by_test_id("WorkspaceNewRoomPage").get_by_text("Workspace").first.click()
    await page.get_by_label(workspace_name).click()
    await page.locator(
        'button[tabindex="0"][data-listener="Enter"]', has_text="Create room"
    ).click()


async def add_member_to_workspace(page: Page, member_email: str):
    members_button = page.locator('div[aria-label="Members"][role="menuitem"]')
    await expect(members_button).to_be_visible()
    await members_button.click()
    invite_members = page.get_by_text("Invite member")
    await expect(invite_members).to_be_visible()
    await invite_members.click()
    await page.locator('input[type="text"]').fill(member_email)
    await page.wait_for_timeout(1000)
    button = page.locator('button[aria-label="Test_2 User_2"]').nth(1)
    await button.click()
    await page.locator('button[tabindex="0"][data-listener="Enter"]').nth(0).click()
    await page.locator('button[tabindex="0"][data-listener="Enter"]', has_text="Invite").click()
    await page.wait_for_timeout(1000)
    back_arrow_button = page.locator('button[aria-label="Back"]')
    await back_arrow_button.click()


async def sign_in_new_dot(page: Page, email: str, password: str):
    """
    Sign in into the new Expensify dot.
    """
    await page.goto("https://dev.new.expensify.com:8082", timeout=60000)
    # Enter email
    await page.get_by_test_id("username").fill(email)
    await page.get_by_role("button", name="Continue").click()

    # Await OTP
    otp = "123456"
    await page.wait_for_timeout(1000)
    await page.get_by_test_id("validateCode").fill(otp)
    try:
        await page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass

    # Wait sign in to complete
    await page.get_by_text("Please enter the magic code").wait_for(state="hidden")


@pytest.mark.asyncio
async def test_duplicated_search(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    proxy = {"server": "http://127.0.0.1:8080/"} if PROXY else None
    launch_args = [
        "--ignore-certificate-errors",
        "--disable-web-security",
        "--disable-features=IsolateOrigins,site-per-process",
    ]

    # Initialise playwright and browser
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        slow_mo=2000, headless=True, args=launch_args, proxy=proxy
    )
    context = await browser.new_context()
    page = await context.new_page()
    await install_online_guard_async(
        context, page
    )  # Install the online guard immediately after creation

    # Start tracing if requested
    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    try:
        alice_email = get_specific_derived_email(email=BASE_EMAIL, suffix=SUFFIX_1)
        bob_email = get_specific_derived_email(email=BASE_EMAIL, suffix=SUFFIX_2)

        # Alice creates workspace and rooms
        await sign_in_new_dot(page, email=alice_email, password=PASSWORD)
        workspace_name = await create_a_new_workspace(page)
        await add_member_to_workspace(page, bob_email)
        await create_room_in_workspace(page, "room1", workspace_name)
        await create_room_in_workspace(page, "room2", workspace_name)

        # Sign out Alice
        await page.wait_for_timeout(2000)
        await page.get_by_label("My settings").click()
        await page.get_by_text("Sign out").click()

        # Bob signs in and searches for the room
        await sign_in_new_dot(page, email=bob_email, password=PASSWORD)
        find_button = page.locator('div[aria-label="Find something..."]')
        await expect(find_button).to_be_visible()
        await find_button.click()
        await page.locator('input[aria-label="Name, email, or phone number"]').fill("room1")

        button_locator = page.locator('button[aria-label="#room1"][role="button"]')
        await button_locator.first.wait_for(state="visible")

        search_results = await page.locator(
            'button[aria-label="#room1"][role="button"]'
        ).evaluate_all("list => list.map(element => element.textContent)")

        count = 0
        for item in search_results:
            if len(item.split(workspace_name)) > 1:
                count += 1

        assert count == 1

    finally:
        # Stop tracing and clean the trace if enabled
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Close browser and playwright
        await close_browser(context, page, playwright)
