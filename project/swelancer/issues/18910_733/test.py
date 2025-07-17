import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright.sync_api import expect, sync_playwright

site_url = "https://dev.new.expensify.com:8082/"
primary_email = "sujildevkota@gmail.com"


def generate_random_email():
    timestamp = int(time.time())
    return f"{primary_email.split('@')[0]}+{timestamp}@{primary_email.split('@')[1]}"


def create_new_account_and_login(page, first_name, last_name):
    user_email = generate_random_email()
    page.goto(site_url, timeout=60000)
    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()
    phone_or_email_input.fill(user_email)
    page.wait_for_timeout(1000)
    continue_button = page.locator('button[tabindex="0"]')
    expect(continue_button).to_be_visible()
    continue_button.click()
    page.wait_for_timeout(1000)
    try:
        print("Clicking the join button again if needed")
        expect(continue_button).to_be_visible()
        continue_button.click()
    except Exception:
        pass
    page.locator("text='Track and budget expenses'").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)
    return user_email


def test_workspace_settings(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    first_name = "Sujil"
    last_name = "Devkota"

    # Initialise variables so they are available in finally block
    context1 = context2 = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                channel="chrome",
                headless=True,
                slow_mo=500,
                args=[
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ],
            )
            try:
                # Context 1
                context1 = browser.new_context()
                if trace_enabled:
                    context1.tracing.start(screenshots=True, snapshots=True, sources=True)
                page1 = context1.new_page()
                install_online_guard_sync(context1, page1)
                create_new_account_and_login(page1, first_name + "1", last_name + "1")

                # Context 2
                context2 = browser.new_context()
                if trace_enabled:
                    context2.tracing.start(screenshots=True, snapshots=True, sources=True)
                page2 = context2.new_page()
                install_online_guard_sync(context2, page2)
                create_new_account_and_login(page2, first_name + "2", last_name + "2")

                # Begin workspace actions
                plus_button = page2.locator('button[aria-label="Start chat (Floating action)"]')
                expect(plus_button).to_be_visible()
                plus_button.click()
                page2.wait_for_timeout(1000)

                new_workspace_option = page2.locator('div[aria-label="New workspace"]')
                expect(new_workspace_option).to_be_visible()
                new_workspace_option.click()
                page2.wait_for_timeout(1000)

                more_features_option = page2.locator('div[aria-label="Members"]')
                expect(more_features_option).to_be_visible()
                more_features_option.click()

                invite_member_button = page2.locator(
                    'button:has(div:has-text("Invite member"))'
                ).last
                expect(invite_member_button).to_be_visible()
                invite_member_button.click()

                email_input = page2.locator('input[aria-label="Name, email, or phone number"]')
                expect(email_input).to_be_visible()
                email_input.fill("sujildevkota+1733384438@gmail.com")
                page2.wait_for_timeout(1000)

                user_email_option = page2.locator(
                    f'''button[aria-label="{first_name + "1" + " " + last_name + "1"}"] div:has-text("sujildevkota+1733384438@gmail.com")'''
                ).last
                expect(user_email_option).to_be_visible()
                user_email_option.click()
                page2.wait_for_timeout(1000)
                page2.keyboard.press("Enter")

                invite_button = page2.locator("button:has(div)").get_by_text("Invite", exact=True)
                expect(invite_button).to_be_visible()
                invite_button.click()
                page2.wait_for_timeout(1000)

                workspace_setting_url = page2.url
                install_online_guard_sync(context1, page1)
                page1.goto(workspace_setting_url)
                page1.wait_for_timeout(5000)
                more_features_menu = page1.locator('div[aria-label="More features"]')
                categories_menu = page1.locator('div[aria-label="Categories"]')
                assert more_features_menu.count() == 0, (
                    "More features menu should not be visible for user1"
                )
                assert categories_menu.count() == 0, (
                    "Categories menu should not be visible for user1"
                )
            finally:
                # Stop tracing and clean traces if enabled
                if trace_enabled:
                    if context1 is not None:
                        trace_path1 = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                        os.makedirs(os.path.dirname(trace_path1), exist_ok=True)
                        context1.tracing.stop(path=trace_path1)
                        trace_cleaner(trace_path1)
                    if context2 is not None:
                        trace_path2 = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                        os.makedirs(os.path.dirname(trace_path2), exist_ok=True)
                        context2.tracing.stop(path=trace_path2)
                        trace_cleaner(trace_path2)

                # Close the browser
                browser.close()
    finally:
        # Additional safeguard to ensure contexts are closed if the browser wasn't
        if context1 is not None:
            try:
                context1.close()
            except Exception:
                pass
        if context2 is not None:
            try:
                context2.close()
            except Exception:
                pass
