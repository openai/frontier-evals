import os
import sys
import time
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright

site_url = "https://dev.new.expensify.com:8082/"
first_name = "Sujil"
last_name = "Devkota"
chat_message = "Hello, how are you?"
user_email_message = "sujil.d@turing.com"


def generate_random_email():
    int(time.time())
    return "freelanceapptest+17330893089@gmail.com"


def create_new_account_and_login(page):
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
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)
    return user_email


def test_current_report_showing(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            proxy={"server": "http://localhost:8080"},
            slow_mo=500,
        )

        # Explicitly create a browser context so we can access tracing API easily
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard

            # Core test steps
            create_new_account_and_login(page)
            setting_button = page.locator("button[aria-label='My settings']")
            expect(setting_button).to_be_visible()
            setting_button.click()

            preferences_div = page.locator("text='Preferences'")
            expect(preferences_div).to_be_visible()
            preferences_div.click()

            priority_mode_option = page.locator("text='Priority mode'")
            expect(priority_mode_option).to_be_visible()
            priority_mode_option.click()

            focus_option = page.locator("text='#focus'")
            expect(focus_option).to_be_visible()
            focus_option.click()

            plus_button = page.locator('button[aria-label="Start chat (Floating action)"]')
            expect(plus_button).to_be_visible()
            plus_button.click()

            start_chat_option = page.locator('div[aria-label="Start chat"]')
            expect(start_chat_option).to_be_visible()
            start_chat_option.click()

            email_input = page.locator('input[aria-label="Name, email, or phone number"]')
            expect(email_input).to_be_visible()
            email_input.fill(user_email_message)
            email_input.press("Enter")
            page.wait_for_timeout(1000)

            page.get_by_role("textbox").fill(chat_message)
            page.get_by_role("textbox").press("Enter")
            page.wait_for_timeout(1000)

            last_chat_message = page.locator('div[aria-label="Last chat message preview"]').last
            expect(last_chat_message).to_have_text(chat_message)

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()  # Ensuring context is closed
            browser.close()
