import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email():
    return "expensifyapp97+0090@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
):
    """
    Launches a browser, creates a context & page, performs initial login / setup,
    and (optionally) starts tracing on the context immediately after creation.
    Returns (browser, context, page, user_email)
    """
    proxy = {"server": "http://127.0.0.1:8080/"}
    args = [
        "--disable-web-security",
        "--disable-features=IsolateOrigins,site-per-process",
    ]
    browser: Browser = p.chromium.launch(headless=True, args=args, proxy=proxy)

    # Explicitly create a context so we can start tracing right away
    context = browser.new_context()

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)

    user_email = generate_random_email()

    # ------  Original login flow  ------
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)
    try:
        page.locator('button[tabindex="0"]').click()
        time.sleep(4)
    except Exception:
        pass

    if page.locator("text='What do you want to do today?'").count() == 1:
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(2000)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(2000)
    else:
        page.locator('button[aria-label="My settings"]').click()
        page.get_by_label(user_email).first.click()
        page.get_by_role("textbox", name="First name").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").click()
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Save").click()
        page.locator('button[aria-label="Inbox"]').click()

    return browser, context, page, user_email


def test_gmail_user_pay(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)
        try:
            test_email = "expensifyapp9988775511@gmail.com"

            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            expect(plus_icon).to_be_visible()
            plus_icon.click()
            page.wait_for_timeout(2000)

            start_chat = page.locator('div[aria-label="Start chat"]')
            expect(start_chat).to_be_visible()
            start_chat.click()
            page.wait_for_timeout(2000)

            search_email = page.locator('input[aria-label="Name, email, or phone number"]')
            expect(search_email).to_be_visible()
            search_email.fill(test_email)
            page.wait_for_timeout(2000)
            search_email.click()
            page.wait_for_timeout(2000)

            select_email = page.locator(f'button[aria-label="{test_email}"]')
            expect(select_email).to_be_visible()
            select_email.click()
            page.wait_for_timeout(4000)

            create_pay_expense = page.locator('button[aria-label="Create"]').last
            expect(create_pay_expense).to_be_visible()
            create_pay_expense.click()

            pay_expense = f"Pay {test_email}"
            pay = page.get_by_text(pay_expense)
            pay.click()
            page.wait_for_timeout(2000)

            currency = page.locator('button[aria-label="Select a currency"]')
            expect(currency).to_be_visible()
            currency.click()

            search_currency = page.locator('input[aria-label="Search"]')
            search_currency.fill("USD")

            choose_usd = page.locator('button[aria-label="USD - $"]')
            expect(choose_usd).to_be_visible()
            choose_usd.click()

            amount = page.locator('input[placeholder="0"]')
            expect(amount).to_be_visible()
            amount.fill("100")

            next_button = page.locator('button[data-listener="Enter"]', has_text="Next")
            expect(next_button).to_be_visible()
            next_button.click()
            page.wait_for_timeout(2000)

            pay_with_expensify = page.locator(
                'button[data-listener="Enter"]', has_text="Pay with Expensify"
            )
            expect(pay_with_expensify).to_be_visible()
            pay_with_expensify.click()
            page.wait_for_timeout(2000)

            not_here_page = page.get_by_text("Oops, this page cannot be found")
            assert not not_here_page.is_visible(), (
                "Not here page opens when Gmail user pays with Expensify"
            )

            page.wait_for_timeout(3000)

        finally:
            if trace_enabled:
                # Export and clean the trace
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
