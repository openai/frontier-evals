import os
import re
import sys
import uuid

import pytest
from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright.sync_api import Page, expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

EMAIL = "Adebayovicktor0+test47282@gmail.com"
TWO_SECONDS = 2000
THREE_SECONDS = 3000
FIVE_SECONDS = 5000


def login_user(
    p: PlaywrightContextManager, *, trace_enabled: bool = False
) -> tuple[Browser, "BrowserContext", Page, str]:
    """
    Launches a browser, creates a context and page, and logs the user in.
    If `trace_enabled` is True, tracing is started immediately after the
    context is created.
    """
    user_email = EMAIL
    browser = p.chromium.launch(
        headless=True,
        proxy={"server": "http://127.0.0.1:8080/"},
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)
    page.goto("https://dev.new.expensify.com:8082", timeout=60000)

    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()
    phone_or_email_input.fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)

    magic_code = "123456"
    page.wait_for_timeout(FIVE_SECONDS)
    validate_code_input = page.locator('input[name="validateCode"]').nth(0)
    expect(validate_code_input).to_be_visible()
    validate_code_input.fill(magic_code)
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    page.wait_for_timeout(THREE_SECONDS)

    return browser, context, page, user_email


def check_members(page: Page, formatted_currency: str):
    page.get_by_label(f"Tracking {formatted_currency}", exact=False).first.click()
    page.wait_for_timeout(1000)
    members = page.locator("div").filter(has_text=re.compile("^Members$")).nth(3)
    assert not members.is_visible()


@pytest.mark.parametrize("first_name, last_name", [("Yagna", "Sai")])
def test_run(first_name: str, last_name: str, pytestconfig) -> None:
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as playwright:
        # ----------------------------------------
        # Setup phase (browser / context creation)
        # ----------------------------------------
        browser = context = page = None  # make the variables visible in finally
        try:
            (
                browser,
                context,
                page,
                user_email,
            ) = login_user(playwright, trace_enabled=trace_enabled)

            print(f"Logged in with email: {user_email}")

            # -----------------
            # Test steps
            # -----------------
            page.wait_for_timeout(THREE_SECONDS)
            page.locator('button[aria-label="Inbox"]').nth(-1).click()
            page.wait_for_timeout(TWO_SECONDS)
            (
                page.locator(
                    'div[data-testid="DisplayNamesWithTooltip"]',
                    has_text=f"{first_name} {last_name}",
                ).last.click()
            )
            page.wait_for_timeout(TWO_SECONDS)
            page.get_by_label("Back").click()
            page.wait_for_timeout(TWO_SECONDS)
            page.locator('button[aria-label="Create"]').last.click()
            page.wait_for_timeout(TWO_SECONDS)
            page.get_by_text("Track expense").click()
            page.get_by_text("Manual").last.click()

            amount = 9226
            formatted_amount = "{:,}".format(amount)
            page.get_by_placeholder("0").fill(str(amount))
            page.wait_for_timeout(TWO_SECONDS)
            page.get_by_role("button", name="Next").first.click()
            page.wait_for_timeout(TWO_SECONDS)

            currencies = (
                page.get_by_test_id("IOURequestStepConfirmation")
                .get_by_test_id("selection-list")
                .filter(has_text=formatted_amount)
                .all_inner_texts()
            )
            currencies = [item for item in currencies[0].split("\n") if formatted_amount in item]
            currency = currencies[0].split("\xa0")[0]

            page.get_by_role("button", name="Track expense").last.click()
            page.get_by_test_id("report-actions-list").get_by_text(
                f"{formatted_amount}.00"
            ).last.click()
            page.wait_for_timeout(TWO_SECONDS)
            check_members(page, currency)

        finally:
            # -----------------
            # Teardown phase
            # -----------------
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                # Export the tracing data
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            if context is not None:
                context.close()

            if browser is not None:
                browser.close()


if __name__ == "__main__":
    pytest.main([__file__])
