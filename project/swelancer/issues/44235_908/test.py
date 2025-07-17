import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re
import time

from playwright.sync_api import expect, sync_playwright

static_email = "testuserexpensify1+35@gmail.com"


def test_open_page_and_validate_row_hover(pytestconfig):
    """
    Opens the Concierge page, signs the user in (creates an account if necessary),
    and validates that the Concierge button is visible.

    If the --user-tool-trace flag is supplied, Playwright tracing data is also
    captured, exported, and cleaned.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    site_url = "https://dev.new.expensify.com:8082/concierge"
    timestamp = int(time.time())

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            slow_mo=1000,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        # Explicitly create a context so that we can start/stop tracing on it.
        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.goto(site_url, timeout=60000)

            phone_or_email_input = page.locator('input[type="email"]')
            phone_or_email_input.fill(static_email)

            continue_button = page.locator('button[tabindex="0"]')
            continue_button.click()

            join_button = page.get_by_role("button", name="Join")
            join_button.click()

            page.locator("div").filter(has_text=re.compile("^Something else$")).first.click()

            page.locator("body").press("Enter")
            page.locator('input[name="fname"]').fill("Account")
            page.locator('input[name="lname"]').fill(f"{timestamp}")
            page.get_by_role("button", name="Continue").last.click()
            page.get_by_role("button", name="Get Started").click()

            page.goto(site_url, timeout=60000)
            concierge_button = page.locator(
                'button[aria-label="concierge@expensify.com"]', has_text="Concierge"
            ).first
            concierge_button.wait_for(state="visible")
            expect(concierge_button).to_be_visible()
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
