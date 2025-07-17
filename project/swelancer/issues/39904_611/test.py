import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re
import time

from playwright.sync_api import sync_playwright
from utils.online_guard import install_online_guard_sync

user_email = "playwright2user+1@gmail.com"
password = "zehr mglm gizg gjcc"


def modify_verification_link_for_local_test(verification_link):
    if verification_link:
        modified_link = re.sub(
            r"https://new\.expensify\.com",
            "https://dev.new.expensify.com:8082",
            verification_link,
        )
        print(f"Modified link: {modified_link}")
        return modified_link
    return None


def login_and_verify(p, trace_enabled: bool = False):
    """
    Launches a browser, performs the login flow and returns the browser,
    context and a page object positioned after the login attempt.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(ignore_https_errors=True, bypass_csp=True)

    # Start tracing for this context if requested
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)

    # Fill email and continue
    email_input = page.locator('input[type="email"]')
    email_input.wait_for()
    email_input.fill(user_email)
    continue_button = page.locator('button[tabindex="0"]')
    continue_button.wait_for()
    continue_button.click()
    page.wait_for_timeout(1000)

    # In some cases, Expensify shows a second button – attempt to click it
    try:
        continue_button = page.locator('button[tabindex="0"]')
        continue_button.wait_for()
        continue_button.click()
    except Exception:
        pass

    # For local testing, we use a hard-coded verification link and adapt it
    verification_link = "https://new.expensify.com/v/18587801/109829"
    modified_link = modify_verification_link_for_local_test(verification_link)
    if modified_link:
        new_tab = context.new_page()
        install_online_guard_sync(context, new_tab)
        new_tab.goto(modified_link)
        # Make sure the magic-code screen NEVER flashes up
        duration = 3
        check_interval = 0.1
        start_time = time.time()
        while time.time() - start_time < duration:
            if new_tab.locator("text=Here's your magic code").is_visible():
                raise AssertionError("Unexpected 'Here's your magic code' screen appeared briefly.")
            time.sleep(check_interval)
    else:
        print("Verification link not found or could not be modified.")

    return browser, context, page


def _export_and_clean_trace(context, trace_enabled: bool):
    """
    Helper that exports and cleans a trace for the given context if tracing is enabled.
    """
    if trace_enabled and context is not None:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_login_magic_code_flow(pytestconfig):
    """
    Reproduction of the original `test` function, now wrapped in a try/finally
    with optional tracing support.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Perform the login & verification flow
            browser, context, page = login_and_verify(p, trace_enabled=trace_enabled)

            # Give the UI a few seconds before tearing everything down
            if page:
                page.wait_for_timeout(5000)

        finally:
            # Stop tracing & clean traces if requested
            _export_and_clean_trace(context, trace_enabled)

            # Close the context if it was created
            if context is not None:
                context.close()

            # Close the browser if it was created
            if browser is not None:
                browser.close()
