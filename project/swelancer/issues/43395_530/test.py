import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright

site_url = "https://dev.new.expensify.com:8082/"
primary_email = "freelanceapptest@gmail.com"
mobile_browser = True
device_name = "Pixel 7"


def generate_email():
    timestamp = "89994947122"
    return f"{primary_email.split('@')[0]}+{timestamp}@{primary_email.split('@')[1]}"


def create_new_account_and_login(page, first_name, last_name):
    user_email = generate_email()
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
        # Attempt to click the join button again if it is still visible
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
    page.get_by_label("Close").click()
    return user_email


def _start_tracing_if_enabled(context, trace_enabled):
    """
    Helper to start Playwright tracing on the provided context when the
    --user-tool-trace flag is enabled.
    """
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_tracing_and_clean(context, trace_enabled):
    """
    Helper that stops tracing on the context, exports the trace to a unique
    zip file, and runs trace_cleaner on that file when tracing is enabled.
    """
    if trace_enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_workspace_settings(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    first_name = "Sujil"
    last_name = "Devkota"

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
            slow_mo=1000,
        )

        # Create browser context (explicitly, so we always have a context handle)
        if mobile_browser:
            mobile_device = p.devices[device_name]
            context = browser.new_context(**mobile_device)
        else:
            context = browser.new_context()

        # Start tracing if requested
        _start_tracing_if_enabled(context, trace_enabled)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard

            create_new_account_and_login(page, first_name, last_name)

            button_search = page.locator('button[aria-label="Search"]')
            expect(button_search).to_be_visible()
            button_search.click()
            page.wait_for_timeout(1000)

            search_icon = page.locator('div[aria-label="Search"] svg')
            expect(search_icon).to_be_visible()
            search_icon.click()
            page.wait_for_timeout(1000)

            search_input = page.locator("input")
            expect(search_input).to_be_visible()
            search_input.fill("Random message")
            page.keyboard.press("Enter")
            page.wait_for_timeout(1000)

            svg_elements = page.locator("svg[height='108']")
            assert svg_elements.count() > 1, "The SVG elements are not set to correct height"

        finally:
            _stop_tracing_and_clean(context, trace_enabled)
            context.close()  # Ensure context is closed before browser
            browser.close()
