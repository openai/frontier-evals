import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from time import time

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright


def generate_random_email():
    """
    Generates a (pseudo-)random email address. Timestamp is used to ensure
    uniqueness, but you are free to adjust this logic as needed.
    """
    timestamp = int(time())
    return f"bobexpensifytest+{timestamp}@gmail.com"


def login_user(
    page: Page, user_email: str, first_name: str = "Milan", last_name: str = "T"
) -> Page:
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
    except Exception:
        # A second click is sometimes needed when the button is rendered twice.
        pass
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    return page


def launch_app(
    p, headless: bool = True, device: str | None = None, geolocation: dict | None = None
):
    """
    Launch the Expensify application in a new browser context.
    """
    browser = p.chromium.launch(
        channel="chrome",
        headless=headless,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=1000,
    )

    context_args: dict = {}
    if device:
        context_args.update(p.devices[device])
    if geolocation:
        context_args["geolocation"] = geolocation
        context_args["permissions"] = ["geolocation"]
    context_args["timezone_id"] = "Asia/Karachi"

    context = browser.new_context(**context_args)
    page = context.new_page()
    return browser, context, page


def test_mobile_get_assistance_page(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser, context, page = launch_app(p, device="iPhone 12 Pro")
        install_online_guard_sync(
            context, page
        )  # Install the online guard immediately after creation

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            user_email = generate_random_email()
            page = login_user(page, user_email, first_name="Bob")

            # Close any onboarding popup if present
            try:
                page.locator('button[aria-label="Close"]').click(timeout=3000)
            except Exception:
                # Popup might not always be present
                pass

            # Begin the UI flow we are validating
            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            expect(plus_icon).to_be_visible()
            plus_icon.click()

            new_workspace_button = page.locator('div[aria-label="New workspace"]')
            expect(new_workspace_button).to_be_visible()
            new_workspace_button.click()

            more_features_button = page.locator('div[aria-label="More features"][role="menuitem"]')
            expect(more_features_button).to_be_visible()
            more_features_button.click()

            workflow_switch = page.locator(
                'button[aria-label="Configure how spend is approved and paid."][role="switch"]'
            )
            expect(workflow_switch).to_be_visible()
            workflow_switch.click()

            workflows_button = page.locator('div[aria-label="Workflows"][role="menuitem"]')
            expect(workflows_button).to_be_visible()
            workflows_button.click()

            connect_bank_account = page.locator(
                'div[aria-label="Connect bank account"][role="menuitem"]'
            )
            expect(connect_bank_account).to_be_visible()
            connect_bank_account.click()

            update_to_usd = page.locator('button[data-listener="Enter"]', has_text="Update to USD")
            expect(update_to_usd).to_be_visible()
            update_to_usd.click()

            page.wait_for_timeout(2000)

            get_assistant = page.locator(
                'button[aria-label="Get assistance from our team"][role="button"]'
            )
            expect(get_assistant).to_be_visible()
            get_assistant.click()

            chat_with_concierge = page.locator(
                'div[aria-label="Chat with Concierge"][role="menuitem"]'
            )
            expect(chat_with_concierge).to_be_visible()
            chat_with_concierge.click()

            back_arrow_button = page.locator('button[aria-label="Back"]').last
            expect(back_arrow_button).to_be_visible()
            back_arrow_button.click()

            chat_with_concierge = page.locator(
                'div[aria-label="Chat with Concierge"][role="menuitem"]'
            )
            expect(chat_with_concierge).to_be_visible()

        finally:
            # Stop tracing if it was started and clean the trace
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Always close the browser to free resources
            browser.close()
