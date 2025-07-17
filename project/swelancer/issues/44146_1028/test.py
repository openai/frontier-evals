import os
import random
import sys
import time
import uuid

from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

# Make utils importable when running directly
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)


def generate_random_email():
    """
    NOTE: The original implementation returned a constant email.
    We keep the behaviour intact to avoid altering the test logic.
    """
    int(time.time())  # still computed but unused (kept for parity)
    return "kenworktestemail+993@gmail.com"


def generate_random_name():
    first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Sam", "Max", "Taylor"]
    last_names = [
        "Smith",
        "Doe",
        "Johnson",
        "Williams",
        "Brown",
        "Davis",
        "Miller",
        "Wilson",
    ]
    return random.choice(first_names), random.choice(last_names)


def login_user(page: Page, email: str, first_name: str, last_name: str):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)

    # Attempt the click-again flow wrapped in a broad try/except
    try:
        page.locator('button[tabindex="0"]').click()
        # Intentional typo from original script kept for parity
        page_wait_for_timeout(1000)
    except Exception:
        pass

    if page.get_by_text("Track and budget expenses").is_visible():
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(1000)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(1000)


def create_workspace(page: Page):
    page.locator('button[aria-label="Workspaces"]').click()
    page.wait_for_timeout(1000)
    (
        page.get_by_test_id("WorkspaceSwitcherPage")
        .get_by_role("button", name="New workspace")
        .click()
    )
    page.locator('button[aria-label="Back"]').click()
    page.wait_for_timeout(1000)
    page.locator('button[aria-label="Back"]').click()
    page.wait_for_timeout(1000)


def _start_tracing_if_enabled(context, trace_enabled: bool):
    """Utility to start tracing on a BrowserContext if requested."""
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


def _stop_tracing_if_enabled(context, trace_enabled: bool):
    """Utility to stop tracing, save and clean if requested."""
    if trace_enabled:
        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        context.tracing.stop(path=trace_path)
        trace_cleaner(trace_path)


def test_lhn_workspace(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins, site-per-process",
            ],
        )
        context = browser.new_context()
        _start_tracing_if_enabled(context, trace_enabled)

        try:
            page = context.new_page()
            install_online_guard_sync(
                context, page
            )  # Installed online guard immediately after page creation

            email = generate_random_email()
            first_name, last_name = generate_random_name()
            workspace_name = f"{email}'s Workspace"

            login_user(page, email, first_name, last_name)
            create_workspace(page)

            page.wait_for_timeout(2000)

            # Attempt to fetch the workspace label text
            try:
                lhn_workspace_text = page.locator(
                    'button[aria-label="Navigates to a chat"]',
                    has_text=workspace_name,
                ).inner_text()
            except Exception:
                lhn_workspace_text = (
                    page.locator(
                        'button[aria-label="Navigates to a chat"]',
                        has_text=workspace_name,
                    )
                    .nth(0)
                    .inner_text()
                )

            assert "No activity yet" not in lhn_workspace_text, (
                "'No activity yet' is shown for workspace in LHN!"
            )

        finally:
            # Always run cleanup
            _stop_tracing_if_enabled(context, trace_enabled)
            browser.close()
