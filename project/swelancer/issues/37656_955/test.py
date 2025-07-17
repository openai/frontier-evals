import os
import sys
import uuid

import pytest
from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

SITE_URL = "https://dev.new.expensify.com:8082/"
PRIMARY_EMAIL = "expensifyolddot+1733111691@gmail.com"
FIRST_NAME = "Expenolddot"
LAST_NAME = "1733111691"


@pytest.fixture
def setup_playwright(request):
    """
    Fixture that starts Playwright, launches a browser/context, and yields a page.
    If the --user-tool-trace flag is supplied, tracing is started immediately
    after context creation and stopped/cleaned up in the tear-down phase.
    """
    # Retrieve viewport parameters (if any)
    params = getattr(request, "param", {}) or {}
    width = params.get("width", 1280)
    height = params.get("height", 720)

    # Check whether tracing has been requested
    trace_enabled = request.config.getoption("--user-tool-trace")

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(
        headless=True,
        proxy={"server": "http://127.0.0.1:8080"},
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(viewport={"width": width, "height": height})

    # Start tracing immediately after the context is created
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation

    try:
        yield page
    finally:
        # Stop tracing and clean trace if enabled
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Always close the context, browser and stop Playwright
        context.close()
        browser.close()
        playwright.stop()


def login_user(page: Page, email: str, first_name: str, last_name: str):
    page.goto(SITE_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator("button", has_text="Join").click()
        page.wait_for_timeout(1000)
    except Exception:
        pass
    if not page.locator("text='Track and budget expenses'").is_visible():
        page.evaluate(
            """
            Onyx.merge('nvp_onboarding', { hasCompletedGuidedSetupFlow: false });
            """
        )
        page.reload()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)


def create_workspace(page: Page):
    page.locator('button[aria-label="Workspaces"]').click()
    page.wait_for_timeout(2000)
    page.get_by_test_id("WorkspaceSwitcherPage").get_by_role("button", name="New workspace").click()
    page.wait_for_timeout(2000)
    page.locator('button[aria-label="Back"]').nth(0).click()
    page.wait_for_timeout(2000)
    page.locator('button[aria-label="Back"]').nth(0).click()
    page.wait_for_timeout(2000)


def browse_to_workspace(page: Page, workspace_name: str):
    page.locator('button[aria-label="Workspaces"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator(f'button[aria-label="{workspace_name}"]').click()
    except Exception:
        page.locator(f'button[aria-label="{workspace_name}"]').nth(1).click()
    page.wait_for_timeout(1000)
    page.locator('button[aria-label="Navigates to a chat"]', has_text=workspace_name).nth(0).click()
    page.wait_for_timeout(1000)


@pytest.mark.parametrize("setup_playwright", [{"width": 1280, "height": 720}], indirect=True)
def test_export_onyx_state(setup_playwright):
    """
    Verifies that the Onyx state is exported correctly when creating
    and navigating between workspaces.
    """
    page = setup_playwright
    login_user(page, PRIMARY_EMAIL, FIRST_NAME, LAST_NAME)
    create_workspace(page)
    create_workspace(page)

    workspace_name = f"{FIRST_NAME} {LAST_NAME}'s Workspace 1"
    browse_to_workspace(page, workspace_name)

    page.locator('button[aria-label="Start chat (Floating action)"]').click()
    page.locator('div[aria-label="Start chat"]').click()
    page.locator('button[aria-label="Room"]').click()

    workspace_new_room_div = page.query_selector('[data-testid="WorkspaceNewRoomPage"]')
    assert workspace_name in workspace_new_room_div.inner_text()
