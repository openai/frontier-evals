import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

static_email = "testuserexpensify1+44@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    first_name: str = "Milan",
    last_name: str = "T",
    trace_enabled: bool = False,
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Helper that launches a browser, creates a context (+page),
    performs the newDot login flow, and returns all three handles
    along with the user email being used.
    """
    browser: Browser = p.chromium.launch(
        headless=True,
        slow_mo=1000,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    context: BrowserContext = browser.new_context()
    if trace_enabled:
        # Immediately start tracing for this context
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)  # Installed online guard

    user_email = static_email
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)

    # Sometimes a second “Continue” appears – ignore failures
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)

    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)

    return browser, context, page, user_email


def test_close_account_bottom_margin(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    # Pre-declare variables so they are visible in finally
    browser = context = page = None

    with sync_playwright() as p:
        try:
            first_name = "John"
            last_name = "Doe"
            new_workspace_name = "Testing workspace"

            # Login (this will also start tracing inside if enabled)
            browser, context, page, user_email = login_user(p, first_name, last_name, trace_enabled)

            # Test workflow begins
            profile = page.locator('button[aria-label="My settings"]')
            profile.click()
            page.wait_for_timeout(1000)

            troubleshoot = page.locator('div[role="menuitem"]').nth(9)
            troubleshoot.click()
            page.wait_for_timeout(1000)

            forced_offline = page.locator('button[aria-label="Force offline"]')
            forced_offline.click()
            page.wait_for_timeout(1000)

            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            plus_icon.click()
            page.wait_for_timeout(2000)

            new_workspace_button = page.locator('div[aria-label="New workspace"]')
            new_workspace_button.click()
            page.wait_for_timeout(1000)

            workspace_name = f"{first_name} {last_name}'s Workspace"
            profile = page.locator(f'div[aria-label="{workspace_name}"]')
            profile.click()
            page.wait_for_timeout(1000)

            page.locator('input[aria-label="Name"]').fill(new_workspace_name)
            save_button = page.locator("button", has_text="Save")
            save_button.click()
            page.wait_for_timeout(1000)

            new_profile = page.locator(f'div[aria-label="{new_workspace_name}"]')
            page.wait_for_timeout(1000)

            profile_opacity = new_profile.evaluate(
                "element => window.getComputedStyle(element.parentElement.parentElement.parentElement.parentElement).opacity"
            )

            assert profile_opacity == "0.5", "Profile is not grayed out (expected opacity 0.5)."

            page.wait_for_timeout(1000)

        finally:
            # Stop tracing & clean the trace, if enabled
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close context before closing the browser
            if context is not None:
                context.close()

            # Always attempt to close the browser
            if browser is not None:
                browser.close()
