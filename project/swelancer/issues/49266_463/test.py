import os
import sys
import uuid

import pytest
from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

# --------------------------------------------------------------------------- #
#  Path set-up                                                                #
# --------------------------------------------------------------------------- #
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #
SITE_URL = "https://dev.new.expensify.com:8082/"
PRIMARY_EMAIL = "expensifyolddot+1733631442@gmail.com"
FIRST_NAME = "Expenolddot"
LAST_NAME = "1733631442"


# --------------------------------------------------------------------------- #
#  Fixture                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def setup_playwright(request):
    """
    Starts Playwright, launches the browser and returns a context.
    Supports indirect parametrisation for viewport size.
    Cleans everything up once the test is finished.
    """
    viewport = request.param if hasattr(request, "param") else {}
    playwright = sync_playwright().start()

    browser = playwright.chromium.launch(
        channel="chrome",
        headless=True,
        slow_mo=500,
        proxy={"server": "http://localhost:8080"},
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    context_args = {"timezone_id": "Asia/Kathmandu"}
    if isinstance(viewport, dict) and viewport:
        context_args["viewport"] = viewport

    context = browser.new_context(**context_args)

    try:
        yield context
    finally:
        browser.close()
        playwright.stop()


# --------------------------------------------------------------------------- #
#  Helper functions                                                            #
# --------------------------------------------------------------------------- #
def login_user(page: Page, email: str):
    page.goto(SITE_URL, timeout=60_000)
    page.locator('input[type="email"]').fill(email)
    page.locator('button[tabindex="0"]').click()
    # “Join” button is present only for new users
    try:
        page.locator("button", has_text="Join").click()
    except Exception:
        pass


def complete_onboarding(page: Page, fname: str, lname: str):
    page.locator("text='Track and budget expenses'").click()
    page.locator('input[name="fname"]').fill(fname)
    page.locator('input[name="lname"]').fill(lname)
    page.get_by_role("button", name="Continue").last.click()


def is_connected(page: Page):
    text = page.locator('div[aria-label="NetSuite"][role="menuitem"]').last.inner_text()
    return "Last synced" in text


# --------------------------------------------------------------------------- #
#  Test                                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("setup_playwright", [{"width": 1280, "height": 360}], indirect=True)
def test_netsuite_integration(setup_playwright, pytestconfig):
    """
    Tests NetSuite integration flow.
    Adds a try/finally block to handle Playwright tracing when the
    `--user-tool-trace` flag is provided.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    context = setup_playwright

    # ----------------------------------------------------------------------- #
    #  Start tracing if requested                                             #
    # ----------------------------------------------------------------------- #
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    try:
        # ------------------------------- Main flow -------------------------- #
        page = context.new_page()
        install_online_guard_sync(context, page)
        login_user(page, PRIMARY_EMAIL)
        complete_onboarding(page, FIRST_NAME, LAST_NAME)

        page.locator('button[aria-label="Workspaces"]').click()
        page.get_by_test_id("WorkspaceSwitcherPage").get_by_role(
            "button", name="New workspace"
        ).click()

        # NetSuite connection wizard
        page.locator('div[aria-label="More features"]').click()
        page.locator('button[aria-label="Sync your chart of accounts and more."]').click()
        page.locator('div[aria-label="Accounting"]').click()
        page.locator('div[aria-label="NetSuite"]').locator("button", has_text="Connect").click()

        # Upgrade steps
        page.locator("button", has_text="Upgrade").last.click()
        page.locator("button", has_text="Got it, thanks").last.click()
        for _ in range(4):
            page.locator("button", has_text="Next").last.click()

        # Credentials
        page.locator('input[aria-label="NetSuite Account ID"]').last.fill("123")
        page.locator('input[aria-label="Token ID"]').last.fill("123")
        page.locator('input[aria-label="Token Secret"]').last.fill("123")
        page.locator("button", has_text="Confirm").last.click()

        # Allow some time for the request to complete
        page.wait_for_timeout(5000)
        page.reload()

        # ------------------------------------------------------------------- #
        #  Custom segment flow                                                #
        # ------------------------------------------------------------------- #
        new_page = context.new_page()
        install_online_guard_sync(context, new_page)
        new_page.goto(
            f"https://dev.new.expensify.com:8082/settings/workspaces/"
            f"{page.url.split('/')[-2]}/accounting/netsuite/import/custom-segment/new"
        )

        new_page.locator('button[aria-label="Custom segment"]').last.click()
        new_page.locator("button", has_text="Next").last.click()
        new_page.locator('input[aria-label="Name"]').last.fill("abc")
        new_page.locator("button", has_text="Next").last.click()
        new_page.locator("button", has_text="Next").last.click()
        new_page.locator('button[aria-label="Back"]').last.click()

        # Validation
        rhn_text = new_page.locator(
            'div[data-testid="NetSuiteImportAddCustomSegmentPage"]'
        ).last.inner_text()

        assert "Please fix the errors in the form before continuing." not in rhn_text, (
            "Error message did not disappear!"
        )

    finally:
        # ------------------------------------------------------------------- #
        #  Stop tracing & cleanup                                             #
        # ------------------------------------------------------------------- #
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)
