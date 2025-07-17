import os
import sys
import uuid

from playwright.sync_api import BrowserContext
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import Browser, Page, expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email() -> str:
    # Keeping the exact behaviour of original function (no real randomness).
    return "testerbro7077+879478@gmail.com"


def login_user(
    p: PlaywrightContextManager,
    trace_enabled: bool,
    first_name: str = "Milan",
    last_name: str = "T",
) -> tuple[Browser, BrowserContext, Page, str]:
    """
    Launches the browser, (optionally) starts tracing,
    signs the user up and returns useful playwright objects.
    """
    browser: Browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        slow_mo=500,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    # Create a dedicated context so we can start/stop tracing on it.
    context: BrowserContext = browser.new_context()

    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page: Page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation

    user_email = generate_random_email()

    # Begin original sign-up / login flow
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass

    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    page.wait_for_timeout(1000)
    # End original sign-up / login flow

    return browser, context, page, user_email


def _export_and_clean_trace(context: BrowserContext) -> None:
    """
    Helper that stops tracing on the given context,
    exports it to a unique zip file and cleans it.
    """
    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_add_reaction_to_chat_message_with_tab_key(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser: Browser | None = None
        context: BrowserContext | None = None
        try:
            # Sign-in / signup flow + (maybe) tracing start
            browser, context, page, user_email = login_user(p, trace_enabled=trace_enabled)

            # ----------------------- Original Test Steps -----------------------
            personal_chat = page.get_by_text("Milan T (you)").nth(0)
            expect(personal_chat).to_be_visible()

            personal_chat.click()

            create_btn = page.locator('button[aria-label="Create"]').nth(1)
            expect(create_btn).to_be_visible()
            create_btn.click()

            task_btn = page.get_by_text("Assign task").nth(0)
            expect(task_btn).to_be_visible()
            task_btn.click()

            page.wait_for_timeout(1000)

            page.locator('input[aria-label="Title"]').nth(0).fill(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            )

            next_button = page.locator("button", has_text="Next").nth(0)
            expect(next_button).to_be_visible()
            next_button.click()

            length_warning = page.get_by_text("Character limit exceeded").nth(0)
            expect(length_warning).to_be_visible()

            page.wait_for_timeout(2000)
            # --------------------------------------------------------------------

        finally:
            # Export & clean traces (if needed)
            if trace_enabled and context is not None:
                _export_and_clean_trace(context)

            # Close context and browser no matter what
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
