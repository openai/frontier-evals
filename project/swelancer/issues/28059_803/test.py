import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager


def generate_random_email():
    return "turingtemp+826403@gmail.com"


def login_user(
    p: PlaywrightContextManager, first_name="Milan", last_name="T"
) -> tuple[Browser, Page, str]:
    """
    Launches the browser, creates a page, signs-up / logs-in a new user
    and returns the browser, the page and the newly generated email.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard

    user_email1 = generate_random_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email1)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
        page.wait_for_timeout(1000)
    except Exception:
        pass
    try:
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.wait_for_timeout(1000)
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
    except Exception:
        pass
    page.wait_for_timeout(1000)
    return (browser, page, user_email1)


def test_save_description_in_submitted_expense(tmp_path, pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        # ------------------------------------------------------------------ #
        #  Create browser / context / page
        # ------------------------------------------------------------------ #
        browser, page, user_email = login_user(p)
        context = page.context
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            # ------------------------------------------------------------------ #
            #  Original test steps
            # ------------------------------------------------------------------ #
            page.wait_for_timeout(1000)
            page.locator('div[placeholder="Write something..."]').last.fill("yo!")
            page.wait_for_timeout(1000)
            page.keyboard.press("Enter")
            page.wait_for_timeout(1000)
            page.locator('div[aria-label="Chat message"]', has_text="yo!").last.click(
                button="right"
            )
            page.wait_for_timeout(1000)
            page.locator('button[aria-label="👍"]').last.click()
            page.route("**/*", lambda route: route.abort())
            page.wait_for_timeout(3000)
            page.locator('div[aria-label="Chat message"]', has_text="yo!").last.click(
                button="right"
            )
            page.wait_for_timeout(1000)
            page.locator('button[aria-label="😂"]').last.click()
            page.wait_for_timeout(1000)
            laugh_emoji_div = page.locator(
                'div[class="css-view-175oi2r"][style="opacity: 0.5;"]', has_text="😂"
            )
            page.wait_for_timeout(1000)
            expect(laugh_emoji_div).to_be_visible()

        finally:
            # ------------------------------------------------------------------ #
            #  Stop tracing (if enabled) and clean-up
            # ------------------------------------------------------------------ #
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
