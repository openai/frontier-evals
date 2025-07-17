import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright._impl._browser import Browser
from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from utils.online_guard import install_online_guard_sync


def generate_random_email():
    return "turingtemp+67867@gmail.com"


def login_user(
    p: PlaywrightContextManager, first_name="Ayush", last_name="G"
) -> tuple[Browser, Page, str]:
    """
    Launches a browser, signs up a new user and returns the browser, the active page
    and the generated email for the user.
    """
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        slow_mo=800,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    page = context.new_page()
    install_online_guard_sync(context, page)
    user_email = generate_random_email()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    try:
        page.locator('button[tabindex="0"]').click()
    except Exception:
        print("Join button not found")
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    return (browser, page, user_email)


def test_save_description_in_submitted_expense(pytestconfig):
    """
    Adds a secondary contact method, verifies the Spanish translation for the
    confirmation message and ensures the workflow behaves as expected.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Sign-up / log-in flow
            browser, page, user_email = login_user(p)
            context = page.context

            # Start tracing right after the context is available
            if trace_enabled:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # === Test body ===
            page.get_by_label("My settings").click()
            page.get_by_text("Contact method").click()
            page.get_by_role("button", name="New contact method").click()
            page.get_by_role("textbox", name="Email/Phone number").fill(
                "turingtemp+89116@gmail.com"
            )
            page.get_by_role("button", name="Add").click()
            page.get_by_test_id("ValidateCodeActionModal").get_by_role("textbox").fill("124435")
            try:
                page.get_by_role(
                    "button",
                    name=re.compile(r"^(continue|sign.?in)$", re.I),
                ).click(timeout=20000)
            except Exception:
                pass

            try:
                page.locator("div").filter(has_text=re.compile("^Verify$")).nth(1).click()
            except Exception:
                print("Verification modal not found")

            page.locator(
                'button[aria-label="Back"][type="button"][role="button"][data-tag="pressable"]'
            ).nth(-1).click()
            page.get_by_text("Sign out").click()

            # Log in as another user and change language
            page.get_by_test_id("username").fill("turingtemp+89117@gmail.com")
            page.get_by_role("button", name="Continue").click()
            page.get_by_role("button", name="Unlink").click()
            page.get_by_test_id("web_picker").select_option("es")
            spanish_link_sent = page.get_by_text("¡Enlace enviado!")
            expect(spanish_link_sent).to_be_visible()
        finally:
            # === Tracing teardown ===
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close the browser regardless of test outcome
            if context is not None:
                context.close()
            if browser is not None:
                browser.close()
