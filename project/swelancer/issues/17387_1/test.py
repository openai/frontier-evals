import os
import re
import uuid

from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

EMAIL = "velmoner+960@gmail.com"
PASSWORD = "aory ezrb qkmi qkas"
NEWDOT_URL = "https://dev.new.expensify.com:8082/"
first_name = "aa"
last_name = "bb"


def user_login(page: Page, email_id: str):
    page.goto(NEWDOT_URL)
    page.locator('input[type="email"]').fill(email_id)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    otp = "123456"
    page.locator('input[data-testid="validateCode"]').fill(otp)
    try:
        page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass
    try:
        page.get_by_role("button", name="Sign In").click()
    except Exception:
        # In some flows the Sign In button might not be present; ignore if not found
        pass


def test_expensify_17387(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            proxy={"server": "http://localhost:8080"},
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            slow_mo=None,
            channel="chromium",
        )

        context = browser.new_context()
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            invalid_email = (
                "jaj@asjjssjdjdjdjdjdjjeiwiwiwowkdjdjdieikdjfidekjcjdkekejdcjdkeekcjcdidjjcdkek"
                "diccjdkejdjcjxisdjjdkedncicdjejejcckdsijcjdsodjcicdkejdicdjejajasjjssjdjdjdjdjdjje"
                "iwiwiwowkdjdjdieikdjfisjksksjsjssskssjskskssksksksksskdkddkddkdksskskdkdkdkssksksk"
                "dkdkdkdkekeekdkddenejeodxkdndekkdjddkeemdjxkdenendkdjddekjcjdkekejdcjdkeekcjcdidjjc"
                "dkekdiccjdkejdjcjxisdjjdkedncicdjejejcckdsijcjdsodjcicdkejdi.cdjd"
            )

            page = context.new_page()
            install_online_guard_sync(context, page)
            page.goto(NEWDOT_URL)

            # First edge case, checking for email validation on login
            page.get_by_test_id("username").fill(invalid_email)
            page.get_by_role("button", name="Continue").click()
            assert page.get_by_text("The email entered is invalid.").is_visible()

            # Valid login flow
            user_login(page=page, email_id=EMAIL)
            page.wait_for_timeout(2000)

            # Second edge case, checking for email validation in start chat flow
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_role("menuitem", name="Start chat").click()
            page.wait_for_timeout(500)
            page.get_by_role("textbox", name="Name, email, or phone number").fill(invalid_email)
            page.wait_for_timeout(500)  # intentional delay to allow validation to complete
            assert page.get_by_text("Invalid email").is_visible()

            # Third edge case, checking for email validation in new contact method
            page.get_by_label("back").click()
            page.get_by_label("Settings").click()
            page.get_by_role("menuitem", name="Profile").click()
            page.get_by_text("Contact method").click()
            page.get_by_role("button", name="New contact method").click()
            page.get_by_role("textbox", name="Email/Phone number").fill(invalid_email)
            page.get_by_role("button", name="Add").click()
            assert page.get_by_text("Invalid contact method").is_visible()

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
