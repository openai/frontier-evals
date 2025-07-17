import os
import re
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import sync_playwright

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1+345@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
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
        # If the Sign In button is not available (already signed in), ignore.
        pass


def copy_fragment(context, html):
    """Open a temp tab containing `html`, copy all, close."""
    tab = context.new_page()
    tab.set_content(f"<body contenteditable='true'>{html}</body>")
    tab.locator("body").click()  # Ensure body is focused
    tab.keyboard.press("ControlOrMeta+A")
    tab.keyboard.press("ControlOrMeta+C")
    tab.close()


def composer_formatting(page):
    ctx = page.context
    copy_fragment(ctx, "<p>Normal</p>")

    page.get_by_role("textbox").click()
    page.keyboard.press("ControlOrMeta+v")
    page.keyboard.press("Enter")
    normal = page.get_by_text("Normal").last

    assert "*" not in normal.text_content()
    copy_fragment(ctx, "<p><b>Bold</b></p>")

    page.get_by_role("textbox").click()
    page.keyboard.press("ControlOrMeta+v")
    page.keyboard.press("Enter")
    bold = page.get_by_text("Bold").last

    assert "**" not in bold.text_content()


def test_expensify_0000(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=False,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(ignore_https_errors=True)

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL)
            page.wait_for_timeout(2000)
            composer_formatting(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            context.close()
            browser.close()
