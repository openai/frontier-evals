import os
import re
import uuid

from playwright.sync_api import sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1+34@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    page.locator('input[data-testid="validateCode"]').fill("123456")
    try:
        page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass


def create_workspace(page):
    page.get_by_label("My settings").click()
    page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").click()
    page.get_by_label("New workspace").click()
    page.get_by_label("Back").click()


def emoji_size(page):
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").click()
    page.get_by_test_id("selection-list-text-input").fill("pehlajrajput34@gmail.com")
    page.wait_for_timeout(1000)
    page.get_by_label("pehlajrajput34@gmail.com").click()
    page.wait_for_timeout(1000)
    page.locator("div").filter(has_text=re.compile("^Heyy$")).nth(4).click()
    page.get_by_label(":joy:").click()
    emoji_locator = page.locator("div").filter(has_text=re.compile("^😂1$")).nth(3)
    page.wait_for_timeout(1000)
    emoji_box = emoji_locator.bounding_box()
    print(f"Emoji Size - Width: {emoji_box['width']} px, Height: {emoji_box['height']} px")
    assert 26 <= emoji_box["width"] <= 45, (
        f"Emoji width should be around 28px, but was {emoji_box['width']}px"
    )
    assert 26 <= emoji_box["height"] <= 45, (
        f"Emoji height should be around 28px, but was {emoji_box['height']}px"
    )


def test_expensify_0000(pytestconfig):
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
            slow_mo=1000,
            channel="chromium",
        )
        context = browser.new_context()

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.goto(NEWDOT_URL, timeout=60000)
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL)
            page.wait_for_timeout(2000)
            emoji_size(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            browser.close()
