import os
import re
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)
from playwright.sync_api import sync_playwright
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner


def generate_random_email1(base_email="testerbro7077@gmail.com"):
    timestamp = int("46344")
    email_user, domain = base_email.split("@")
    return f"{email_user}+{timestamp}@{domain}"


def login_user1(p, trace_enabled, first_name="Yagna", slow_mo=1000, last_name="Sai") -> tuple:
    user_email1 = generate_random_email1()
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    install_online_guard_sync(context, page)  # Install guard immediately after page creation
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    phone_or_email_input = page.locator('input[type="email"]')
    phone_or_email_input.fill(user_email1)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    validate_code_input = page.locator('input[name="validateCode"]').nth(0)
    validate_code_input.fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    return (browser, context, page, user_email1)


def generate_random_email2(base_email="testerbro7077@gmail.com"):
    timestamp = int("46345")
    email_user, domain = base_email.split("@")
    return f"{email_user}+{timestamp}@{domain}"


def login_user2(p, trace_enabled, first_name="Yagna", last_name="Sai") -> tuple:
    user_email2 = generate_random_email2()
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context()
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    install_online_guard_sync(context, page)  # Install guard immediately after page creation
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    phone_or_email_input = page.locator('input[type="email"]')
    phone_or_email_input.fill(user_email2)
    page.locator('button[tabindex="0"]').click()
    validate_code_input = page.locator('input[name="validateCode"]').nth(0)
    validate_code_input.fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass
    return (browser, context, page, user_email2)


def add_member(page):
    page.get_by_label("My settings").click()
    page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").get_by_text(
        "Workspaces"
    ).click()
    page.get_by_test_id("WorkspacesListPage").locator("button").filter(
        has_text="Testerbro7077+46345@gmail.com"
    ).get_by_label("More").click()
    assert not page.get_by_text("Delete Workspace").is_visible()
    page.mouse.click(0.5 * page.viewport_size["width"], 0.5 * page.viewport_size["height"])
    page.get_by_test_id("WorkspacesListPage").locator("button").filter(
        has_text="Testerbro7077+46345@gmail.com"
    ).click()
    assert not page.get_by_label("Delete").is_visible()


def test_run(
    pytestconfig,
    admin_first_name="Yagna",
    admin_last_name="Sai",
    member_first_name="Sai",
    member_last_name="Yagna",
):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        try:
            browser_member, context_member, page_member, user_email1 = login_user1(
                p, trace_enabled, member_first_name, member_last_name
            )
            browser_admin, context_admin, page_admin, admin_email = login_user2(
                p, trace_enabled, admin_first_name, admin_last_name
            )
            add_member(page_member)
        finally:
            if trace_enabled:
                if context_member:
                    trace_path1 = "/app/expensify/user_tool/output_browser1.zip"
                    os.makedirs(os.path.dirname(trace_path1), exist_ok=True)
                    context_member.tracing.stop(path=trace_path1)
                    trace_cleaner(trace_path1)
                if context_admin:
                    trace_path2 = "/app/expensify/user_tool/output_browser2.zip"
                    os.makedirs(os.path.dirname(trace_path2), exist_ok=True)
                    context_admin.tracing.stop(path=trace_path2)
                    trace_cleaner(trace_path2)

            if browser_member:
                browser_member.close()
            if browser_admin:
                browser_admin.close()
