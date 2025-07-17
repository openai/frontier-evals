import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import Page, ViewportSize, expect, sync_playwright

EMAIL = "expensify66+6548748765433279763@gmail.com"
FIRST_NAME = "Asaad"
LAST_NAME = "I"


def login_user(page: Page):
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.get_by_test_id("username").fill(EMAIL)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()


def onboarding_user(page: Page):
    page.get_by_text("What do you want to do today?").wait_for()
    activity_text = "Track and budget expenses"
    page.get_by_label(activity_text).click()
    page.get_by_role("textbox", name="First name").fill(FIRST_NAME)
    page.get_by_role("textbox", name="Last name").fill(LAST_NAME)
    page.get_by_role("button", name="Continue").last.click()


def create_workspace_with_tags(page: Page):
    page.get_by_label("Start chat (Floating action)").click()
    page.get_by_label("New workspace").click()
    page.get_by_label("More features").click()
    page.get_by_label("Classify costs and track").click()
    page.get_by_label("Tags").click()
    for i in range(1, 21):
        page.get_by_role("button", name="Add tag").click()
        page.get_by_role("textbox", name="Name").fill(f"{i}")
        page.get_by_role("button", name="Save").click()
    page.get_by_test_id("WorkspaceInitialPage").get_by_label("Back").click()
    row_content = page.get_by_label("row").first.text_content()
    workspace_name = row_content.split(".default-", 1)[0]
    return workspace_name


def go_to_select_tag(page: Page, workspace_name: str):
    page.get_by_label("Inbox").click()
    page.get_by_label("Navigates to a chat").get_by_text(workspace_name, exact=True).click()
    page.get_by_label("Create").locator("visible=true").click()
    page.get_by_label("Submit expense").click()
    page.get_by_label("Manual").click()
    page.get_by_placeholder("0").fill("123")
    page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
    page.get_by_label("Show more").click()
    page.get_by_role("menuitem", name="Tag").click()


def test_popup(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(viewport=ViewportSize(width=1280, height=720))

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard

            login_user(page)
            onboarding_user(page)
            workspace_name = create_workspace_with_tags(page)
            go_to_select_tag(page, workspace_name)
            page.get_by_test_id("selection-list-text-input").fill("rabbit")
            expect(page.get_by_text("No results found")).to_be_visible()
            page.close()
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            browser.close()
