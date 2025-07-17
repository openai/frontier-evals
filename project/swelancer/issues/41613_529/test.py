import os
import sys
import uuid

import pytest
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import Page, expect, sync_playwright


def create_user(page: Page, firstname: str, lastname: str, email: str):
    page.get_by_test_id("username").fill(email)
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Continue").click()
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Join").click()
    page.get_by_text("Track and budget expenses").click()
    page.get_by_role("textbox", name="First name").fill(firstname)
    page.get_by_role("textbox", name="Last name").fill(lastname)
    page.wait_for_timeout(1000)
    page.get_by_role("button", name="Continue").click()
    return email


def _start_tracing_if_enabled(context, trace_enabled: bool):
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)


@pytest.mark.order(1)
def test_task_preview(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    # Containers to keep track of created contexts and pages
    contexts = []
    pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            slow_mo=1000,
            proxy={"server": "http://127.0.0.1:8080/"},
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        context_args = {
            "geolocation": {"latitude": 37.7749, "longitude": 122.4194},
            "permissions": ["geolocation"],
            "viewport": {"width": 1280, "height": 800},
            "user_agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/94.0.4606.61 Safari/537.36"
            ),
            "ignore_https_errors": True,
        }

        # First context / page
        context = browser.new_context(**context_args)
        contexts.append(context)
        _start_tracing_if_enabled(context, trace_enabled)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed guard
            pages.append(page)
            page.wait_for_timeout(5000)
            page.goto("https://127.0.0.1:8082/", timeout=60000)

            create_user(page, "Test", "User1", "t1533148+1734358702@gmail.com")

            # Second context / page
            context1 = browser.new_context(**context_args)
            contexts.append(context1)
            _start_tracing_if_enabled(context1, trace_enabled)

            page1 = context1.new_page()
            install_online_guard_sync(context1, page1)  # Installed guard
            pages.append(page1)
            page1.wait_for_timeout(5000)
            page1.goto("https://127.0.0.1:8082/")
            email2 = create_user(page1, "Test", "User2", "t1533148+1734358725@gmail.com")

            # Workspace and member management
            page.wait_for_timeout(2000)
            page.get_by_label("My settings").click()
            page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").click()
            page.get_by_label("New workspace").last.click()
            page.get_by_label("Members").click()
            page.get_by_role("button", name="Invite member").click()
            page.get_by_test_id("selection-list-text-input").fill(email2)
            page.get_by_label("Test User2").last.click()
            page.get_by_role("button", name="Next").click()
            page.get_by_test_id("WorkspaceInviteMessagePage").get_by_role(
                "button", name="Invite"
            ).click()

            # Configure spend workflows
            page.get_by_label("More features").click()
            page.get_by_label("Configure how spend is").click()
            page.get_by_label("Workflows").click()
            page.get_by_label("Require additional approval").click()
            page.get_by_label("Back").click()
            page.get_by_label("Inbox").click()

            # User2 performs expense submission
            page1.wait_for_timeout(2000)
            page1.reload()
            page1.locator(
                'button[aria-label="Navigates to a chat"]',
                has_text="Test User1's Workspace",
            ).last.click()
            page1.get_by_label("Create").nth(-1).click()
            page1.get_by_label("Submit expense").click()
            page1.get_by_label("Manual").click()
            page1.get_by_placeholder("0").fill("100")
            page1.locator("#numPadContainerView").get_by_role("button", name="Next").click()
            page1.locator('div[role="menuitem"]', has_text="Merchant").click()
            page1.locator('input[aria-label="Merchant"]').fill("Test Merchant")
            page1.locator("button", has_text="Save").click()
            page1.locator("button", has_text="Submit ₹").last.click()
            page1.get_by_label("View details").click()
            page1.wait_for_timeout(2000)

            # Hold the expense
            current_page_url = page1.url
            page1.goto(f"{current_page_url}/details")
            page1.get_by_role("button", name="Hold").click()
            page1.get_by_role("textbox", name="Reason").fill("NA")
            page1.get_by_role("button", name="Hold expense").click()
            page1.get_by_role("button", name="Got it").click()

            # User1 reviews the expense
            page.wait_for_timeout(2000)
            page.reload()
            page.locator(
                'button[aria-label="Navigates to a chat"]', has_text="Test User2"
            ).last.click()
            page.wait_for_timeout(2000)
            page.get_by_role("button", name="Approve").click()
            page.wait_for_timeout(5000)

            expect(
                page.locator("text=Approve what's not on hold, or approve the entire report.")
            ).not_to_be_visible()

        finally:
            # Stop tracing and clean traces
            if trace_enabled:
                for ctx in contexts:
                    try:
                        trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                        ctx.tracing.stop(path=trace_path)
                        trace_cleaner(trace_path)
                    except Exception:
                        # If tracing wasn't started or already stopped, continue
                        pass

            # Close pages
            for pg in pages:
                try:
                    pg.close()
                except Exception:
                    pass

            # Close contexts
            for ctx in contexts:
                try:
                    ctx.close()
                except Exception:
                    pass

            # Close the browser
            try:
                browser.close()
            except Exception:
                pass
