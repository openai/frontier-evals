import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright.sync_api import Page, sync_playwright
from utils.online_guard import install_online_guard_sync

first_name = "Alice"
last_name = "Aliceoglu"
workspace_name = f"{first_name} {last_name}'s Workspace"


def generate_random_email():
    # NOTE: the original implementation returned a constant address;
    # we keep behaviour unchanged to avoid altering test logic.
    int(time.time())
    return "aliceoglualice+1733406917@gmail.com"


def login_user(page: Page) -> None:
    username_input = page.get_by_test_id("username")
    username_input.click()
    username_input.fill(generate_random_email())
    continue_button = page.get_by_role("button", name="Continue")
    continue_button.click()
    join_button = page.get_by_role("button", name="Join")
    join_button.click()
    track_expenses_text = page.get_by_text("Track and budget expenses")
    track_expenses_text.click()


def enter_user_information(page: Page) -> None:
    first_name_input = page.get_by_role("textbox", name="First name")
    first_name_input.fill(first_name)
    first_name_input.press("Tab")
    last_name_input = page.get_by_role("textbox", name="Last name")
    last_name_input.fill(last_name)
    continue_button_after_names = page.get_by_role("button", name="Continue")
    continue_button_after_names.click()


def reproduce_scenario(page: Page) -> None:
    page.get_by_label("Start chat (Floating action)").click()
    page.get_by_text("New workspace").click()
    page.get_by_text("More features").click()
    page.get_by_label("Add, update, and enforce").click()
    page.get_by_label("Back").click()
    page.get_by_label("Inbox").click()
    page.get_by_label("Start chat (Floating action)").click()
    page.locator("body").press("ControlOrMeta+d")
    page.get_by_label("Force offline").click()
    page.get_by_label("Force offline").press("Escape")
    page.get_by_label("Start chat (Floating action)").click()
    page.get_by_text("Submit expense", exact=True).click()
    page.get_by_label("Distance").click()
    page.get_by_text("ManualManualScanScanDistanceDistance").click()
    page.get_by_label("Distance").click()
    page.get_by_test_id("IOURequestStartPage").get_by_role("button", name="Start").click()
    page.get_by_test_id("IOURequestStepWaypoint").get_by_role("textbox").fill("fakeAddress1")
    page.get_by_role("button", name="Save").click()
    page.get_by_role("button", name="Stop").click()
    page.get_by_test_id("IOURequestStepWaypoint").get_by_role("textbox").fill("fakeAddress2")
    page.get_by_role("button", name="Save").click()
    page.get_by_role("button", name="Next").nth(1).click()
    page.get_by_test_id("selection-list").get_by_label("Alice Aliceoglu's Workspace").click()
    page.get_by_role("button", name="Submit expense").click()
    page.locator("body").press("ControlOrMeta+d")
    page.get_by_label("Force offline").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Force offline").press("Escape")
    page.wait_for_timeout(1000)
    page.get_by_label("View details").click()
    page.get_by_label("Start chat (Floating action)").click()
    page.get_by_text("Submit expense", exact=True).click()
    page.get_by_test_id("IOURequestStartPage").get_by_role("button", name="Start").click()
    page.get_by_role("textbox").click()
    page.wait_for_timeout(1000)
    assert not page.get_by_text("fakeAddress1").is_visible(), (
        "The 'fakeAddress1' is unexpectedly visible"
    )


def test_room_name_is_highlighted(pytestconfig) -> None:
    """
    Re-implemented to add optional Playwright tracing when the
    '--user-tool-trace' CLI flag is provided.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            slow_mo=1000,
            args=[
                "--disable-web-security",
                "--disable-feature=IsolaterOrigins,site-per-process",
            ],
        )
        context = browser.new_context(viewport={"width": 1000, "height": 550})

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(
                context, page
            )  # Install the online guard immediately after page creation
            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
            login_user(page)
            enter_user_information(page)
            reproduce_scenario(page)

        finally:
            # Stop tracing & clean traces
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Always close context & browser
            context.close()
            browser.close()
