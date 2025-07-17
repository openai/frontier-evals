import os
import re
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import time

from playwright.sync_api import TimeoutError, expect, sync_playwright

EXPENSIFY_URL = "https://dev.new.expensify.com:8082/"
EMAIL = "t1533148@gmail.com"
PASSWORD = "logb hzkg pkfk kskg"


def generate_random_email():
    int(time.time())
    return "t1533148+173350934@gmail.com"


def login_user(page, email, first_name="John", last_name="Doe"):
    """
    Log into the Expensify app.
    """
    page.goto(EXPENSIFY_URL, timeout=60000)
    page.get_by_test_id("username").fill(email)
    page.get_by_role("button", name="Continue").click()
    try:
        expect(page.get_by_test_id("SignInPage").get_by_test_id("validateCode")).to_be_visible(
            timeout=5000
        )
    except AssertionError:
        page.get_by_test_id("SignInPage").get_by_role("button", name="Join").click()
    else:
        otp_code = "123456"
        page.get_by_test_id("SignInPage").get_by_test_id("validateCode").fill(otp_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
        page.get_by_test_id("SignInPage").get_by_role("button", name="Sign in").click()
    try:
        expect(page.get_by_text("What do you want to do today?")).to_be_visible(timeout=5000)
    except AssertionError:
        pass
    else:
        page.get_by_label("Track and budget expenses").click()
        page.get_by_role("textbox", name="First name").fill(first_name)
        page.get_by_role("textbox", name="Last name").fill(last_name)
        page.get_by_role("button", name="Continue").click()
        try:
            page.get_by_role("button", name="Back").first.click(timeout=3000)
        except (AssertionError, TimeoutError):
            pass
    try:
        page.get_by_role("button", name="Close").click(timeout=3000)
    except (AssertionError, TimeoutError):
        pass
    page.wait_for_timeout(5000)


def test_user_current_location_is_not_shown_under_recent_destinations_for_distance_expense_request(
    pytestconfig,
):
    """
    Verify that user's current location is not shown under recent destinations while creating a distance expense request.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            slow_mo=500,
            proxy={"server": "http://127.0.0.1:8080/"},
            args=[
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

        # Prepare context with geolocation
        geolocation = {"longitude": 41.890221, "latitude": 12.492348}
        context_args = {
            "geolocation": geolocation,
            "permissions": ["geolocation"],
            "ignore_https_errors": True,
        }
        context = browser.new_context(**context_args)

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)  # Installed online guard
            page.wait_for_timeout(1000)

            email = generate_random_email()
            first_name = "Fname"
            last_name = "Lname"
            login_user(page, email, first_name=first_name, last_name=last_name)

            page.get_by_role("button", name="My settings").click()
            page.get_by_test_id("InitialSettingsPage").get_by_role(
                "menuitem", name="Workspaces", exact=True
            ).click()
            page.get_by_test_id("WorkspacesListPage").get_by_role(
                "button", name="New workspace"
            ).first.click()

            texts = (
                page.get_by_test_id("WorkspacePageWithSections")
                .get_by_role("menuitem")
                .all_inner_texts()
            )
            workspace_name = texts[0].split("\n")[-1]

            page.get_by_test_id("WorkspaceInitialPage").get_by_role("button", name="Back").click()
            page.get_by_role("button", name="Inbox", exact=True).click()
            page.get_by_test_id("BaseSidebarScreen").get_by_text(workspace_name, exact=True).click()

            # Create distance expense
            page.get_by_test_id("report-actions-view-wrapper").get_by_role(
                "button", name="Create", exact=True
            ).click()
            page.get_by_role("menuitem", name="Submit expense", exact=True).click()
            page.get_by_test_id("IOURequestStartPage").get_by_role(
                "button", name="Distance", exact=True
            ).click()
            page.get_by_test_id("IOURequestStartPage").get_by_role(
                "menuitem", name="Start", exact=True
            ).click()
            page.wait_for_timeout(1000)

            # Use current location
            page.get_by_test_id("IOURequestStepWaypoint").get_by_label(
                "Use current location", exact=True
            ).click()
            page.wait_for_timeout(10000)

            # Enter destination
            page.get_by_test_id("IOURequestStartPage").get_by_role(
                "menuitem", name="Stop", exact=True
            ).click()
            page.wait_for_timeout(1000)
            page.get_by_test_id("IOURequestStepWaypoint").get_by_role("textbox").fill("Germany")
            page.wait_for_timeout(1000)
            page.get_by_test_id("IOURequestStepWaypoint").get_by_text("Germany", exact=True).nth(
                1
            ).click()
            page.wait_for_timeout(15000)
            page.get_by_test_id("IOURequestStartPage").get_by_role(
                "button", name="Next", exact=True
            ).last.click()
            page.get_by_test_id("IOURequestStepConfirmation").get_by_role(
                "button", name="Submit", exact=False
            ).click()

            # Reload and start new distance expense
            page.reload()
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(3000)
            page.get_by_test_id("report-actions-view-wrapper").get_by_role(
                "button", name="Create", exact=True
            ).click()
            page.get_by_role("menuitem", name="Submit expense", exact=True).click()
            page.get_by_test_id("IOURequestStartPage").get_by_role(
                "button", name="Distance", exact=True
            ).click()
            page.get_by_test_id("IOURequestStartPage").get_by_role(
                "menuitem", name="Start", exact=True
            ).click()

            # Assert current location is NOT under recent destinations
            expect(
                page.get_by_test_id("IOURequestStepWaypoint").get_by_text(
                    "Recent destinations", exact=True
                )
            ).to_be_visible(timeout=5000)
            page.wait_for_timeout(3000)

            current_location = f"Lat: {geolocation['latitude']} Lng: {geolocation['longitude']}"
            assert_message = "The user current location must not be shown under recent destinations"
            expect(
                page.get_by_test_id("IOURequestStepWaypoint").get_by_text(
                    current_location, exact=True
                ),
                assert_message,
            ).not_to_be_visible()

        finally:
            # Stop tracing & clean traces if enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Clean up browser resources
            context.close()
            browser.close()
