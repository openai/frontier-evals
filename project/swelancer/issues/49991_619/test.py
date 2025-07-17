import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright

static_email = "totherandomuser+40121580@gmail.com"


def login(
    p,
    if_phone: bool = False,
    first_name: str = "John",
    last_name: str = "Cena",
    trace_enabled: bool = False,
):
    """
    Starts a browser, creates a context (optionally mobile), optionally starts tracing,
    performs login / onboarding flow and returns the browser, context and page.
    """
    browser = p.chromium.launch(
        headless=True,
        slow_mo=1000,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )

    # Create context (mobile or desktop)
    if if_phone:
        phone = p.devices["iPhone 12 Pro"]
        context = browser.new_context(**phone)
    else:
        context = browser.new_context()

    # Start tracing immediately after context creation if requested
    if trace_enabled:
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page = context.new_page()
    install_online_guard_sync(
        context, page
    )  # Install the online guard immediately after page creation
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)

    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()
    phone_or_email_input.fill(static_email)
    page.get_by_role("button", name="Continue").click()
    page.get_by_role("button", name="Join").click()
    page.get_by_role("button", name="Join").wait_for(state="hidden")
    expect(page.locator("text=What do you want to do today?")).to_be_visible(timeout=50000)

    # If this is the first time user reaches onboarding, fill name details
    try:
        page.locator("text='Track and budget expenses'").click()
        page.wait_for_timeout(1000)
        first_name_input = page.locator('input[name="fname"]')
        expect(first_name_input).to_be_visible()
        first_name_input.fill(first_name)

        last_name_input = page.locator('input[name="lname"]')
        expect(last_name_input).to_be_visible()
        last_name_input.fill(last_name)

        continue_button = page.locator(
            'button[data-tag="pressable"][tabindex="0"]', has_text="Continue"
        )
        expect(continue_button).to_be_visible()
        continue_button.click()
    except Exception:
        # Onboarding already completed – just proceed
        pass

    return browser, context, page


def _stop_and_clean_trace(context, trace_enabled: bool):
    """
    Helper to stop tracing on a context (if enabled) and clean the produced file.
    """
    if not trace_enabled or context is None:
        return

    # Ensure output directory exists
    trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)

    context.tracing.stop(path=trace_path)
    trace_cleaner(trace_path)


def test_invite_member_and_delete_task(pytestconfig):
    """
    End-to-end test that invites a member to a workspace,
    creates a task, deletes it and makes sure `[Deleted task]`
    text does not persist.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser = None
        context = None
        try:
            # Login flow (mobile context) – tracing is started inside if enabled
            browser, context, page = login(p, if_phone=True, trace_enabled=trace_enabled)

            # Begin the actual test steps
            page.get_by_label("Back").last.click()
            page.get_by_label("Close").last.click()

            plus_icon = page.locator('button[aria-label="Start chat (Floating action)"]')
            expect(plus_icon).to_be_visible()
            plus_icon.click()

            new_workspace_button = page.locator('div[aria-label="New workspace"]')
            expect(new_workspace_button).to_be_visible()
            new_workspace_button.click()

            page.get_by_text("Members").click()
            page.get_by_role("button", name="Invite member").click()
            page.get_by_test_id("selection-list-text-input").click()
            page.get_by_test_id("selection-list-text-input").fill("totherandomuser@gmail.com")
            page.wait_for_timeout(2000)
            page.get_by_text("totherandomuser@gmail.com").last.click()
            page.get_by_role("button", name="Next").click()
            page.get_by_test_id("WorkspaceInviteMessagePage").get_by_role(
                "button", name="Invite"
            ).click()

            # Navigate back to main screen
            page.get_by_test_id("WorkspaceMembersPage").get_by_label("Back").last.click()
            page.get_by_label("Back").last.click()
            page.wait_for_timeout(1000)
            page.get_by_label("Back").last.click()
            page.wait_for_timeout(1000)
            page.get_by_label("Inbox").click()
            page.wait_for_timeout(1000)

            # Select chat with the invited user
            user_chat = page.locator(
                'button[aria-label="Navigates to a chat"]', has_text="Deepak Dhaka"
            ).last
            expect(user_chat).to_be_visible(timeout=5000)
            user_chat.click()

            # Create a task
            page.get_by_label("Create").click()
            page.get_by_text("Assign task").click()
            page.get_by_role("textbox", name="Title").click()
            page.get_by_role("textbox", name="Title").fill("task1")
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Next").click()
            page.get_by_role("button", name="Confirm task").click()
            page.wait_for_timeout(1000)
            page.get_by_role("button", name="Task").last.click()
            page.wait_for_timeout(1000)

            # Delete the task
            task_container = page.locator(
                'div[data-testid="DisplayNamesWithTooltip"]', has_text="task1"
            )
            expect(task_container).to_be_visible(timeout=5000)
            task_container.click()
            page.locator(
                '[data-selection-scrapper-hidden-element="false"]', has_text="Delete"
            ).click()
            page.locator('button[data-tag="pressable"]', has_text="Delete").click()

            # Verify `[Deleted task]` does NOT appear
            delete_detected = False
            count_f = 0
            for _ in range(50):
                count = page.locator(
                    '[data-testid="DisplayNamesWithTooltip"]',
                    has_text="[Deleted task]",
                ).count()
                if count > 0:
                    print(f"Detected '[Deleted task]' {count} time(s) in page content!")
                    count_f += count
                    delete_detected = True
                page.wait_for_timeout(10)

            if not delete_detected:
                print("Delete text was not detected after retries.")

            assert count_f == 0, (
                "[Deleted task] text has been found, It should not be there ideally!!"
            )

        finally:
            # Stop tracing (if enabled) and clean
            _stop_and_clean_trace(context, trace_enabled)

            # Close context if it was started
            if context is not None:
                context.close()

            # Close browser if it was started
            if browser is not None:
                browser.close()
