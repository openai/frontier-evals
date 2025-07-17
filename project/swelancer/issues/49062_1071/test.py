import os
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.async_api import expect
from utils.browser import close_browser, start_browser
from utils.login import complete_onboarding, sign_up
from utils.online_guard import install_online_guard_async

# Constants
EXPENSIFY_USERNAME_FIRSTNAME = "osaidtest"
EXPENSIFY_USERNAME_LASTNAME = "testaccount"
EXPENSIFY_USERNAME = f"{EXPENSIFY_USERNAME_FIRSTNAME} {EXPENSIFY_USERNAME_LASTNAME}"
EXPENSIFY_EMAIL_USERNAME = "osaidtest5"
EXPENSIFY_EMAIL = "osaidtest5@gmail.com"
NEW_DOT_URL = "https://dev.new.expensify.com:8082"
OLD_DOT_URL = "http://localhost:9000/"
EXPENSIFY_PASSWORD = "ntvd acej mmaa qqmd"
DATA_DIR = "browser_context"

EXPENSIFY_DERIVED_EMAIL = "osaidtest5+49062_7@gmail.com"

console_warning_detected = False


async def get_user_email(page):
    """
    Get User Email.
    """
    return await page.evaluate(
        """async () => {
        const credentials = await Onyx.get('credentials');
        const {login} = credentials;
        return login
    }"""
    )


def track_console_warning(msg):
    global console_warning_detected
    if (
        msg.type == "error"
        and "Warning: Function components cannot be given refs. Attempts to access this ref will fail. Did you mean to use React.forwardRef()?"
        in msg.text
    ):
        console_warning_detected = True


async def open_self_dm_page(page, email):
    await page.get_by_label("Start chat (Floating action)").click()
    await page.get_by_text("Start chat").click()
    await page.get_by_test_id("selection-list-text-input").fill(email)
    try:
        await page.get_by_label(f"{EXPENSIFY_USERNAME} (you)").filter(has_text=email).click()
    except Exception:
        await page.get_by_label(EXPENSIFY_DERIVED_EMAIL).click()


@pytest.mark.asyncio
async def test_console_warning_for_function_components_cannot_be_given_refs_should_not_appear_in_self_dm(
    pytestconfig,
):
    """
    Ensure that 'Function components cannot be given refs' console warning
    does not appear on the self-DM page.
    """
    global console_warning_detected
    console_warning_detected = False  # reset for this test run

    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    # Start browser
    context, page, playwright = await start_browser(
        persistent=False,
        data_dir=DATA_DIR,
        launch_args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    await install_online_guard_async(
        context, page
    )  # Install online guard immediately after context and page creation

    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    page.on("console", track_console_warning)

    try:
        derived_email = EXPENSIFY_DERIVED_EMAIL
        await sign_up(page, derived_email)
        try:
            await complete_onboarding(
                page,
                first_name=EXPENSIFY_USERNAME_FIRSTNAME,
                last_name=EXPENSIFY_USERNAME_LASTNAME,
            )
        except Exception:
            # If onboarding is already completed or fails for acceptable reasons,
            # continue with the test.
            pass

        # Open Self-DM page.
        await open_self_dm_page(page, derived_email)

        # Validate the absence of the console warning.
        assert not console_warning_detected, (
            "Console Warning 'Function components cannot be given refs' appears in self-dm."
        )

    finally:
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Close browser irrespective of test outcome
        await close_browser(context, page, playwright)
