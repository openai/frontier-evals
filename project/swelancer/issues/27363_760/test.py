import os
import re
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)
from playwright.sync_api import (
    expect,
    sync_playwright,
)

GMAIL_NAME = "nomandevtur+27363"
GMAIL_APP_PASSWORD = "tpye xlsj axyq nltq"


def get_test_user_info(seed=None, first_name=None, last_name=None):
    if first_name is None:
        first_name = GMAIL_NAME
    if last_name is None:
        last_name = ""
    email = f"{GMAIL_NAME}+{seed}@gmail.com" if seed is not None else f"{GMAIL_NAME}@gmail.com"
    return {
        "email": email,
        "password": GMAIL_APP_PASSWORD,
        "first_name": first_name,
        "last_name": last_name,
    }


def select_activity(page, first_name, last_name, activity_text):
    expect(page.get_by_text("What do you want to do today?")).to_be_visible()
    page.get_by_label(activity_text).click()
    page.get_by_role("textbox", name="First name").fill(first_name)
    page.get_by_role("textbox", name="Last name").fill(last_name)
    page.get_by_role("button", name="Continue").last.click()


def login_user(page, user_info, activity_text="Track and budget expenses"):
    page.context.clear_cookies()
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.wait_for_load_state("load")
    try:
        expect(page.get_by_label("Inbox")).to_be_visible(timeout=3000)
        return
    except Exception:
        pass
    page.get_by_test_id("username").fill(user_info["email"])
    page.get_by_role("button", name="Continue").click()
    join_button = page.get_by_role("button", name="Join")
    validate_code_input = page.locator('input[data-testid="validateCode"]')
    if join_button.is_visible():
        join_button.click(timeout=3000)
    else:
        magic_code = "123456"
        print(f"Magic code: {magic_code}")
        validate_code_input.fill(magic_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
    page.wait_for_timeout(3000)
    select_activity_dialog = page.get_by_text("What do you want to do today?")
    if select_activity_dialog.count() > 0:
        select_activity(page, user_info["first_name"], user_info["last_name"], activity_text)


def launch_app(pw, headless=True, device=None, geolocation=None):
    browser = pw.chromium.launch(
        headless=headless,
        slow_mo=500,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context_args = {"viewport": {"width": 1024, "height": 640}}
    if device:
        context_args.update(pw.devices[device])
    if geolocation:
        context_args["geolocation"] = geolocation
        context_args["permissions"] = ["geolocation"]
    context = browser.new_context(**context_args)
    page = context.new_page()
    return browser, context, page


def test_assign_share_url(pytestconfig):
    """
    Tests assigning a share URL in Expensify while optionally capturing a Playwright trace.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        # Launch the browser / context / page
        browser, context, page = launch_app(p)

        # Install online guard immediately after context & page creation
        install_online_guard_sync(context, page)

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            user_info = get_test_user_info(first_name="abc", last_name="abc")
            login_user(page, user_info)

            page.wait_for_timeout(3000)
            page.locator("span:has-text('random email (you)')").click()
            page.wait_for_timeout(1000)

            screen_id = page.url.split("/")[-1]
            page.get_by_test_id(f"report-screen-{screen_id}").get_by_label("Create").click()
            page.get_by_label("Assign task").click()
            page.get_by_role("textbox", name="Title").fill("task99")
            page.get_by_role("button", name="Next").click()
            page.get_by_test_id("NewTaskPage").get_by_label("Back").click()
            page.wait_for_timeout(1000)
            page.get_by_label("Back").click()
            page.wait_for_timeout(1000)

            page.get_by_role("textbox").fill(
                "https://dev.new.expensify.com:8082/new/task/somewhere11"
            )
            page.get_by_test_id(f"report-screen-{screen_id}").get_by_label("Send").click()

            page.get_by_test_id(f"report-screen-{screen_id}").get_by_test_id(
                "report-actions-list"
            ).get_by_text("https://dev.new.expensify.com:8082/new/task/somewhere11").click()

            page.wait_for_timeout(3000)
            locator = page.locator('div:has-text("Hmm... it\'s not here")')
            assert locator.count() > 0, "The div with the text 'Hmm... it's not here' is not found."
            print("The message 'Hmm... it's not here' is visible on page")
            page.wait_for_timeout(4000)

        finally:
            # Stop tracing & clean the trace if it was enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Ensure browser resources are closed even on failure
            try:
                page.close()
            except Exception:
                pass
            try:
                context.close()
            except Exception:
                pass
            browser.close()
