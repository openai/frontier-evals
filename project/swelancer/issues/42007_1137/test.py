import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright._impl._page import Page
from playwright.sync_api import expect, sync_playwright
from utils.online_guard import install_online_guard_sync

EMAIL = "exxpp00112+42007@gmail.com"


def login_user(page: Page, first_name: str = "Milan", last_name: str = "T") -> tuple[Page, str]:
    user_email = EMAIL
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(1000)
    try:
        page.locator('button[tabindex="0"]').click()
    except Exception:
        # Second click is only present on some flows; ignore failures.
        pass
    expect(page.locator("text=What do you want to do today?")).to_be_visible()
    page.locator("text='Track and budget expenses'").click()
    page.get_by_role("button", name="Continue").click()
    page.locator('input[name="fname"]').fill(first_name)
    page.locator('input[name="lname"]').fill(last_name)
    page.get_by_role("button", name="Continue").last.click()
    return page, user_email


def verify_button_in_bottom_right_quarter(page: Page):
    page.wait_for_timeout(1000)
    got_it = page.locator("button[aria-label]").locator("visible=true").get_by_text("Got it")
    viewport = page.viewport_size
    button_bbox = got_it.bounding_box()
    assert is_button_in_bottom_right_quarter(button_bbox, viewport)


def is_button_in_bottom_right_quarter(button_bbox, viewport):
    quarter_x_start = viewport["width"] / 2
    quarter_y_start = viewport["height"] / 2
    return (
        button_bbox["x"] >= quarter_x_start
        and button_bbox["y"] >= quarter_y_start
        and (button_bbox["x"] + button_bbox["width"] <= viewport["width"])
        and (button_bbox["y"] + button_bbox["height"] <= viewport["height"])
    )


def test_submit_expense_button(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context()

        # Start tracing if requested
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)

            page, _ = login_user(page)

            create_chat_button = page.locator("button[aria-label='Start chat (Floating action)']")
            create_chat_button.click()
            page.wait_for_timeout(1000)

            submit_expense_option = page.locator(
                "div[aria-label='Submit expense'][role='menuitem']"
            )
            submit_expense_option.click()
            page.wait_for_timeout(1000)

            manual_option = page.locator("button[aria-label='Manual']")
            manual_option.click()
            page.wait_for_timeout(1000)

            page.keyboard.type("100")
            page.keyboard.press("Enter")
            page.wait_for_timeout(1000)

            page.keyboard.type("+923211234567")
            selection_list = page.locator("div[data-testid='selection-list']")
            page.wait_for_timeout(2000)
            first_button = selection_list.locator("button").all()
            first_button[0].click()
            page.wait_for_timeout(1000)

            page.keyboard.press("Enter")
            create = page.locator("button[aria-label='Create']").locator("visible=true")
            create.click()
            page.wait_for_timeout(1000)

            submit_expense = (
                page.locator("div[aria-label='Submit expense'][role='menuitem']")
                .locator("visible=true")
                .get_by_text("Submit expense")
            )
            submit_expense.click()
            page.wait_for_timeout(1000)

            page.keyboard.type("200")
            page.keyboard.press("Enter")
            page.keyboard.press("Enter")

            search = page.locator("button[aria-label='Search']").locator("visible=true")
            page.wait_for_timeout(1000)
            search.click()
            page.wait_for_timeout(1000)

            while True:
                view_button = (
                    page.locator("button[aria-label]")
                    .locator("visible=true")
                    .get_by_text("View")
                    .all()
                )
                if view_button:
                    break

            view_button = view_button[0]
            view_button.click()
            page.wait_for_timeout(1000)

            header = (
                page.locator("div[data-testid='DisplayNamesWithTooltip']")
                .locator("visible=true")
                .get_by_text("expense")
            )
            header.click(force=True)
            page.wait_for_timeout(1000)

            hold_button = page.locator("button[aria-label]").get_by_text("Hold")
            hold_button.click(force=True)
            page.wait_for_timeout(1000)

            page.keyboard.type("Hold reason")
            page.keyboard.press("Enter")

            # Verification – same as original behaviour
            temp_page = verify_button_in_bottom_right_quarter(page)

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
