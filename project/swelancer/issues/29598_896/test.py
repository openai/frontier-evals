import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from PIL import Image
from playwright.sync_api import Locator, Page, sync_playwright


def generate_random_email(username: str = "expensifyapp97"):
    return f"{username}-19876543112@gmail.com".lower()


def login_user(page: Page, first_name="Milan", last_name="T"):
    user_email = generate_random_email(f"{first_name}-{last_name}")
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator("button", has_text="Continue").click()
    page.locator("button", has_text="Join").click()
    is_name_set = True
    try:
        page.locator("text='Track and budget expenses'").click()
        page.get_by_role("button", name="Continue").click()
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("form").get_by_role("button", name="Continue").click()
        page.get_by_text("Welcome to Expensify").wait_for()
    except Exception:
        is_name_set = False
    return (user_email, is_name_set)


def check_if_an_element_is_entirely_clickable(page: Page, element: Locator):
    """
    Checks if an element is entirely clickable by sampling multiple points
    across the element’s area and verifying the cursor is ‘pointer’.
    """
    element.wait_for()
    element_box = element.bounding_box()
    rows, cols = (3, 3)
    for i in range(rows):
        for j in range(cols):
            x = element_box["x"] + (j + 0.5) * element_box["width"] / cols
            y = element_box["y"] + (i + 0.5) * element_box["height"] / rows
            page.mouse.move(x, y)
            is_clickable = page.evaluate(
                """
                ([x, y]) => {
                    const el = document.elementFromPoint(x, y);
                    return window.getComputedStyle(el).cursor === 'pointer';
                }
                """,
                [x, y],
            )
            if not is_clickable:
                return False
    return True


def check_if_close_button_is_entirely_clickable(page: Page, file_name: str):
    """
    Uploads an attachment and verifies the close button on the attachment
    modal is entirely clickable.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)

    page.get_by_text("Asaad I (you)").wait_for()
    page.get_by_text("Asaad I (you)").click()
    page.get_by_test_id("report-actions-list").get_by_text("Your space").wait_for()
    page.get_by_label("Create").last.click()

    with page.expect_file_chooser() as fc_info:
        page.get_by_label("Add attachment").click()
    file_chooser = fc_info.value
    file_chooser.set_files(file_path)

    close_button = page.get_by_label("Close")
    is_clickable = check_if_an_element_is_entirely_clickable(page, close_button)
    assert is_clickable, "The Close button is not entirely clickable"


def create_test_image(file_path: str):
    """
    Creates a simple PNG image and saves it at the specified path.
    """
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    img.save(file_path)


def test_close_button_is_entirely_clickable(pytestconfig):
    """
    Test to verify that the close button is entirely clickable.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        first_name = "Asaad"
        last_name = "I"
        proxy = {"server": "http://127.0.0.1:8080/"}
        args = [
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ]

        browser = p.chromium.launch(headless=True, args=args, proxy=proxy)
        context = browser.new_context()

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)

            # User onboarding / sign-in.
            user_email, is_name = login_user(page, first_name, last_name)
            if not is_name:
                # Complete profile setup if it wasn't done during sign-up.
                page.locator('button[aria-label="My settings"]').click()
                page.get_by_label(user_email).first.click()
                page.get_by_role("textbox", name="First name").fill(first_name)
                page.get_by_role("textbox", name="Last name").fill(last_name)
                page.get_by_role("button", name="Save").click()
                page.locator('button[aria-label="Inbox"]').click()

            # Prepare an image to upload.
            home_directory = os.path.expanduser("~")
            test_image_path = os.path.join(home_directory, "test_image_upload.png")
            create_test_image(test_image_path)

            # Perform the main assertion.
            check_if_close_button_is_entirely_clickable(page, test_image_path)

        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()
