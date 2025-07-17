import io
import os
import sys
import tempfile
import uuid

import pytest
from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

# Ensure repository root is on PYTHONPATH
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)


@pytest.fixture(scope="function")
def generate_password_protected_pdf() -> io.BytesIO:
    """
    Generates a password-protected PDF file in memory and returns it.
    """
    password = "12345"
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 750, "This is a password-protected PDF.")
    c.save()
    pdf_buffer.seek(0)

    reader = PdfReader(pdf_buffer)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    protected_pdf_buffer = io.BytesIO()
    writer.encrypt(user_password=password, use_128bit=True)
    writer.write(protected_pdf_buffer)
    protected_pdf_buffer.seek(0)
    return protected_pdf_buffer


def login_user(p: PlaywrightContextManager, first_name: str = "Milan", last_name: str = "T"):
    """
    Logs a user into the Expensify NewDot site and returns the browser, page,
    and context objects along with the email used.
    """
    browser = p.chromium.launch(
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = browser.new_context(ignore_https_errors=True)
    page = context.new_page()
    install_online_guard_sync(context, page)  # Install online guard immediately after page creation

    user_email = "test25570expensf@gmail.com"
    page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
    page.locator('input[type="email"]').fill(user_email)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)
    page.locator('button[tabindex="0"]').click()
    page.wait_for_timeout(2000)
    try:
        page.locator("text=What do you want to do today?").wait_for(state="visible", timeout=5000)
        element = page.locator("text='Track and budget expenses'")
        element.scroll_into_view_if_needed()
        element.click()
        page.get_by_role("button", name="Continue").click()
        page.locator('input[name="fname"]').fill(first_name)
        page.locator('input[name="lname"]').fill(last_name)
        page.get_by_role("button", name="Continue").last.click()
        page.wait_for_timeout(2000)
    except TimeoutError:
        # User has already completed onboarding – ignore.
        pass

    return browser, page, context, user_email


def test_PDFPasswordForm_textInput_focus(generate_password_protected_pdf: io.BytesIO, pytestconfig):
    """
    Uploads a password-protected PDF and asserts the password input retains focus
    after submitting an incorrect password.
    """
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")

    with sync_playwright() as p:
        browser, page, context, _ = login_user(p)

        # Start tracing if required
        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            # Write protected PDF to a temp file for upload
            protected_pdf_stream = generate_password_protected_pdf
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(protected_pdf_stream.read())
                temp_pdf_path = temp_pdf.name

            # Begin chat flow
            page.get_by_label("Start chat (Floating action)").click()
            page.get_by_text("Start chat").click()
            page.get_by_test_id("selection-list").get_by_label("Milan T (you)").click()
            page.get_by_label("Create").last.click()

            # Upload the PDF
            with page.expect_file_chooser() as fc_info:
                page.get_by_text("Add attachment").click()
            file_chooser = fc_info.value
            file_chooser.set_files(temp_pdf_path)

            # Interact with password prompt
            page.get_by_role("link", name="enter the password").click()
            page.get_by_role("textbox", name="Password").fill("558885558558")
            page.get_by_role("button", name="Confirm").click()

            # Assertion – the textbox should stay focused
            password_textbox = page.get_by_role("textbox", name="Password")
            expect(password_textbox).to_be_focused()

        finally:
            # Clean up temporary file
            try:
                os.remove(temp_pdf_path)
            except Exception:
                pass

            # Handle tracing teardown
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Close Playwright resources
            context.close()
            browser.close()
