import os
import sys
import uuid

from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re
from io import BytesIO

from playwright.sync_api import expect, sync_playwright
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from utils.online_guard import install_online_guard_sync

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EMAIL = "pehlajrajput1+89@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    page.locator('input[data-testid="validateCode"]').fill("123456")
    try:
        page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
            timeout=20000
        )
    except Exception:
        pass


def create_sample_pdf():
    pdf_byte_arr = BytesIO()
    c = canvas.Canvas(pdf_byte_arr, pagesize=letter)
    c.drawString(100, 750, "Sample PDF")
    c.drawString(100, 730, "This is a sample PDF file generated in Python.")
    c.showPage()
    c.save()
    pdf_byte_arr.seek(0)
    return pdf_byte_arr.getvalue()


def attachemet_issue(page):
    file_format = "pdf"
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").fill("tosomenewuser@gmail.com")
    page.wait_for_timeout(1000)
    page.get_by_label("tosomenewuser@gmail.com").click()
    page.wait_for_timeout(2000)
    current_url = page.url
    match = re.search("/r/(\\d+)", current_url)
    if match:
        last_id = match.group(1)
        print(f"Extracted last ID: {last_id}")
        test_id = f"report-screen-{last_id}"
        page.wait_for_timeout(1000)
    else:
        print("No matching ID found in the URL.")
    page.wait_for_timeout(4000)
    with page.expect_file_chooser() as fc_info:
        page.wait_for_timeout(2000)
        page.get_by_test_id(test_id).get_by_label("Create").click()
        page.wait_for_timeout(1000)
        page.get_by_text("Add attachment").click()
    file_chooser = fc_info.value
    page.wait_for_timeout(1000)
    file_chooser.set_files(
        {
            "name": f"test_pdf.{file_format.lower()}",
            "mimeType": f"pdf/{file_format.lower()}",
            "buffer": create_sample_pdf(),
        }
    )
    page.wait_for_timeout(1000)
    page.get_by_role("dialog").get_by_role("button", name="Send").click()
    assertion_1 = expect(page.get_by_text("Uh-oh, something went wrong!")).not_to_be_visible()
    if assertion_1 is False:
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Sign out").click()
    else:
        page.get_by_label("My settings").click()
        page.get_by_label("Sign out").click()
        page.wait_for_timeout(2000)
    page.wait_for_timeout(2000)
    new_dot_login(page, "pehlajrajput1+89@gmail.com")
    page.get_by_label("Start chat (Floating action)").click()
    page.wait_for_timeout(1000)
    page.get_by_label("Start chat", exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("selection-list-text-input").fill("tosomenewuser@gmail.com")
    page.wait_for_timeout(1000)
    page.get_by_label("tosomenewuser@gmail.com").click()
    page.wait_for_timeout(1000)
    page.get_by_test_id("report-screen-2023706297795678").get_by_label(
        "View attachment"
    ).last.click()
    page.wait_for_timeout(1000)
    expect(page.get_by_text("Uh-oh, something went wrong!")).not_to_be_visible()


def test_expensify_0000(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            proxy={"server": "http://localhost:8080"},
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--ignore-certificate-errors",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
            slow_mo=None,
            channel="chromium",
        )
        first_user_context = browser.new_context(ignore_https_errors=True)
        if trace_enabled:
            first_user_context.tracing.start(screenshots=True, snapshots=True, sources=True)
        try:
            page = first_user_context.new_page()
            install_online_guard_sync(first_user_context, page)
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL)
            page.wait_for_timeout(2000)
            attachemet_issue(page)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                first_user_context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)
            first_user_context.close()
            browser.close()
