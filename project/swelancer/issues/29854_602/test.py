import logging
import os
import re
import sys
import sys as _sys
import uuid
from io import BytesIO

import pytest
from PIL import Image
from playwright.async_api import async_playwright, expect
from utils.online_guard import install_online_guard_async
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

# CONSTS
NEWDOT_URL = "https://dev.new.expensify.com:8082/"
OLDDOT_URL = "http://localhost:9000/"
EXPENSIFY_EMAIL = "testingq186@gmail.com"
EXPENSIFY_PASSWORD = "kkfr hqiv yuxq rbso"

# Logger
logging.basicConfig(level=logging.INFO, stream=_sys.stdout, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def create_dummy_image():
    """Create an in-memory dummy PNG file."""
    img = Image.new("RGB", (100, 100), color="red")
    byte_io = BytesIO()
    img.save(byte_io, "PNG")
    byte_io.seek(0)
    return byte_io


# True when testing the flow file, False while recording
TESTING_FLOW = True


@pytest.mark.asyncio
async def test_the_issue(pytestconfig):
    """
    Demonstrate the Expensify app email login verification.
    """
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    # Initialise playwright & browser
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--ignore-certificate-errors",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=500,
    )
    context = await browser.new_context(ignore_https_errors=True)

    # Start tracing if required
    if trace_enabled:
        await context.tracing.start(screenshots=True, snapshots=True, sources=True)

    try:
        page = await context.new_page()
        await install_online_guard_async(context, page)
        await page.goto(NEWDOT_URL)

        await page.get_by_role("textbox", name="Phone or email").fill(EXPENSIFY_EMAIL)
        await page.get_by_role("button", name="Continue").click()

        # Await OTP
        otp = "123456"
        await page.locator('input[name="validateCode"]').first.fill(otp)
        try:
            await page.get_by_role(
                "button",
                name=re.compile(r"^(continue|sign.?in)$", re.I),
            ).click(timeout=20000)
        except Exception:
            pass

        # Wait sign in to complete
        await page.get_by_text("Please enter the magic code").wait_for(state="hidden")
        logging.info("Sign in complete.")

        await page.locator('div[aria-label="Start chat (Floating action)"]').last.click()
        await page.locator('text="Request money"').click()
        await page.locator('text="Manual"').last.click()
        await page.locator('input[placeholder="0"]').fill("200")
        await page.keyboard.press("Enter")
        await page.locator('input[aria-label="Name, email, or phone number"]').fill(
            "test1@gmail.com"
        )
        await page.locator('div[aria-label="test1@gmail.com"]').last.click()
        await page.locator('div[aria-label="More"]').last.click()
        await page.locator('text="Add receipt"').last.click()

        # Upload in-memory image
        dummy_image = create_dummy_image()
        async with page.expect_file_chooser() as fc_info:
            await page.locator('text="Choose file"').last.click()
            file_chooser = await fc_info.value
            await file_chooser.set_files(
                [
                    {
                        "name": "dummy_image.png",
                        "mimeType": "image/png",
                        "buffer": dummy_image.getvalue(),
                    }
                ]
            )

        two_100_dollar = page.get_by_role("button", name="Request").filter(has_text="200.0")
        await expect(two_100_dollar).to_be_visible()

    finally:
        # Stop tracing and clean trace if enabled
        if trace_enabled:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            await context.tracing.stop(path=trace_path)
            trace_cleaner(trace_path)

        # Close context, browser & playwright
        await context.close()
        await browser.close()
        await pw.stop()
