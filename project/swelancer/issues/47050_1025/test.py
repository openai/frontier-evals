import os
import sys
import uuid

import pytest
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright, expect
from utils.browser import close_browser
from utils.login import complete_onboarding, sign_up
from utils.online_guard import install_online_guard_async


# -----------------------------  Browser Helpers  ----------------------------- #
async def start_browser(
    headless: bool = True,
    proxy: dict | None = None,
    launch_args: list[str] | None = None,
):
    """
    Starts a Playwright browser and returns (context, page, playwright) tuple.
    `playwright` is returned so the caller can close it later on.
    """
    launch_args = launch_args or ["--ignore-certificate-errors"]
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=headless, proxy=proxy, args=launch_args)
    context = await browser.new_context(ignore_https_errors=True)
    page = await context.new_page()
    return context, page, playwright


# --------------------------  Image Generation Helper  ------------------------ #
def generate_invoice_template(filename: str):
    """Generate a dummy invoice template image and save it to `filename`."""
    width, height = 750, 1000
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    draw.text((20, 20), "Invoice Template", fill="black", font=font)
    draw.text((20, 60), "Company Name: [Your Company]", fill="black", font=font)
    draw.text((20, 100), "Date: [Invoice Date]", fill="black", font=font)
    draw.text((20, 140), "Bill To: [Customer Name]", fill="black", font=font)
    draw.text((20, 180), "Address: [Customer Address]", fill="black", font=font)

    draw.text((20, 240), "Item Description", fill="black", font=font)
    draw.text((400, 240), "Quantity", fill="black", font=font)
    draw.text((520, 240), "Price", fill="black", font=font)
    draw.text((620, 240), "Total", fill="black", font=font)

    draw.line((20, 270, 730, 270), fill="black", width=2)
    y = 310
    for _ in range(5):
        draw.line((20, y, 730, y), fill="black", width=1)
        y += 40

    image.save(filename)


# ----------------------------------  Test  ----------------------------------- #
@pytest.mark.asyncio
async def test_image(pytestconfig):
    trace_enabled: bool = pytestconfig.getoption("--user-tool-trace")
    context = page = playwright = None  # ensure defined for finally scope

    try:
        # ------------------------------------------------------------------ #
        #                    Setup browser & (optional) trace                #
        # ------------------------------------------------------------------ #
        context, page, playwright = await start_browser(
            proxy={"server": "http://127.0.0.1:8080"},
            launch_args=[
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        await install_online_guard_async(context, page)

        if trace_enabled:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # ------------------------------------------------------------------ #
        #                           Test steps                               #
        # ------------------------------------------------------------------ #
        derived_email = "testuserexpensify+15443388@gmail.com"

        # Step 1: Sign up / log in
        await sign_up(page, derived_email)

        # Complete onboarding (ignore failures)
        try:
            await complete_onboarding(page)
        except Exception:
            pass

        # Verify Inbox visible
        await expect(page.get_by_label("Inbox")).to_be_visible()

        # Step 2: Start an expense submission
        await page.get_by_label("Start chat (Floating action)").click()
        await page.get_by_label("Submit expense").click()
        await page.get_by_label("Manual").click()
        await page.get_by_placeholder("0").fill("1")
        await page.locator("#numPadContainerView").get_by_role("button", name="Next").click()
        await page.get_by_test_id("selection-list-text-input").fill("a@e.com")
        await page.get_by_label("a@e.com").click()
        await page.get_by_label("More", exact=True).click()
        await page.get_by_label("Add receipt").click()

        # Step 3: Upload a receipt image
        file_path = os.path.expanduser("./invoice-template-us-neat-750px.png")
        generate_invoice_template(file_path)

        if os.path.exists(file_path):
            async with page.expect_file_chooser() as fc_info:
                await page.get_by_label("Choose file").click()
            file_chooser = await fc_info.value
            await file_chooser.set_files(file_path)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 4: Submit the expense
        await page.get_by_role("button", name=re.compile(r"^Submit")).click()

        # Step 5: Validate the thumbnail shows a blob URL (while uploading)
        parent_locator = page.get_by_label("View details")
        receipt_thumbnail = parent_locator.locator(
            "div.css-view-175oi2r.r-flexBasis-1mlwlqe img.css-accessibilityImage-9pa8cd"
        )
        await expect(receipt_thumbnail).to_have_attribute("src", re.compile(r"blob:"), timeout=500)

    finally:
        # Clean up generated file if present
        try:
            if "file_path" in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        # Handle tracing if enabled
        if trace_enabled and context is not None:
            trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            try:
                await context.tracing.stop(path=trace_path)
            finally:
                # Even if stopping trace fails, attempt to clean whatever was produced
                if os.path.exists(trace_path):
                    trace_cleaner(trace_path)

        # Close browser/resources
        if context and page and playwright:
            await close_browser(context, page, playwright)
