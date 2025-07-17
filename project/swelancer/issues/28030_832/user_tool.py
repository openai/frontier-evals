import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import io

import pytest
from utils.browser import close_browser, start_browser
from utils.login import NEW_DOT_URL, check_if_logged_in
from utils.trace_cleaner import trace_cleaner

EMAIL = "testexpen@gmail.com"


def generate_in_memory_text_file(content):
    # Create an in-memory text file
    text_file = io.BytesIO()
    text_file.write(content.encode("utf-8"))
    text_file.seek(0)  # Reset file pointer to the beginning
    return text_file, "example_text_file.txt"


@pytest.mark.asyncio
async def test_issue_28030():
    # Start browser
    context, page, playwright = await start_browser(
        persistent=True,
        launch_args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins, site-per-process",
        ],
    )

    await context.tracing.start(title="test_issue_28030", screenshots=True, snapshots=True)

    try:
        if not await check_if_logged_in(page=page, url=NEW_DOT_URL):
            await page.get_by_test_id("username").fill(EMAIL)
            await page.get_by_role("button", name="Continue").click()
            await page.get_by_test_id("validateCode").fill("123456")

        await page.get_by_label("My settings").click()

        # Generate an in-memory text file
        content = "This is a sample text file generated for demonstration purposes."
        text_file, file_name = generate_in_memory_text_file(content)

        # Expect the file chooser to trigger
        async with page.expect_file_chooser() as fc_info:
            # Click the "Edit Photo" button that triggers the file chooser dialog
            await page.get_by_label("Edit photo").click()

            # Await the file chooser object from the file chooser event
            file_chooser = await fc_info.value

            # Use the in-memory text file as the input file (invalid image)
            await file_chooser.set_files(
                {"name": file_name, "mimeType": "text/plain", "buffer": text_file.read()}
            )

        await page.get_by_text("Profile picture must be one").hover()

        # Check if the text is selectable by selecting and copying it
        await page.mouse.move(455, 340)
        await page.mouse.down()  # Simulates starting a text selection
        await page.mouse.move(900, 340)
        await page.mouse.up()  # Simulate end of a text selection

        selected_text = await page.evaluate("() => window.getSelection().toString()")
        # Assert text has been selected
        assert (
            selected_text
            == "Profile picture must be one of the following types: jpg, jpeg, png, gif, bmp, svg."
        ), f"Text not selectable, got '{selected_text}'"
    except Exception:
        # Re-raise to mark the test as failed
        raise
    finally:
        # Stop tracing and save it
        await context.tracing.stop(path="/app/expensify/user_tool/output_browser1.zip")
        # Clean the trace
        trace_cleaner("/app/expensify/user_tool/output_browser1.zip")
        # Close the browser
        await close_browser(page, context, playwright)
