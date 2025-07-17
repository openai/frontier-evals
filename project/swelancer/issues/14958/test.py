import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import re

from playwright.sync_api import expect, sync_playwright

NEWDOT_URL = "https://dev.new.expensify.com:8082/"
EMAIL = "pehlajrajput1+345@gmail.com"
PASSWORD = "vjycqezohxxrdhzr"


def new_dot_login(page, email):
    page.goto(NEWDOT_URL, timeout=60000)
    page.locator('input[type="email"]').fill(email)
    page.wait_for_timeout(2000)
    page.get_by_role("button", name="Continue").nth(0).click()
    page.wait_for_timeout(10000)
    page.locator('input[data-testid="validateCode"]').fill("123456")
    try:
        page.get_by_role(
            "button",
            name=re.compile(r"^(continue|sign.?in)$", re.I),
        ).click(timeout=20000)
    except Exception:
        pass


def zip_error(page, assertion_errors):
    try:
        page.get_by_label("My settings").click()
        page.wait_for_timeout(1000)
        page.get_by_label("England's Lane, London,").click()
        page.wait_for_timeout(1000)
        page.get_by_label("Address line 1").get_by_role("textbox").fill("Maliebaan 87")
        page.wait_for_timeout(1000)
        page.get_by_text("Maliebaan").first.click()
        page.wait_for_timeout(1000)
        page.get_by_role("textbox", name="Zip / Postcode").fill("1012 JS")
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Save").click()
        page.wait_for_timeout(1000)
        expect(page.get_by_text("Incorrect zip code format.")).not_to_be_visible()
    except AssertionError as e:
        assertion_errors.append(f"Error: Zip code validation failed for 'Maliebaan 87': {str(e)}")
    try:
        page.get_by_label("87 Maliebaan, Utrecht,").click()
        page.wait_for_timeout(1000)
        page.get_by_label("Address line 1").get_by_role("textbox").fill("tokyo")
        page.wait_for_timeout(1000)
        page.get_by_text("Tokyo Metro Marunouchi Line").click()
        page.wait_for_timeout(1000)
        page.get_by_role("textbox", name="Zip / Postcode").fill("100-0001")
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Save").click()
        page.wait_for_timeout(1000)
        expect(page.get_by_text("Incorrect zip code format.")).not_to_be_visible()
    except AssertionError as e:
        assertion_errors.append(
            f"Error: Zip code validation failed for 'Tokyo Metro Marunouchi Line': {str(e)}"
        )
    try:
        page.get_by_label("Tokyo Metro Marunouchi Line,").click()
        page.wait_for_timeout(1000)
        page.get_by_label("Address line 1").get_by_role("textbox").fill("engla")
        page.wait_for_timeout(1000)
        page.get_by_text("England's Lane").click()
        page.wait_for_timeout(1000)
        page.get_by_role("textbox", name="Zip / Postcode").fill("SW1A 1AA")
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Save").click()
        page.wait_for_timeout(1000)
        expect(page.get_by_text("Incorrect zip code format.")).not_to_be_visible()
    except AssertionError as e:
        assertion_errors.append(f"Error: Zip code validation failed for 'SW1A 1AA': {str(e)}")
    try:
        page.get_by_label("England's Lane, London,").click()
        page.get_by_role("textbox", name="Zip / Postcode").fill(",")
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Save").click()
        page.wait_for_timeout(1000)
        expect(page.get_by_text("Incorrect zip code format.")).to_be_visible()
    except AssertionError as e:
        assertion_errors.append(f"Error: Zip code validation failed for ',': {str(e)}")
    try:
        page.get_by_label("Back").click()
        page.get_by_label("England's Lane, London,").click()
        page.wait_for_timeout(1000)
        page.get_by_label("Address line 1").get_by_role("textbox").click()
        page.get_by_label("Address line 1").get_by_role("textbox").fill("2574 Fairfield Road")
        page.wait_for_timeout(1000)
        page.get_by_text("Fairfield Road").first.click()
        page.wait_for_timeout(1000)
        page.get_by_role("textbox", name="Zip / Postcode").fill("234GEs")
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Save").click()
        page.wait_for_timeout(1000)
        expect(page.get_by_text("Incorrect zip code format.")).to_be_visible()
    except AssertionError as e:
        assertion_errors.append(
            f"Error: Zip code validation failed for 'England's Lane, London': {str(e)}"
        )


def test_expensify_0000(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    assertion_errors = []

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
        context = browser.new_context(ignore_https_errors=True)

        if trace_enabled:
            context.tracing.start(screenshots=True, snapshots=True, sources=True)

        try:
            page = context.new_page()
            install_online_guard_sync(context, page)
            page.wait_for_timeout(2000)
            new_dot_login(page, EMAIL)
            zip_error(page, assertion_errors)
        finally:
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            context.close()
            browser.close()

    if assertion_errors:
        print("\nAssertion Errors:")
        for error in assertion_errors:
            print(f"- {error}")
        raise AssertionError("One or more assertions failed. Check the logs above.")


if __name__ == "__main__":
    try:

        class _DummyConfig:
            def getoption(self, name):
                return False

        test_expensify_0000(_DummyConfig())
    except AssertionError as e:
        print(str(e))
