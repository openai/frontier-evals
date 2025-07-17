import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

from playwright.sync_api import expect, sync_playwright


def check_focus(page):
    page.get_by_test_id("username").fill("aquasaddam+28553@gmail.com")
    page.get_by_role("button", name="Continue").click()
    page.get_by_test_id("validateCode").press("ArrowRight")
    page.get_by_test_id("validateCode").press("ArrowRight")
    page.get_by_test_id("validateCode").press("ArrowRight")
    page.wait_for_timeout(5000)
    page.get_by_label("Didn't receive a magic code?").click()
    page.wait_for_timeout(2000)
    element = page.locator(
        "#root > div > div > div > div > div:nth-child(1) > div > div.css-view-175oi2r.r-flex-13awgt0 > div > div > div.css-view-175oi2r.r-flex-13awgt0 > div > div.css-view-175oi2r.r-bottom-1p0dtai.r-left-1d2f490.r-position-u8s1d.r-right-zchlnj.r-top-ipm5af.r-pointerEvents-12vffkv > div.css-view-175oi2r.r-flex-13awgt0.r-pointerEvents-12vffkv > div > div > div > div.css-view-175oi2r.r-flex-13awgt0 > div > div > div > div > div > div:nth-child(2) > div > div:nth-child(1) > div > div > div > div:nth-child(2) > form > div:nth-child(4) > div:nth-child(1) > div:nth-child(1) > div:nth-child(2) > div"
    )
    expect(element).to_have_css("border-color", "rgb(3, 212, 124)")
    page.wait_for_timeout(1000)


def test_run(pytestconfig):
    """
    Ported test with optional playwright tracing controlled by --user-tool-trace.
    """
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            proxy={"server": "http://127.0.0.1:8080"},
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
            page.goto("https://dev.new.expensify.com:8082/", timeout=60000)
            check_focus(page)
        finally:
            # Stop tracing (export + clean) if it was enabled
            if trace_enabled:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            # Clean-up playwright resources
            context.close()
            browser.close()
