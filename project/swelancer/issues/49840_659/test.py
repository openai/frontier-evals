import os
import sys
import uuid

from utils.online_guard import install_online_guard_sync
from utils.trace_cleaner import trace_cleaner

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

import email
import imaplib
import re
import time
from email.header import decode_header
from enum import Enum

from playwright.sync_api import expect, sync_playwright
from playwright.sync_api._context_manager import PlaywrightContextManager

TEST_URL = "https://dev.new.expensify.com:8082"
USER_NAME = "caraxxes555"
USER_PASSWORD = "wawm wmyw omkr dpmt"


class TodayOptions(Enum):
    TRACK_AND_BUDGET_EXPENSES = 1
    SOMETHING_ELSE = 4


def get_test_user_info(seed=None):
    if seed is None:
        return {
            "email": f"{USER_NAME}@gmail.com",
            "password": USER_PASSWORD,
            "first_name": f"{USER_NAME}",
            "last_name": "Test",
        }

    if isinstance(seed, bool):
        seed = int(time.time())

    return {
        "email": f"{USER_NAME}+{seed}@gmail.com",
        "password": USER_PASSWORD,
        "first_name": f"{USER_NAME}+{seed}",
        "last_name": "Test",
    }


def wait(page, for_seconds=1):
    page.wait_for_timeout(for_seconds * 1000)


def get_magic_code(user_email, password, page, retries=5, delay=10):
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(user_email, password)
    for _ in range(retries):
        imap.select("inbox")
        status, messages = imap.search(None, '(UNSEEN SUBJECT "Expensify magic sign-in code:")')
        if status == "OK":
            email_ids = messages[0].split()
            if email_ids:
                latest_email_id = email_ids[-1]
                status, msg_data = imap.fetch(latest_email_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding or "utf-8")
                        match = re.search(r"Expensify magic sign-in code: (\d+)", subject)
                        if match:
                            code = match.group(1)
                            imap.logout()
                            return code
        wait(page)
    imap.logout()
    return None


def choose_what_to_do_today_if_any(page, option: TodayOptions, retries=5, **kwargs):
    try:
        wait(page)
        for _ in range(retries):
            wdyw = page.locator("text=What do you want to do today?")
            if wdyw.count() == 0:
                wait(page)
            else:
                break
        if wdyw.count() == 0:
            set_full_name(
                page=page,
                first_name=kwargs["first_name"],
                last_name=kwargs["last_name"],
            )
            return
        expect(wdyw).to_be_visible()
        text = (
            "Something else"
            if option == TodayOptions.SOMETHING_ELSE
            else "Track and budget expenses"
        )
        page.locator(f"text='{text}'").click()
        page.get_by_role("button", name="Continue").click()
        wait(page)
        page.locator('input[name="fname"]').fill(kwargs["first_name"])
        page.locator('input[name="lname"]').fill(kwargs["last_name"])
        wait(page)
        page.get_by_role("button", name="Continue").last.click()
        wait(page)
        if page.get_by_label("Close").count() > 0:
            page.get_by_label("Close").click()
    except Exception:
        pass


def choose_link_if_any(page, link_text, retries=5):
    try:
        wait(page)
        for _ in range(retries):
            link = page.locator(f"text={link_text}")
            if link.count() == 0:
                wait(page)
            else:
                break
        if link.count() == 0:
            return
        expect(link).to_be_visible()
        link.click()
    except Exception:
        return


def login(p: PlaywrightContextManager, user_info, if_phone=False):
    permissions = ["clipboard-read", "clipboard-write"]
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
        proxy={"server": "http://localhost:8080"},
        slow_mo=500,
    )
    if if_phone:
        phone = p.devices["iPhone 12 Pro"]
        context = browser.new_context(
            **phone, permissions=permissions, reduced_motion="no-preference"
        )
    else:
        context = browser.new_context()

    page = context.new_page()
    install_online_guard_sync(context, page)  # Install the online guard immediately after creation

    page.goto(TEST_URL, timeout=120000)
    phone_or_email_input = page.locator('input[type="email"]')
    expect(phone_or_email_input).to_be_visible()
    phone_or_email_input.fill(user_info["email"])
    continue_button = page.locator('button[tabindex="0"]')
    expect(continue_button).to_be_visible()
    continue_button.click()
    wait(page)
    join_button = page.locator('button:has-text("Join")')
    if join_button.count() > 0:
        join_button.click()
    else:
        magic_code = (
            get_magic_code(user_info["email"], user_info["password"], page, retries=3, delay=10)
            or "123456"
        )
        validate_code_input = page.locator('input[data-testid="validateCode"]')
        expect(validate_code_input).to_be_visible()
        validate_code_input.fill(magic_code)
        try:
            page.get_by_role("button", name=re.compile(r"^(continue|sign.?in)$", re.I)).click(
                timeout=20000
            )
        except Exception:
            pass
    return browser, context, page


def set_full_name(page, first_name, last_name):
    if page.get_by_label("Close").count() > 0:
        page.get_by_label("Close").click()
    page.get_by_label("My settings").click()
    page.get_by_role("menuitem", name="Profile").click()
    page.get_by_text("Display name").click()
    page.get_by_role("textbox", name="First name").fill(first_name)
    page.get_by_role("textbox", name="Last name").fill(last_name)
    page.get_by_role("button", name="Save").click()
    wait(page)
    if page.get_by_label("Back").count() > 0:
        page.get_by_label("Back").last.click()
    page.get_by_label("Inbox").click()


def create_workspace_change_language(page):
    wait(page)
    page.get_by_label("My settings").click()
    page.get_by_test_id("InitialSettingsPage").get_by_label("Workspaces").get_by_text(
        "Workspaces"
    ).click()
    page.get_by_label("New workspace").first.click()
    wait(page)
    page.get_by_label("Back").click()
    wait(page)
    page.get_by_label("Preferences").click()
    page.get_by_label("English").click()
    page.get_by_label("Spanish").click()
    wait(page)
    page.get_by_label("Recibidos").click()


def submit_expense_in_workspace_chat(page):
    wait(page)
    page.locator('button[aria-label="Navega a un chat"]', has_text="'s Workspace").last.click()
    wait(page)
    page.get_by_label("Crear").last.click()
    page.get_by_label("Presentar gasto").click()
    page.get_by_label("Manual").click()
    page.get_by_label("Selecciona una moneda").click()
    page.get_by_test_id("selection-list-text-input").fill("usd")
    page.get_by_label("USD - $").click()
    page.get_by_placeholder("0").fill("1000")
    wait(page)
    page.locator("#numPadContainerView").get_by_role("button", name="Siguiente").click()
    page.get_by_role("button", name=re.compile(r"Solicitar")).click()
    page.get_by_role("menuitem", name="Comerciante Este campo es").click()
    page.get_by_role("textbox", name="Comerciante").fill("test")
    page.get_by_role("button", name="Guardar").click()
    page.get_by_role("button", name=re.compile(r"Solicitar")).click()


def test_49840(pytestconfig):
    trace_enabled = pytestconfig.getoption("--user-tool-trace")
    with sync_playwright() as p:
        browser = None
        context = None
        try:
            user_info = get_test_user_info(seed=498403)

            browser, context, page = login(p, user_info, False)

            if trace_enabled:
                context.tracing.start(screenshots=True, snapshots=True, sources=True)

            choose_what_to_do_today_if_any(page, TodayOptions.SOMETHING_ELSE, **user_info)
            choose_link_if_any(page, "Get started")
            choose_link_if_any(page, "Go back to home page")
            create_workspace_change_language(page)
            submit_expense_in_workspace_chat(page)

            page.get_by_label("Ver detalles").get_by_role("button").nth(1).click()
            page.get_by_label(re.compile(r"Pagar .* de otra")).click()
            page.get_by_role("button", name=re.compile(r"agar .* de otra")).click()

            assert page.get_by_text("Pago completo").is_visible()
            wait(page)
        finally:
            if trace_enabled and context is not None:
                trace_path = f"/app/expensify/user_tool/output_{uuid.uuid4().hex}.zip"
                os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                context.tracing.stop(path=trace_path)
                trace_cleaner(trace_path)

            if context is not None:
                context.close()

            if browser is not None:
                browser.close()
