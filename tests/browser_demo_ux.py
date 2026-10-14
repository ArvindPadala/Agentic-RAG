"""Optional real-Chrome visitor smoke check with synthetic sources/providers.

Run: python -m tests.browser_demo_ux --artifacts /path/to/temporary-directory
Requires Selenium and a local Chrome browser, not a live Gemini/Supabase account.
Screenshots and browser profile/cache are kept in the supplied temporary directory.
"""

import argparse
import os
import re
import time
from types import SimpleNamespace
from pathlib import Path
import socket
import tempfile
from contextlib import ExitStack
from unittest.mock import patch

from tests import conftest  # noqa: F401 - disable dotenv, tracing and live credentials
import fitz
import gradio as gr
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from app import build_ui, DEMO_CSS
from access_control import RuntimePolicy
from google.genai import types
from cryptography.fernet import Fernet
from private_sessions import PrivateSessions, COOKIE_NAME
from tests.visitor_fixtures import VisitorMemoryBackend, extract_visitor_memory
from showcase_corpus import ShowcaseCorpus, curated_sources
from tests.test_runtime_privacy import ready_collection


def check_layout(browser, mobile):
    assert browser.execute_script("return document.documentElement.scrollWidth <= window.innerWidth + 2")
    chat = browser.find_element(By.ID, "conversation-col").rect
    evidence = browser.find_element(By.ID, "evidence-col").rect
    if mobile:
        assert evidence["y"] >= chat["y"] + chat["height"] - 2
    else:
        assert evidence["x"] > chat["x"]
        assert chat["width"] > evidence["width"]

    styles = browser.execute_script("""
        const field = document.querySelector('#research-question textarea');
        const placeholder = getComputedStyle(field, '::placeholder');
        let node = field, background;
        while (node) {
            background = getComputedStyle(node).backgroundColor;
            if (background !== 'rgba(0, 0, 0, 0)' && background !== 'transparent') break;
            node = node.parentElement;
        }
        return {color: placeholder.color, opacity: placeholder.opacity, background};
    """)
    foreground = [float(v) for v in re.findall(r"[\d.]+", styles["color"])][:3]
    background = [float(v) for v in re.findall(r"[\d.]+", styles["background"])][:3]
    alpha = float(styles["opacity"])
    foreground = [fg * alpha + bg * (1 - alpha) for fg, bg in zip(foreground, background)]

    def luminance(rgb):
        values = [v / 255 for v in rgb]
        linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
        return sum(v * weight for v, weight in zip(linear, (0.2126, 0.7152, 0.0722)))

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    assert (light + 0.05) / (dark + 0.05) >= 4.5, "Question placeholder contrast is insufficient"


def activate(browser, element, mobile):
    if not mobile:
        element.click()
        return
    # Exercise keyboard activation at the true mobile viewport; desktop click
    # coordinates in headless mobile emulation do not reliably follow scrolling.
    element.send_keys(Keys.ENTER)


def idle(browser, wait):
    wait.until(lambda driver: not any(e.is_displayed() for e in
        driver.find_elements(By.CSS_SELECTOR, "[data-testid='status-tracker']")))


def tab(browser, title, mobile):
    activate(browser, browser.find_element(By.XPATH,
        f"//button[@role='tab' and contains(., '{title}')]"), mobile)


def open_memory(browser):
    buttons = browser.find_elements(By.ID, "start-private-session")
    if not buttons or not buttons[0].is_displayed():
        browser.find_element(By.CSS_SELECTOR, "#research-memory button").send_keys(Keys.ENTER)


def ask(browser, wait, question, mobile):
    field = browser.find_element(By.CSS_SELECTOR, "#research-question textarea")
    field.clear()
    field.send_keys(question)
    activate(browser, browser.find_element(By.ID, "send-btn"), mobile)
    idle(browser, wait)


def visitor_recovery_checks(browser, wait, scenario, turns, mobile):
    generation_calls = len(turns)
    for state, expected in (("empty", "No public reports are ready"),
                            ("unavailable", "temporarily unavailable"), ("ready", "Reports ready")):
        scenario["availability"] = state
        activate(browser, browser.find_element(By.ID, "refresh-reports"), mobile)
        wait.until(lambda driver: expected in driver.find_element(By.ID, "corpus-status").text)
        tab(browser, "Research", mobile)
        assert browser.find_element(By.ID, "preset-fact").is_enabled() is (state == "ready")
        assert browser.find_element(By.ID, "send-btn").is_enabled() is (state == "ready")
        tab(browser, "Explore reports", mobile)
    assert len(turns) == generation_calls, "Availability checks must not request generation"
    tab(browser, "Research", mobile)
    scenario["missing_evidence"] = True
    activate(browser, browser.find_element(By.ID, "preset-fact"), mobile)
    wait.until(lambda driver: "do not establish this answer" in driver.find_element(By.ID, "research-chat").text)
    assert "without captured document evidence" in browser.find_element(By.ID, "request-status").text
    assert not browser.find_elements(By.CSS_SELECTOR, '#research-chat a[href^="#citation-"]')
    scenario["missing_evidence"] = False
    browser.find_element(By.ID, "clear-chat").send_keys(Keys.ENTER)
    wait.until(EC.visibility_of_element_located((By.ID, "research-welcome")))
    scenario["provider_error"] = True
    activate(browser, browser.find_element(By.ID, "preset-fact"), mobile)
    wait.until(lambda driver: "overloaded" in driver.find_element(By.ID, "request-status").text)
    assert "synthetic provider" not in browser.find_element(By.ID, "research-chat").text
    scenario["provider_error"] = False
    ask(browser, wait, "Retry the supported research question.", mobile)
    wait.until(lambda driver: "Answer ready" in driver.find_element(By.ID, "request-status").text)
    browser.find_element(By.ID, "clear-chat").send_keys(Keys.ENTER)
    wait.until(EC.visibility_of_element_located((By.ID, "research-welcome")))
    print("PASS recovery: empty/outage/recheck, missing evidence, provider error and successful retry")


def visitor_memory_checks(browser, wait, base_url, backend, mobile, marker, forget=True):
    # Only CAPTCHA and remote storage are synthetic. The real frontend start/
    # save/forget/end callbacks, encrypted cookies and identity guards execute.
    open_memory(browser)
    browser.execute_script("""window.turnstile = {
        render: (target, options) => { setTimeout(() => options.callback('synthetic-browser-captcha'), 0); return 1; },
        remove: () => {}
    };""")
    signups = backend.signups
    activate(browser, browser.find_element(By.ID, "start-private-session"), mobile)
    wait.until(lambda driver: "Private demo session active" in driver.find_element(By.ID, "private-session-status").text)
    assert backend.signups == signups + 1
    cookie = next(cookie for cookie in browser.get_cookies() if cookie["name"] == COOKIE_NAME)
    assert cookie["httpOnly"] and cookie["sameSite"] == "Lax"
    assert "synthetic-token" not in cookie["value"]
    assert COOKIE_NAME not in browser.execute_script("return document.cookie")
    assert not browser.find_element(By.ID, "save-research-memory").is_enabled()
    ask(browser, wait, marker, mobile)
    wait.until(lambda driver: "Answer ready" in driver.find_element(By.ID, "request-status").text)
    open_memory(browser)
    wait.until(EC.element_to_be_clickable((By.ID, "save-research-memory")))
    activate(browser, browser.find_element(By.ID, "save-research-memory"), mobile)
    wait.until(lambda driver: "Private demo memory saved" in driver.find_element(
        By.CSS_SELECTOR, "#research-memory textarea").get_property("value"))
    assert marker in browser.find_element(By.ID, "research-memory").text
    browser.get(base_url)
    wait.until(EC.element_to_be_clickable((By.ID, "preset-fact")))
    open_memory(browser)
    wait.until(lambda driver: "Private demo session active" in driver.find_element(By.ID, "private-session-status").text)
    wait.until(lambda driver: marker in driver.find_element(By.ID, "research-memory").text)
    assert not browser.find_element(By.ID, "save-research-memory").is_enabled()
    if forget:
        forget_memory(browser, wait, mobile, marker)
    print("PASS memory: explicit opt-in, encrypted HttpOnly cookie, save/panel and reload")


def forget_memory(browser, wait, mobile, marker):
    activate(browser, browser.find_element(By.ID, "forget-private-session"), mobile)
    wait.until(lambda driver: "Public guest mode" in driver.find_element(By.ID, "private-session-status").text)
    assert marker not in browser.find_element(By.ID, "research-memory").text
    assert COOKIE_NAME not in {cookie["name"] for cookie in browser.get_cookies()}
    assert marker not in browser.find_element(By.ID, "evidence-cards").text
    browser.find_element(By.CSS_SELECTOR, "#execution-details button").send_keys(Keys.ENTER)
    assert "Run a research question" in browser.find_element(By.ID, "execution-summary").text
    browser.find_element(By.CSS_SELECTOR, "#execution-details button").send_keys(Keys.ENTER)
    print("PASS forget: public mode, cleared cookie/evidence and execution details")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    parser.add_argument("--full-journey", action="store_true",
                        help="Also validate memory/isolation and empty/evidence/provider recovery in a browser")
    parser.add_argument("--memory-only", action="store_true",
                        help="Run the private-memory portion without repeating public research checks")
    args = parser.parse_args()
    if args.memory_only:
        args.full_journey = True
    args.artifacts.mkdir(parents=True, exist_ok=True)
    os.environ["SE_CACHE_PATH"] = str(args.artifacts / "selenium-cache")
    browser = None
    other_browser = None
    demo = None
    with tempfile.TemporaryDirectory(dir=args.artifacts) as directory:
        root = Path(directory)
        pdf_path = root / "fixture.pdf"
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Synthetic climate evidence for browser verification only.")
            pdf.save(pdf_path)
        sources = curated_sources()
        corpus = ShowcaseCorpus(root, [(source, pdf_path, "fixture") for source in sources])
        source_key = f"showcase/fixture/{sources[0]['id']}.pdf"
        collection = ready_collection()
        collection.get.return_value = {"ids": ["public-fixture"],
                                       "metadatas": [{"source_pdf_key": source_key, "page": 0}]}
        retrieved = {"chunk_id": "public-fixture", "source_document": sources[0]["title"],
                     "source_pdf_key": source_key, "source_url": sources[0]["url"],
                     "page": 0, "bbox": [0.1, 0.07, 0.9, 0.1],
                     "text": "Synthetic climate evidence for browser verification only.",
                     "score": 0.9, "chunk_type": "text"}
        turns = []
        scenario = {"availability": "ready", "missing_evidence": False, "provider_error": False}

        def get_records(**kwargs):
            if scenario["availability"] == "unavailable":
                raise RuntimeError("synthetic availability failure")
            ids = ["public-fixture"] if scenario["availability"] == "ready" else []
            return {"ids": ids, "metadatas": [{"source_pdf_key": source_key, "page": 0}] if ids else []}

        collection.get.side_effect = get_records

        def retrieve(**kwargs):
            return [] if scenario["missing_evidence"] else [retrieved]

        def response(text):
            return types.GenerateContentResponse(candidates=[types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]))])

        planning = {"used": False}

        def generate_content(model, contents, config):
            # The real graph, optimiser and checker run. Only provider responses
            # and retrieval data are synthetic; latency makes real progress visible.
            time.sleep(0.8)
            if scenario["provider_error"]:
                raise RuntimeError("503 synthetic provider unavailable")
            instruction = config.system_instruction or ""
            if "expert search query optimizer" in instruction:
                planning["used"] = True
                return response('["Compare warming risks", "Compare emissions pathways"]')
            if "strict faithfulness evaluator" in instruction:
                turns[-1] = (turns[-1][0], True)
                return response("malformed grounding verdict")
            last = contents[-1]
            tool_results = [p.function_response for p in last.parts if p.function_response]
            if tool_results:
                match = re.search(r"Evidence \[(\d+)\]", tool_results[0].response["result"])
                if not match:
                    return response("The available reports do not establish this answer. Try a different research question.")
                number = match.group(1)
                return response(f"Synthetic answer for browser verification [{number}]. Invalid reference [999].")
            turns.append((planning["used"], False))
            planning["used"] = False
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
                role="model", parts=[types.Part(function_call=types.FunctionCall(
                    name="search_knowledge_base", args={"query": "synthetic climate query"}))]))])

        provider = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        memory_backend = VisitorMemoryBackend()
        private_sessions = (PrivateSessions(memory_backend, Fernet.generate_key(), base_url, "synthetic-site-key")
                            if args.full_journey else None)
        try:
            with ExitStack() as patches:
                patches.enter_context(patch("agent.search_chroma", side_effect=retrieve))
                patches.enter_context(patch("hybrid_search.search_chroma_hybrid", side_effect=retrieve))
                if args.full_journey:
                    patches.enter_context(patch("app.update_memory_from_conversation", side_effect=extract_visitor_memory))
                demo = build_ui(provider, {}, "unused.json", None, collection, None, showcase=corpus,
                                private_sessions=private_sessions)
                demo.queue().launch(server_name="127.0.0.1", server_port=port,
                                    prevent_thread_lock=True, blocked_paths=[str(root)],
                                    app_kwargs={"routes": corpus.routes(collection, RuntimePolicy()) +
                                                (private_sessions.routes() if private_sessions else [])},
                                    max_file_size=0, css=DEMO_CSS,
                                    theme=gr.themes.Monochrome(primary_hue="slate", neutral_hue="slate", font=["system-ui", "sans-serif"]))
                options = webdriver.ChromeOptions()
                options.binary_location = args.chrome
                for argument in ("--headless=new", "--disable-dev-shm-usage", "--no-first-run",
                                 f"--user-data-dir={root / 'chrome-profile'}"):
                    options.add_argument(argument)
                browser = webdriver.Chrome(options=options)
                wait = WebDriverWait(browser, 30)
                for width, height, label in ((1360, 1000, "desktop"), (390, 844, "mobile")):
                    if label == "mobile":
                        # Fresh renderer/profile avoids carrying frontend state
                        # across a desktop-to-mobile emulation transition.
                        browser.quit()
                        mobile_options = webdriver.ChromeOptions()
                        mobile_options.binary_location = args.chrome
                        mobile_options.add_argument("--headless=new")
                        mobile_options.add_argument(f"--user-data-dir={root / 'mobile-browser-profile'}")
                        browser = webdriver.Chrome(options=mobile_options)
                        wait = WebDriverWait(browser, 30)
                    browser.set_window_size(width, height)
                    if label == "mobile":
                        browser.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                            "width": width, "height": height, "deviceScaleFactor": 1, "mobile": True})
                    browser.get(base_url)
                    wait.until(EC.element_to_be_clickable((By.ID, "preset-fact")))
                    wait.until(lambda driver: not any(element.is_displayed() for element in
                        driver.find_elements(By.CSS_SELECTOR, "[data-testid='status-tracker']")))
                    check_layout(browser, mobile=label == "mobile")
                    if args.memory_only:
                        visitor_memory_checks(browser, wait, base_url, memory_backend, label == "mobile",
                                              marker="synthetic-memory-" + label, forget=label != "mobile")
                        continue
                    browser.save_screenshot(str(args.artifacts / (label + "-welcome.png")))
                    activate(browser, browser.find_element(By.ID, "preset-fact"), mobile=label == "mobile")
                    wait.until(lambda driver: "Generating a response or next search" in driver.find_element(By.TAG_NAME, "body").text)
                    wait.until(lambda driver: "Synthetic answer" in driver.find_element(By.ID, "research-chat").text)
                    assert turns[-1] == (False, False)
                    wait.until(EC.invisibility_of_element_located((By.ID, "research-welcome")))
                    assert "Cited in this answer" in browser.find_element(By.ID, "evidence-cards").text
                    assert "#citation-999" not in browser.page_source
                    assert str(root) not in browser.find_element(By.ID, "research-chat").text
                    citation = browser.find_element(By.CSS_SELECTOR, '#research-chat a[href="#citation-1"]')
                    activate(browser, citation, mobile=label == "mobile")
                    wait.until(lambda driver: driver.find_element(By.CSS_SELECTOR, "#visual-grounding-html img").get_property("naturalWidth") > 0)
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#visual-grounding-html img")))
                    wait.until(lambda driver: not any(e.is_displayed() for e in
                        driver.find_elements(By.CSS_SELECTOR, "[data-testid='status-tracker']")))
                    assert "/showcase/citations/" in browser.find_element(By.CSS_SELECTOR, "#visual-grounding-html img").get_attribute("src")
                    browser.execute_script("arguments[0].scrollIntoView({block:'center', behavior:'instant'})", browser.find_element(By.ID, "visual-grounding-html"))
                    browser.save_screenshot(str(args.artifacts / (label + "-evidence.png")))
                    activate(browser, browser.find_element(By.ID, "followup-explain"), mobile=label == "mobile")
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#research-chat a[href="#citation-2"]')))
                    wait.until(lambda driver: not any(e.is_displayed() for e in
                        driver.find_elements(By.CSS_SELECTOR, "[data-testid='status-tracker']")))
                    assert turns[-1] == (False, False)
                    calls_before_inspection = len(turns)
                    activate(browser, browser.find_element(By.CSS_SELECTOR, '#research-chat a[href="#citation-1"]'), mobile=label == "mobile")
                    wait.until(lambda driver: "[1]" in driver.find_element(By.ID, "visual-grounding-html").text)
                    wait.until(lambda driver: not any(e.is_displayed() for e in
                        driver.find_elements(By.CSS_SELECTOR, "[data-testid='status-tracker']")))
                    summaries = browser.find_elements(By.CSS_SELECTOR, "#evidence-cards summary")
                    summaries[-1].send_keys(Keys.ENTER)
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "#citation-1 button[data-citation]")))
                    browser.execute_script("arguments[0].scrollIntoView({block:'center', behavior:'instant'})", browser.find_element(By.CSS_SELECTOR, "#citation-1 button[data-citation]"))
                    activate(browser, browser.find_element(By.CSS_SELECTOR, "#citation-1 button[data-citation]"), mobile=label == "mobile")
                    wait.until(lambda driver: "[1]" in driver.find_element(By.ID, "visual-grounding-html").text)
                    assert len(turns) == calls_before_inspection, "Source inspection must not run generation"
                    wait.until(EC.element_to_be_clickable((By.ID, "clear-chat"))).send_keys(Keys.ENTER)
                    wait.until(EC.visibility_of_element_located((By.ID, "research-welcome")))
                    activate(browser, browser.find_element(By.ID, "preset-compare"), mobile=label == "mobile")
                    wait.until(lambda driver: "Synthetic answer" in driver.find_element(By.ID, "research-chat").text)
                    assert turns[-1] == (True, False)
                    browser.find_element(By.CSS_SELECTOR, "#execution-details button").send_keys(Keys.ENTER)
                    details = browser.find_element(By.ID, "execution-summary").text
                    assert "Compare warming risks" in details and "Compare emissions pathways" in details
                    assert "Search calls: 1" in details and "Agent generation calls: 2" in details
                    assert "Not Run" in details
                    browser.find_element(By.CSS_SELECTOR, "#execution-details button").send_keys(Keys.ENTER)
                    wait.until(EC.element_to_be_clickable((By.ID, "clear-chat"))).send_keys(Keys.ENTER)
                    wait.until(EC.visibility_of_element_located((By.ID, "research-welcome")))
                    browser.find_element(By.CSS_SELECTOR, "#evidence-limits button").send_keys(Keys.ENTER)
                    activate(browser, browser.find_element(By.ID, "preset-limits"), mobile=label == "mobile")
                    wait.until(lambda driver: "Automated grounding check: Unavailable" in driver.find_element(By.ID, "research-chat").text)
                    assert turns[-1] == (False, True)
                    assert "grounding check was unavailable" in browser.find_element(By.ID, "request-status").text
                    wait.until(EC.element_to_be_clickable((By.ID, "clear-chat"))).send_keys(Keys.ENTER)
                    wait.until(EC.visibility_of_element_located((By.ID, "research-welcome")))
                    browser.find_element(By.CSS_SELECTOR, "#advanced-options button").send_keys(Keys.ENTER)
                    assert browser.find_element(By.CSS_SELECTOR, "#advanced-options input").is_displayed()
                    browser.find_element(By.CSS_SELECTOR, "#advanced-options button").send_keys(Keys.ENTER)
                    activate(browser, browser.find_element(By.XPATH, "//button[@role='tab' and contains(., 'Explore reports')]"), mobile=label == "mobile")
                    wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "report-card")))
                    assert len(browser.find_elements(By.CLASS_NAME, "report-card")) == 3
                    for link in browser.find_elements(By.CSS_SELECTOR, ".report-card a"):
                        assert link.get_attribute("href").startswith("https://www.ipcc.ch/")
                    browser.save_screenshot(str(args.artifacts / (label + "-reports.png")))
                    if args.full_journey:
                        visitor_recovery_checks(browser, wait, scenario, turns, label == "mobile")
                        visitor_memory_checks(browser, wait, base_url, memory_backend, label == "mobile",
                                              marker="synthetic-memory-" + label, forget=label != "mobile")
                    print(f"PASS {label}: real graph progress, decomposition/details, unavailable grounding, citations/pages, historical sources, follow-ups/clear, keyboard, reports and layout")
                if args.full_journey:
                    other_options = webdriver.ChromeOptions()
                    other_options.binary_location = args.chrome
                    other_options.add_argument("--headless=new")
                    other_options.add_argument(f"--user-data-dir={root / 'independent-browser-profile'}")
                    other_browser = webdriver.Chrome(options=other_options)
                    other_browser.get(base_url)
                    other_wait = WebDriverWait(other_browser, 30)
                    other_wait.until(EC.element_to_be_clickable((By.ID, "preset-fact")))
                    assert COOKIE_NAME not in {cookie["name"] for cookie in other_browser.get_cookies()}
                    assert "synthetic-memory-" not in other_browser.find_element(By.ID, "research-memory").text
                    visitor_memory_checks(other_browser, other_wait, base_url, memory_backend, False,
                                          marker="synthetic-independent-memory", forget=False)
                    assert "synthetic-memory-mobile" not in other_browser.find_element(By.ID, "research-memory").text
                    assert "synthetic-independent-memory" not in browser.find_element(By.ID, "research-memory").text
                    assert "synthetic-memory-mobile" in browser.find_element(By.ID, "research-memory").text
                    forget_memory(browser, wait, True, "synthetic-memory-mobile")
                    other_browser.get(base_url)
                    other_wait.until(EC.element_to_be_clickable((By.ID, "preset-fact")))
                    open_memory(other_browser)
                    other_wait.until(lambda driver: "synthetic-independent-memory" in driver.find_element(By.ID, "research-memory").text)
                    activate(other_browser, other_browser.find_element(By.ID, "end-private-session"), False)
                    other_wait.until(lambda driver: "Public guest mode" in driver.find_element(By.ID, "private-session-status").text)
                    assert "synthetic-independent-memory" not in other_browser.find_element(By.ID, "research-memory").text
                    assert any("synthetic-independent-memory" in str(row.memory) for row in memory_backend.rows.values())
                    print("PASS independent browser: isolated guest identity, own save/reload/forget")
        except Exception:
            if browser is not None:
                try:
                    browser.save_screenshot(str(args.artifacts / "failure.png"))
                    print("Browser console:", browser.get_log("browser"))
                    print("Synthetic guest signups:", memory_backend.signups,
                          "Private cookie present:", COOKIE_NAME in {c["name"] for c in browser.get_cookies()})
                except Exception as diagnostic_error:
                    print("Browser diagnostics unavailable:", type(diagnostic_error).__name__)
            raise
        finally:
            if browser is not None:
                browser.quit()
            if other_browser is not None:
                other_browser.quit()
            if demo is not None:
                demo.close()


if __name__ == "__main__":
    main()
