"""Optional real-Chrome visitor smoke check with synthetic sources/providers.

Run: python -m tests.browser_demo_ux --artifacts /path/to/temporary-directory
Requires Selenium and a local Chrome browser, not a live Gemini/Supabase account.
Screenshots and browser profile/cache are kept in the supplied temporary directory.
"""

import argparse
import os
import re
from pathlib import Path
import socket
import tempfile
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    os.environ["SE_CACHE_PATH"] = str(args.artifacts / "selenium-cache")
    browser = None
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

        def synthetic_turn(**kwargs):
            turns.append((kwargs["use_decomposition"], kwargs["use_guardrail"]))
            context = kwargs["tool_map"]["search_knowledge_base"]("synthetic climate query")
            number = re.search(r"Evidence \[(\d+)\]", context).group(1)
            kwargs["conversation_history"].append("synthetic completed turn")
            # Deliberately omit all image URLs: structured evidence must survive.
            return f"Synthetic answer for browser verification [{number}]. Invalid reference [999]."

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        try:
            with patch("app.run_agent_turn", side_effect=synthetic_turn), \
                    patch("agent.search_chroma", return_value=[retrieved]), \
                    patch("hybrid_search.search_chroma_hybrid", return_value=[retrieved]):
                demo = build_ui(None, {}, "unused.json", None, collection, None, showcase=corpus)
                demo.queue().launch(server_name="127.0.0.1", server_port=port,
                                    prevent_thread_lock=True, blocked_paths=[str(root)],
                                    app_kwargs={"routes": corpus.routes(collection, RuntimePolicy())},
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
                    browser.set_window_size(width, height)
                    if label == "mobile":
                        browser.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
                            "width": width, "height": height, "deviceScaleFactor": 1, "mobile": True})
                    browser.get(f"http://127.0.0.1:{port}")
                    wait.until(EC.element_to_be_clickable((By.ID, "preset-fact")))
                    wait.until(lambda driver: not any(element.is_displayed() for element in
                        driver.find_elements(By.CSS_SELECTOR, "[data-testid='status-tracker']")))
                    check_layout(browser, mobile=label == "mobile")
                    browser.save_screenshot(str(args.artifacts / (label + "-welcome.png")))
                    activate(browser, browser.find_element(By.ID, "preset-fact"), mobile=label == "mobile")
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
                    print(f"PASS {label}: validated citations, opaque page images, retained earlier sources, contextual follow-ups, presets/clear, keyboard, reports and layout")
        except Exception:
            if browser is not None:
                browser.save_screenshot(str(args.artifacts / "failure.png"))
                print("Browser console:", browser.get_log("browser"))
            raise
        finally:
            if browser is not None:
                browser.quit()
            if demo is not None:
                demo.close()


if __name__ == "__main__":
    main()
