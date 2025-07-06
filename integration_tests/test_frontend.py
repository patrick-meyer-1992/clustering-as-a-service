import re
from playwright.sync_api import Page, expect

STREAMLIT_URL = "http://localhost:8501/"
def test_select_dataset(page: Page) -> None:
    page.goto(STREAMLIT_URL)
    page.get_by_role("button").get_by_text("✅").click()
    expect(page.get_by_role("alert").filter(has_text="Datensatz ausgewählt: iris.")).to_be_visible()
    expect(page.get_by_test_id("stMainBlockContainer")).to_contain_text("Datensatz ausgewählt: iris.csv")
    page.get_by_test_id("stMain").click()
