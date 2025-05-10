#!/usr/bin/env python3
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ── Configuration ───────────────────────────────────────────────────────────────
BASE_URL      = "https://datasheet4u.com"
FEATURED_PAGE = BASE_URL + "/Featured.html"
DOWNLOAD_DIR  = "datasheet4u_pdfs"

# Spoof a real browser User-Agent to avoid blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup parser."""
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")

def list_featured_parts() -> list[str]:
    """
    Scrape the Featured page and return a list of part-page URLs.
    We filter for <a href="XXX"> links where XXX contains no slash—
    these are the part codes.
    """
    soup = get_soup(FEATURED_PAGE)
    hrefs = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # skip navigation or external links (those contain '/')
        if "/" not in href:
            full = urljoin(BASE_URL + "/", href)
            hrefs.append(full)
    return sorted(set(hrefs))

def find_pdf_link(part_page_url: str) -> str | None:
    """
    On a part’s page, locate the first <a href="...pdf"> link
    (the ‘Download Datasheet’ button).
    """
    soup = get_soup(part_page_url)
    # find any anchor whose href ends with .pdf
    link = soup.find("a", href=re.compile(r"\.pdf$"))
    if link:
        return urljoin(BASE_URL, link["href"])
    return None

def download_pdf(pdf_url: str, out_path: str) -> bool:
    """Download the PDF at pdf_url and save it to out_path."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resp = requests.get(pdf_url, headers=HEADERS)
    if resp.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return True
    return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parts = list_featured_parts()
    if not parts:
        print("❌ No featured parts found on the page.")
        return

    print(f"Found {len(parts)} parts. Starting download…\n")
    for idx, part_url in enumerate(parts, 1):
        part_code = part_url.rstrip("/").split("/")[-1]
        print(f"[{idx}/{len(parts)}] {part_code}", end=" … ")

        try:
            pdf_url = find_pdf_link(part_url)
            if not pdf_url:
                print("no PDF link found, skipped.")
                continue

            filename = f"{part_code}.pdf"
            out_path = os.path.join(DOWNLOAD_DIR, filename)

            if download_pdf(pdf_url, out_path):
                print("downloaded.")
            else:
                print(f"failed to download (HTTP {requests.get(pdf_url).status_code}).")
        except Exception as e:
            print(f"error: {e!r}")

    print("\n✔ All done. PDFs are in “%s”." % DOWNLOAD_DIR)

if __name__ == "__main__":
    main()