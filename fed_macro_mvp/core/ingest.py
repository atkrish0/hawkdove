from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from .config import PipelineConfig


def normalize_pdf_url(href: str, base_url: str) -> str | None:
    href = (href or "").strip()
    if not href:
        return None
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://www.federalreserve.gov" + href
    elif not href.lower().startswith("http"):
        href = requests.compat.urljoin(base_url, href)

    if ".pdf" not in href.lower() or "federalreserve.gov" not in href.lower():
        return None
    return href.split("#")[0]


def extract_date_hint(text: str) -> str | None:
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", text)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{mo}-{d}"
    y = re.search(r"(20\d{2})", text)
    return f"{y.group(1)}-01-01" if y else None


def scrape_seed_pages(cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; hawkdove-mvp/1.0)"})

    for page in cfg.seed_pages:
        try:
            r = session.get(page, timeout=cfg.request_timeout)
            if r.status_code != 200:
                continue
        except Exception:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a"):
            pdf_url = normalize_pdf_url(a.get("href"), page)
            if not pdf_url:
                continue
            title = " ".join(a.get_text(" ", strip=True).split()) or Path(pdf_url).name
            rows.append(
                {
                    "source": "seed_page",
                    "source_page": page,
                    "pdf_url": pdf_url,
                    "title": title,
                    "date_hint": extract_date_hint(pdf_url),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["source", "source_page", "pdf_url", "title", "date_hint"])
    return pd.DataFrame(rows).drop_duplicates(subset=["pdf_url"]).reset_index(drop=True)


def probe_known_patterns(cfg: PipelineConfig) -> pd.DataFrame:
    base = "https://www.federalreserve.gov/monetarypolicy/files/"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; hawkdove-mvp/1.0)"})

    rows = []
    today = datetime.now(timezone.utc).date()

    for offset in tqdm(range(cfg.days_back), desc="Probing patterns"):
        dt = today - timedelta(days=offset)
        ymd = dt.strftime("%Y%m%d")

        for fname in [f"fomcminutes{ymd}.pdf", f"monetary{ymd}a1.pdf"]:
            url = base + fname
            try:
                r = session.head(url, allow_redirects=True, timeout=8)
            except Exception:
                continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            if r.status_code == 200 and ("pdf" in ctype or url.endswith(".pdf")):
                rows.append(
                    {
                        "source": "known_pattern",
                        "source_page": "monetarypolicy/files",
                        "pdf_url": url,
                        "title": fname,
                        "date_hint": dt.isoformat(),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["source", "source_page", "pdf_url", "title", "date_hint"])
    return pd.DataFrame(rows).drop_duplicates(subset=["pdf_url"]).reset_index(drop=True)


def build_catalog(cfg: PipelineConfig) -> pd.DataFrame:
    seed = scrape_seed_pages(cfg)
    patt = probe_known_patterns(cfg)
    combined = pd.concat([seed, patt], ignore_index=True).drop_duplicates(subset=["pdf_url"])
    if combined.empty:
        return combined

    combined["parsed_date"] = pd.to_datetime(combined["date_hint"], errors="coerce")
    return combined.sort_values("parsed_date", ascending=False, na_position="last").reset_index(drop=True)


def download_catalog(catalog_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if catalog_df.empty:
        return catalog_df

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; hawkdove-mvp/1.0)"})
    rows = []

    for _, row in catalog_df.head(cfg.max_pdfs).iterrows():
        url = row["pdf_url"]
        fname = Path(url).name
        local = cfg.raw_pdf_dir / fname

        if local.exists() and local.stat().st_size > 0:
            rows.append({**row.to_dict(), "local_path": str(local), "status": "exists"})
            continue

        try:
            r = session.get(url, timeout=cfg.request_timeout)
            if r.status_code == 200 and r.content:
                local.write_bytes(r.content)
                rows.append({**row.to_dict(), "local_path": str(local), "status": "downloaded"})
            else:
                rows.append({**row.to_dict(), "local_path": str(local), "status": f"http_{r.status_code}"})
        except Exception as e:
            rows.append({**row.to_dict(), "local_path": str(local), "status": f"error:{e}"})

    return pd.DataFrame(rows)


def run_ingestion(cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    catalog_df = build_catalog(cfg)
    download_df = download_catalog(catalog_df, cfg)
    manifest_path = None
    if not download_df.empty:
        manifest_path = cfg.processed_dir / "download_manifest.csv"
        download_df.to_csv(manifest_path, index=False)
    return catalog_df, download_df, manifest_path
