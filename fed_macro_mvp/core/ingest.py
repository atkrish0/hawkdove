from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import time

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


def classify_doc_type(url_or_name: str) -> str:
    name = Path(str(url_or_name or "")).name.lower()
    if re.search(r"fomcminutes20\d{6}\.pdf$", name):
        return "fomc_minutes"
    if re.search(r"monetary20\d{6}a1\.pdf$", name):
        return "mpr"
    if "mprfullreport" in name:
        return "mpr"
    return "other"


def scrape_seed_pages(cfg: PipelineConfig, observer=None) -> pd.DataFrame:
    t0 = time.time()
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
                    "doc_type": classify_doc_type(pdf_url),
                }
            )

    if not rows:
        out = pd.DataFrame(columns=["source", "source_page", "pdf_url", "title", "date_hint", "doc_type"])
    else:
        out = pd.DataFrame(rows).drop_duplicates(subset=["pdf_url"]).reset_index(drop=True)
    if observer is not None:
        observer.emit(
            "stage_completed",
            stage="ingest_seed_pages",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={"seed_pages": len(cfg.seed_pages), "candidate_links": int(len(out))},
        )
    return out


def probe_known_patterns(cfg: PipelineConfig, observer=None) -> pd.DataFrame:
    t0 = time.time()
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
                        "doc_type": classify_doc_type(fname),
                    }
                )

    if not rows:
        out = pd.DataFrame(columns=["source", "source_page", "pdf_url", "title", "date_hint", "doc_type"])
    else:
        out = pd.DataFrame(rows).drop_duplicates(subset=["pdf_url"]).reset_index(drop=True)
    if observer is not None:
        observer.emit(
            "stage_completed",
            stage="ingest_pattern_probe",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={"days_back": int(cfg.days_back), "candidate_links": int(len(out))},
        )
    return out


def apply_catalog_filters(
    catalog_df: pd.DataFrame,
    cfg: PipelineConfig,
    today_utc: datetime | None = None,
    observer=None,
) -> pd.DataFrame:
    if catalog_df.empty:
        return catalog_df

    initial_count = int(len(catalog_df))
    out = catalog_df.copy()
    if "doc_type" not in out.columns:
        out["doc_type"] = out["pdf_url"].apply(classify_doc_type)

    out["parsed_date"] = pd.to_datetime(out["date_hint"], errors="coerce", utc=True)
    out = out[out["parsed_date"].notna()].copy()
    if out.empty:
        return out

    if today_utc is None:
        today_utc = datetime.now(timezone.utc)
    today = today_utc.date()
    cutoff = today - timedelta(days=max(1, int(cfg.days_back)))
    out["_date"] = out["parsed_date"].dt.date
    out = out[(out["_date"] >= cutoff) & (out["_date"] <= today)].copy()

    allowed = {str(x).lower() for x in (cfg.allowed_doc_types or ["all"])}
    if allowed and "all" not in allowed:
        out = out[out["doc_type"].str.lower().isin(allowed)].copy()

    if out.empty:
        return out.drop(columns=["_date"], errors="ignore")

    out["age_days"] = (today_utc - out["parsed_date"]).dt.days
    out = out.sort_values(["parsed_date", "age_days"], ascending=[False, True], na_position="last")
    out = out.drop(columns=["_date"], errors="ignore").reset_index(drop=True)
    if observer is not None:
        doc_counts = (
            out["doc_type"].astype(str).value_counts().to_dict()
            if not out.empty and "doc_type" in out.columns
            else {}
        )
        observer.emit(
            "stage_completed",
            stage="catalog_filter",
            status="ok",
            payload={
                "input_candidates": initial_count,
                "filtered_candidates": int(len(out)),
                "allowed_doc_types": list(cfg.allowed_doc_types),
                "doc_type_counts": doc_counts,
                "days_back": int(cfg.days_back),
            },
        )
    return out


def build_catalog(cfg: PipelineConfig, observer=None) -> pd.DataFrame:
    t0 = time.time()
    seed = scrape_seed_pages(cfg, observer=observer)
    patt = probe_known_patterns(cfg, observer=observer)
    combined = pd.concat([seed, patt], ignore_index=True).drop_duplicates(subset=["pdf_url"])
    if combined.empty:
        if observer is not None:
            observer.emit(
                "stage_completed",
                stage="catalog_build",
                status="ok",
                duration_ms=(time.time() - t0) * 1000.0,
                payload={"seed_candidates": int(len(seed)), "pattern_candidates": int(len(patt)), "catalog_candidates": 0},
            )
        return combined
    filtered = apply_catalog_filters(combined, cfg, observer=observer)
    if observer is not None:
        observer.emit(
            "stage_completed",
            stage="catalog_build",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={
                "seed_candidates": int(len(seed)),
                "pattern_candidates": int(len(patt)),
                "catalog_candidates": int(len(filtered)),
            },
        )
    return filtered


def download_catalog(catalog_df: pd.DataFrame, cfg: PipelineConfig, observer=None) -> pd.DataFrame:
    t0 = time.time()
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

    out = pd.DataFrame(rows)
    if observer is not None:
        status_counts = out["status"].astype(str).value_counts().to_dict() if not out.empty else {}
        observer.emit(
            "stage_completed",
            stage="catalog_download",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={"download_attempted": int(len(out)), "status_counts": status_counts},
        )
    return out


def run_ingestion(cfg: PipelineConfig, observer=None) -> tuple[pd.DataFrame, pd.DataFrame, Path | None, dict[str, object]]:
    t0 = time.time()
    catalog_df = build_catalog(cfg, observer=observer)
    download_df = download_catalog(catalog_df, cfg, observer=observer)
    manifest_path = None
    if not download_df.empty:
        manifest_path = cfg.processed_dir / "download_manifest.csv"
        download_df.to_csv(manifest_path, index=False)

    available_count = 0
    if not download_df.empty and "status" in download_df.columns:
        available_count = int(download_df["status"].isin(["downloaded", "exists"]).sum())

    warnings = []
    if cfg.profile_name == "fast_default":
        if available_count > cfg.fast_profile_doc_warning:
            warnings.append(
                f"fast_default: available docs {available_count} exceed recommended cap {cfg.fast_profile_doc_warning}"
            )

    stats: dict[str, object] = {
        "profile_name": cfg.profile_name,
        "days_back": cfg.days_back,
        "max_pdfs": cfg.max_pdfs,
        "catalog_candidates": int(len(catalog_df)),
        "download_attempted": int(len(download_df)),
        "downloaded_or_exists": available_count,
        "warnings": warnings,
    }
    if observer is not None:
        observer.emit(
            "stage_completed",
            stage="ingestion",
            status="ok",
            duration_ms=(time.time() - t0) * 1000.0,
            payload={**stats, "manifest_path": str(manifest_path) if manifest_path else None},
        )
    return catalog_df, download_df, manifest_path, stats
