"""TinyFish acquisition and data validation for DataForge AI."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tinyfish import TinyFish, CompleteEvent

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "laptop_listings.csv"

FIELDS = [
    "brand", "model", "cpu", "gpu", "ram_gb", "storage_gb",
    "age_years", "condition", "price_inr", "source_url", "source_title",
    "collected_at",
]

def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def _parse_json_payload(payload: Any) -> dict:
    """Accept SDK JSON objects or JSON embedded in a text response."""
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        payload = str(payload)
    payload = payload.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", payload, flags=re.S)
        if match:
            return json.loads(match.group(0))
    raise ValueError("TinyFish did not return valid JSON.")

def _number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    digits = re.sub(r"[^0-9.]", "", str(value).replace(",", ""))
    try:
        return float(digits) if digits else None
    except ValueError:
        return None

def normalize_listing(item: dict, url: str, title: str) -> dict | None:
    """Normalize one extracted listing; do not invent missing values."""
    price = _number(item.get("price_inr", item.get("price")))
    if price is None or not (3000 <= price <= 1_000_000):
        return None

    ram = _number(item.get("ram_gb", item.get("ram")))
    storage = _number(item.get("storage_gb", item.get("storage")))
    age = _number(item.get("age_years", item.get("age")))
    condition = str(item.get("condition") or "Unknown").strip().title()
    if condition not in {"Excellent", "Good", "Fair", "Poor", "Unknown", "New"}:
        condition = "Unknown"

    return {
        "brand": str(item.get("brand") or "Unknown").strip(),
        "model": str(item.get("model") or item.get("title") or title or "Unknown").strip(),
        "cpu": str(item.get("cpu") or "Unknown").strip(),
        "gpu": str(item.get("gpu") or "Unknown").strip(),
        "ram_gb": ram,
        "storage_gb": storage,
        "age_years": age,
        "condition": condition,
        "price_inr": price,
        "source_url": str(item.get("source_url") or url).strip(),
        "source_title": str(item.get("source_title") or title).strip(),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

def _agent_extract(client: TinyFish, url: str) -> list[dict]:
    goal = """
You are extracting public used gaming-laptop marketplace listings for a research dataset.
Read the page and return ONLY valid JSON in this exact shape:
{
  "listings": [
    {
      "brand": "string or Unknown",
      "model": "listing title/model",
      "cpu": "CPU model or Unknown",
      "gpu": "GPU model or Unknown",
      "ram_gb": number or null,
      "storage_gb": number or null,
      "age_years": number or null,
      "condition": "Excellent, Good, Fair, Poor, New, or Unknown",
      "price_inr": number,
      "source_url": "absolute URL for this individual listing if visible"
    }
  ]
}
Rules:
- Extract only listings actually visible on this page. Never invent or infer a price.
- Only include laptops explicitly described as used, refurbished, or whose condition is stated;
  if the page is clearly a used-laptop marketplace, listings may be included when condition is unknown.
- Convert prices to numeric INR. If currency is not INR or price is unclear, omit that listing.
- RAM/storage should be numeric GB (e.g. 16, 512). Convert 1 TB to 1024 GB.
- age_years should be null unless explicitly stated.
- Keep unknown specs as "Unknown" or null.
- Return an empty listings array if no relevant listing data is present.
- Return at most 15 listings from this page.
"""
    found = []
    with client.agent.stream(url=url, goal=goal) as stream:
        for event in stream:
            if isinstance(event, CompleteEvent):
                payload = _get(event, "result_json")
                if payload is None:
                    payload = _get(event, "result")
                parsed = _parse_json_payload(payload)
                found = parsed.get("listings", []) or []
    return found if isinstance(found, list) else []

def collect_laptops(query: str = "used gaming laptop RTX India price", max_sources: int = 8,
                    progress=None) -> tuple[pd.DataFrame, list[str]]:
    """TinyFish Search -> Fetch -> Agent extraction, then normalize/dedupe."""
    if not __import__("os").environ.get("TINYFISH_API_KEY"):
        raise RuntimeError("TINYFISH_API_KEY is missing. Add it to .env and restart Streamlit.")

    client = TinyFish()
    errors: list[str] = []
    if progress:
        progress("Searching the live web with TinyFish…")

    search_response = client.search.query(
        query=query,
        location="IN",
        purpose="Find public Indian listings for used gaming laptops with visible prices and specifications.",
    )
    results = list(_get(search_response, "results", []) or [])
    candidates = []
    for result in results:
        url = _get(result, "url")
        title = _get(result, "title", "")
        if url and str(url).startswith("http"):
            candidates.append((str(url), str(title)))
    candidates = candidates[:max_sources]

    if not candidates:
        raise RuntimeError("TinyFish Search returned no usable URLs. Try a broader query.")

    # Fetch the candidate pages first; retain provenance and determine whether extraction is viable.
    urls = [url for url, _ in candidates]
    if progress:
        progress(f"Fetching {len(urls)} candidate pages…")
    try:
        fetch_response = client.fetch.get_contents(urls=urls, format="markdown")
        fetched = list(_get(fetch_response, "results", []) or [])
        fetch_by_url = {str(_get(page, "url", "")): page for page in fetched}
    except Exception as exc:
        fetch_by_url = {}
        errors.append(f"Fetch stage: {exc}")

    rows = []
    for idx, (url, title) in enumerate(candidates, start=1):
        if progress:
            progress(f"Agent extraction {idx}/{len(candidates)}: {title[:70] or url}")
        page = fetch_by_url.get(url)
        page_text = str(_get(page, "text", "") or "")
        # Skip obvious non-listing pages unless their fetched text appears listing-related.
        hints = ("₹", "inr", "rtx", "gaming laptop", "price", "laptop")
        if page_text and not any(h in page_text.lower() for h in hints):
            errors.append(f"Skipped page with no obvious laptop/price text: {url}")
            continue
        try:
            extracted = _agent_extract(client, url)
            for item in extracted:
                if not isinstance(item, dict):
                    continue
                row = normalize_listing(item, url, title)
                if row:
                    rows.append(row)
        except Exception as exc:
            errors.append(f"Agent failed for {url}: {exc}")

    columns = FIELDS
    new_df = pd.DataFrame(rows, columns=columns)
    if DATA_PATH.exists():
        try:
            old_df = pd.read_csv(DATA_PATH)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            pass
    if not new_df.empty:
        # Keep the latest record for the same URL + price + title.
        new_df = new_df.drop_duplicates(
            subset=["source_url", "price_inr", "model"], keep="last"
        )
        new_df.to_csv(DATA_PATH, index=False)
    return new_df, errors

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=FIELDS)
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return pd.DataFrame(columns=FIELDS)
