from __future__ import annotations

import asyncio
import aiohttp
import csv
import json
import os
import ssl
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

API_URL = "https://www.kaggle.com/api/i/datasets.DatasetService/SearchDatasets"
KAGGLE_DATASETS_URL = "https://www.kaggle.com/datasets"

# Progress file to track state
PROGRESS_FILE = "kaggle_scraper_progress.json"
OUTPUT_FILE = "kaggle_datasets.csv"

HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Origin": "https://www.kaggle.com",
    "Referer": "https://www.kaggle.com/datasets",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

BASE_PAYLOAD = {
    "categoryIds": [],
    "feedbackIds": [],
    "fileType": "DATASET_FILE_TYPE_GROUP_ALL",
    "group": "DATASET_SELECTION_GROUP_PUBLIC",
    "includeTopicalDatasets": False,
    "license": "DATASET_LICENSE_GROUP_ALL",
    "minUsabilityRating": 0,
    "search": "",
    "size": "DATASET_SIZE_GROUP_ALL",
    "sortBy": "DATASET_SORT_BY_HOTTEST",
    "viewed": "DATASET_VIEWED_GROUP_UNSPECIFIED",
}

SORT_OPTIONS = [
    "DATASET_SORT_BY_HOTTEST",
    "DATASET_SORT_BY_VOTES",
    "DATASET_SORT_BY_UPDATED",
    "DATASET_SORT_BY_ACTIVE",
    "DATASET_SORT_BY_PUBLISHED",
    "DATASET_SORT_BY_USABILITY",
]

# CSV column headers - all fields from the API response
CSV_COLUMNS = [
    "dataset_id", "dataset_url", "dataset_slug", "rank", "medal_url",
    "has_hash_link", "firestore_path", "owner_name", "owner_url",
    "owner_user_id", "owner_tier", "owner_avatar_url", "creator_name",
    "creator_url", "creator_user_id", "view_count", "download_count",
    "script_count", "total_votes", "scripts_url", "forum_url", "download_url",
    "new_kernel_notebook_url", "new_kernel_script_url", "date_created",
    "date_updated", "license_name", "license_short_name", "dataset_size",
    "common_file_types", "categories", "category_names", "usability_score",
    "usability_column_description_score", "usability_cover_image_score",
    "usability_file_description_score", "usability_file_format_score",
    "usability_license_score", "usability_overview_score",
    "usability_provenance_score", "usability_public_kernel_score",
    "usability_subtitle_score", "usability_tag_score",
    "usability_update_frequency_score", "datasource_dataset_id",
    "datasource_current_version_id", "datasource_current_version_number",
    "datasource_type", "datasource_diff_type", "datasource_title",
    "datasource_overview", "datasource_thumbnail_url",
]


def extract_dataset_info(item: dict[str, Any]) -> dict[str, Any]:
    """Extract all fields from a dataset item."""
    vote_button = item.get("voteButton", {})
    usability = item.get("usabilityRating", {})
    datasource = item.get("datasource", {})
    categories = item.get("categories", [])
    category_names = ", ".join([cat.get("name", "") for cat in categories])

    return {
        "dataset_id": vote_button.get("datasetId", ""),
        "dataset_url": item.get("datasetUrl", ""),
        "dataset_slug": item.get("datasetSlug", ""),
        "rank": item.get("rank", ""),
        "medal_url": item.get("medalUrl", ""),
        "has_hash_link": item.get("hasHashLink", False),
        "firestore_path": item.get("firestorePath", ""),
        "owner_name": item.get("ownerName", ""),
        "owner_url": item.get("ownerUrl", ""),
        "owner_user_id": item.get("ownerUserId", ""),
        "owner_tier": item.get("ownerTier", ""),
        "owner_avatar_url": item.get("ownerAvatarUrl", ""),
        "creator_name": item.get("creatorName", ""),
        "creator_url": item.get("creatorUrl", ""),
        "creator_user_id": item.get("creatorUserId", ""),
        "view_count": item.get("viewCount", 0),
        "download_count": item.get("downloadCount", 0),
        "script_count": item.get("scriptCount", 0),
        "total_votes": vote_button.get("totalVotes", 0),
        "scripts_url": item.get("scriptsUrl", ""),
        "forum_url": item.get("forumUrl", ""),
        "download_url": item.get("downloadUrl", ""),
        "new_kernel_notebook_url": item.get("newKernelNotebookUrl", ""),
        "new_kernel_script_url": item.get("newKernelScriptUrl", ""),
        "date_created": item.get("dateCreated", ""),
        "date_updated": item.get("dateUpdated", ""),
        "license_name": item.get("licenseName", ""),
        "license_short_name": item.get("licenseShortName", ""),
        "dataset_size": item.get("datasetSize", 0),
        "common_file_types": json.dumps(item.get("commonFileTypes", [])),
        "categories": json.dumps(categories),
        "category_names": category_names,
        "usability_score": usability.get("score", 0),
        "usability_column_description_score": usability.get("columnDescriptionScore", 0),
        "usability_cover_image_score": usability.get("coverImageScore", 0),
        "usability_file_description_score": usability.get("fileDescriptionScore", 0),
        "usability_file_format_score": usability.get("fileFormatScore", 0),
        "usability_license_score": usability.get("licenseScore", 0),
        "usability_overview_score": usability.get("overviewScore", 0),
        "usability_provenance_score": usability.get("provenanceScore", 0),
        "usability_public_kernel_score": usability.get("publicKernelScore", 0),
        "usability_subtitle_score": usability.get("subtitleScore", 0),
        "usability_tag_score": usability.get("tagScore", 0),
        "usability_update_frequency_score": usability.get("updateFrequencyScore", 0),
        "datasource_dataset_id": datasource.get("datasetId", ""),
        "datasource_current_version_id": datasource.get("currentDatasetVersionId", ""),
        "datasource_current_version_number": datasource.get("currentDatasetVersionNumber", ""),
        "datasource_type": datasource.get("type", ""),
        "datasource_diff_type": datasource.get("diffType", ""),
        "datasource_title": datasource.get("title", ""),
        "datasource_overview": datasource.get("overview", ""),
        "datasource_thumbnail_url": datasource.get("thumbnailImageUrl", ""),
    }


def load_progress() -> dict:
    """Load progress from file."""
    if Path(PROGRESS_FILE).exists():
        try:
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading progress: {e}")
    return {"completed_sorts": [], "seen_ids": [], "total_scraped": 0}


def save_progress(progress: dict) -> None:
    """Save progress to file."""
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f)
    except Exception as e:
        print(f"Error saving progress: {e}")


def load_existing_csv() -> tuple[list[dict], set]:
    """Load existing CSV data and return datasets and seen IDs."""
    datasets = []
    seen_ids = set()

    if Path(OUTPUT_FILE).exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    datasets.append(row)
                    ds_id = row.get("dataset_id", "")
                    if ds_id:
                        seen_ids.add(str(ds_id))
            print(f"Loaded {len(datasets)} existing datasets from {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error loading existing CSV: {e}")

    return datasets, seen_ids


def append_to_csv(datasets: list[dict], filename: str = OUTPUT_FILE) -> None:
    """Append datasets to CSV file."""
    if not datasets:
        return

    file_exists = Path(filename).exists()

    try:
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerows(datasets)
    except Exception as e:
        print(f"Error appending to CSV: {e}")
        # Try backup file
        backup = f"kaggle_datasets_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(backup, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(datasets)
            print(f"Saved backup to {backup}")
        except Exception as e2:
            print(f"Error saving backup: {e2}")


async def init_session(session: aiohttp.ClientSession) -> str | None:
    """Initialize session by visiting the datasets page to get cookies and XSRF token."""
    for attempt in range(3):
        try:
            async with session.get(KAGGLE_DATASETS_URL) as response:
                if response.status == 200:
                    cookies = session.cookie_jar.filter_cookies(KAGGLE_DATASETS_URL)
                    for cookie in cookies.values():
                        if cookie.key == "XSRF-TOKEN":
                            print(f"Session initialized, XSRF token found")
                            return cookie.value
                print(f"Failed to initialize session: HTTP {response.status}")
        except Exception as e:
            print(f"Error initializing session (attempt {attempt + 1}): {e}")
            await asyncio.sleep(2 ** attempt)
    return None


async def scrape_with_sort(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    xsrf_token: str | None,
    sort_by: str,
    seen_ids: set,
    max_pages: int | None = None,
    save_interval: int = 50,
) -> list[dict]:
    """Scrape datasets using a specific sort order with crash protection."""
    all_new_datasets = []
    batch_datasets = []
    page = 1
    consecutive_errors = 0
    max_consecutive_errors = 10

    print(f"\n{'='*50}")
    print(f"Scraping with sort: {sort_by}")
    print(f"{'='*50}")

    while True:
        if max_pages and page > max_pages:
            break

        if consecutive_errors >= max_consecutive_errors:
            print(f"Too many consecutive errors, stopping {sort_by}")
            break

        payload = {**BASE_PAYLOAD, "page": page, "sortBy": sort_by}
        headers = {**HEADERS}
        if xsrf_token:
            headers["X-XSRF-TOKEN"] = xsrf_token

        try:
            async with semaphore:
                async with session.post(API_URL, json=payload, headers=headers) as response:
                    if response.status == 404:
                        consecutive_errors += 1
                        if consecutive_errors >= 3:
                            print(f"Hit page limit at page {page} for {sort_by}")
                            break
                        page += 1
                        continue

                    if response.status == 429:
                        wait_time = min(60, 2 ** consecutive_errors * 5)
                        print(f"Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        consecutive_errors += 1
                        continue

                    if response.status != 200:
                        print(f"Page {page}: HTTP {response.status}")
                        consecutive_errors += 1
                        page += 1
                        await asyncio.sleep(1)
                        continue

                    consecutive_errors = 0
                    data = await response.json()
                    dataset_list = data.get("datasetList", {})
                    items = dataset_list.get("items", [])
                    has_more = data.get("hasMore", False)

                    new_count = 0
                    for item in items:
                        try:
                            extracted = extract_dataset_info(item)
                            ds_id = str(extracted.get("dataset_id", ""))
                            if ds_id and ds_id not in seen_ids:
                                seen_ids.add(ds_id)
                                batch_datasets.append(extracted)
                                all_new_datasets.append(extracted)
                                new_count += 1
                        except Exception as e:
                            print(f"Error extracting item: {e}")

                    print(f"Page {page}: {len(items)} items, {new_count} new (total unique: {len(seen_ids)})")

                    # Save batch to CSV periodically
                    if len(batch_datasets) >= save_interval:
                        append_to_csv(batch_datasets)
                        print(f"  -> Saved {len(batch_datasets)} datasets to CSV")
                        batch_datasets = []

                    if not has_more or len(items) == 0:
                        print(f"No more data for {sort_by}")
                        break

                    page += 1
                    await asyncio.sleep(0.1)

        except asyncio.TimeoutError:
            print(f"Page {page}: Timeout")
            consecutive_errors += 1
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            # Save remaining data before exit
            if batch_datasets:
                append_to_csv(batch_datasets)
                print(f"Cancelled - saved {len(batch_datasets)} datasets")
            raise
        except Exception as e:
            print(f"Page {page}: Error - {e}")
            consecutive_errors += 1
            await asyncio.sleep(2)

    # Save remaining batch
    if batch_datasets:
        append_to_csv(batch_datasets)
        print(f"  -> Saved remaining {len(batch_datasets)} datasets to CSV")

    return all_new_datasets


async def scrape_all_datasets(
    max_concurrent: int = 10,
    max_pages: int | None = None,
) -> int:
    """Scrape all datasets using multiple sort orders with crash protection."""

    # Load previous progress
    progress = load_progress()
    completed_sorts = set(progress.get("completed_sorts", []))

    # Load existing data
    existing_datasets, seen_ids = load_existing_csv()
    # Merge saved IDs (convert to strings for consistency)
    seen_ids.update(str(id) for id in progress.get("seen_ids", []))

    total_new = 0

    print(f"\nResuming from progress:")
    print(f"  Completed sorts: {len(completed_sorts)}")
    print(f"  Existing datasets: {len(existing_datasets)}")
    print(f"  Seen IDs: {len(seen_ids)}")

    # Create SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60)
    cookie_jar = aiohttp.CookieJar(unsafe=True)
    semaphore = asyncio.Semaphore(max_concurrent)

    try:
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            cookie_jar=cookie_jar
        ) as session:
            # Initialize session
            print("\nInitializing session...")
            xsrf_token = await init_session(session)

            if not xsrf_token:
                print("Warning: No XSRF token found, requests may fail")

            # Scrape using each sort order
            for sort_by in SORT_OPTIONS:
                if sort_by in completed_sorts:
                    print(f"\nSkipping already completed: {sort_by}")
                    continue

                try:
                    new_datasets = await scrape_with_sort(
                        session, semaphore, xsrf_token, sort_by, seen_ids, max_pages
                    )
                    total_new += len(new_datasets)

                    # Mark as completed and save progress
                    completed_sorts.add(sort_by)
                    progress["completed_sorts"] = list(completed_sorts)
                    progress["seen_ids"] = [str(id) for id in seen_ids]
                    progress["total_scraped"] = len(seen_ids)
                    save_progress(progress)

                    print(f"Completed {sort_by}: {len(new_datasets)} new datasets")

                except asyncio.CancelledError:
                    print(f"\nCancelled during {sort_by}")
                    save_progress(progress)
                    raise
                except Exception as e:
                    print(f"Error in {sort_by}: {e}")
                    # Continue with next sort option

    except Exception as e:
        print(f"\nSession error: {e}")
        save_progress(progress)

    return total_new


async def main():
    print("=" * 60)
    print("Kaggle Dataset Scraper (Crash-Proof)")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Progress file: {PROGRESS_FILE}")
    print()

    try:
        total_new = await scrape_all_datasets(
            max_concurrent=10,
            max_pages=None,
        )

        # Count total in CSV
        total_in_csv = 0
        if Path(OUTPUT_FILE).exists():
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                total_in_csv = sum(1 for _ in f) - 1  # Subtract header

        print(f"\n{'='*60}")
        print(f"SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"Finished at: {datetime.now()}")
        print(f"New datasets this run: {total_new}")
        print(f"Total datasets in CSV: {total_in_csv}")
        print(f"Output file: {OUTPUT_FILE}")

        # Clean up progress file on successful completion
        progress = load_progress()
        if len(progress.get("completed_sorts", [])) == len(SORT_OPTIONS):
            print("\nAll sort options completed. Cleaning up progress file...")
            if Path(PROGRESS_FILE).exists():
                os.remove(PROGRESS_FILE)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        print("Progress saved. Re-run to resume.")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
