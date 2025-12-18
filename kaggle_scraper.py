from __future__ import annotations

import asyncio
import aiohttp
import csv
import json
import ssl
from datetime import datetime
from typing import Any, Optional

API_URL = "https://www.kaggle.com/api/i/datasets.DatasetService/SearchDatasets"
KAGGLE_DATASETS_URL = "https://www.kaggle.com/datasets"

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

# CSV column headers - all fields from the API response
CSV_COLUMNS = [
    # Basic info
    "dataset_id",
    "dataset_url",
    "dataset_slug",
    "rank",
    "medal_url",
    "has_hash_link",
    "firestore_path",

    # Owner info
    "owner_name",
    "owner_url",
    "owner_user_id",
    "owner_tier",
    "owner_avatar_url",

    # Creator info
    "creator_name",
    "creator_url",
    "creator_user_id",

    # Stats
    "view_count",
    "download_count",
    "script_count",
    "total_votes",

    # URLs
    "scripts_url",
    "forum_url",
    "download_url",
    "new_kernel_notebook_url",
    "new_kernel_script_url",

    # Dates
    "date_created",
    "date_updated",

    # License
    "license_name",
    "license_short_name",

    # Size info
    "dataset_size",

    # File types (JSON string)
    "common_file_types",

    # Categories (JSON string)
    "categories",
    "category_names",

    # Usability rating
    "usability_score",
    "usability_column_description_score",
    "usability_cover_image_score",
    "usability_file_description_score",
    "usability_file_format_score",
    "usability_license_score",
    "usability_overview_score",
    "usability_provenance_score",
    "usability_public_kernel_score",
    "usability_subtitle_score",
    "usability_tag_score",
    "usability_update_frequency_score",

    # Datasource info
    "datasource_dataset_id",
    "datasource_current_version_id",
    "datasource_current_version_number",
    "datasource_type",
    "datasource_diff_type",
    "datasource_title",
    "datasource_overview",
    "datasource_thumbnail_url",
]


def extract_dataset_info(item: dict[str, Any]) -> dict[str, Any]:
    """Extract all fields from a dataset item."""
    vote_button = item.get("voteButton", {})
    usability = item.get("usabilityRating", {})
    datasource = item.get("datasource", {})
    categories = item.get("categories", [])

    # Extract category names as comma-separated string
    category_names = ", ".join([cat.get("name", "") for cat in categories])

    return {
        # Basic info
        "dataset_id": vote_button.get("datasetId", ""),
        "dataset_url": item.get("datasetUrl", ""),
        "dataset_slug": item.get("datasetSlug", ""),
        "rank": item.get("rank", ""),
        "medal_url": item.get("medalUrl", ""),
        "has_hash_link": item.get("hasHashLink", False),
        "firestore_path": item.get("firestorePath", ""),

        # Owner info
        "owner_name": item.get("ownerName", ""),
        "owner_url": item.get("ownerUrl", ""),
        "owner_user_id": item.get("ownerUserId", ""),
        "owner_tier": item.get("ownerTier", ""),
        "owner_avatar_url": item.get("ownerAvatarUrl", ""),

        # Creator info
        "creator_name": item.get("creatorName", ""),
        "creator_url": item.get("creatorUrl", ""),
        "creator_user_id": item.get("creatorUserId", ""),

        # Stats
        "view_count": item.get("viewCount", 0),
        "download_count": item.get("downloadCount", 0),
        "script_count": item.get("scriptCount", 0),
        "total_votes": vote_button.get("totalVotes", 0),

        # URLs
        "scripts_url": item.get("scriptsUrl", ""),
        "forum_url": item.get("forumUrl", ""),
        "download_url": item.get("downloadUrl", ""),
        "new_kernel_notebook_url": item.get("newKernelNotebookUrl", ""),
        "new_kernel_script_url": item.get("newKernelScriptUrl", ""),

        # Dates
        "date_created": item.get("dateCreated", ""),
        "date_updated": item.get("dateUpdated", ""),

        # License
        "license_name": item.get("licenseName", ""),
        "license_short_name": item.get("licenseShortName", ""),

        # Size info
        "dataset_size": item.get("datasetSize", 0),

        # File types (as JSON string)
        "common_file_types": json.dumps(item.get("commonFileTypes", [])),

        # Categories (as JSON string and names)
        "categories": json.dumps(categories),
        "category_names": category_names,

        # Usability rating
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

        # Datasource info
        "datasource_dataset_id": datasource.get("datasetId", ""),
        "datasource_current_version_id": datasource.get("currentDatasetVersionId", ""),
        "datasource_current_version_number": datasource.get("currentDatasetVersionNumber", ""),
        "datasource_type": datasource.get("type", ""),
        "datasource_diff_type": datasource.get("diffType", ""),
        "datasource_title": datasource.get("title", ""),
        "datasource_overview": datasource.get("overview", ""),
        "datasource_thumbnail_url": datasource.get("thumbnailImageUrl", ""),
    }


async def init_session(session: aiohttp.ClientSession) -> str | None:
    """Initialize session by visiting the datasets page to get cookies and XSRF token."""
    try:
        async with session.get(KAGGLE_DATASETS_URL) as response:
            if response.status == 200:
                # Get XSRF token from cookies
                cookies = session.cookie_jar.filter_cookies(KAGGLE_DATASETS_URL)
                xsrf_token = None
                for cookie in cookies.values():
                    if cookie.key == "XSRF-TOKEN":
                        xsrf_token = cookie.value
                        break
                print(f"Session initialized, XSRF token: {'found' if xsrf_token else 'not found'}")
                return xsrf_token
            else:
                print(f"Failed to initialize session: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error initializing session: {e}")
        return None


async def fetch_page(
    session: aiohttp.ClientSession,
    page: int,
    semaphore: asyncio.Semaphore,
    xsrf_token: str | None = None,
    retry_count: int = 3,
) -> tuple[list[dict], bool, int]:
    """Fetch a single page of datasets. Returns (datasets, has_more, total_results)."""
    async with semaphore:
        payload = {**BASE_PAYLOAD, "page": page}
        headers = {**HEADERS}

        if xsrf_token:
            headers["X-XSRF-TOKEN"] = xsrf_token

        for attempt in range(retry_count):
            try:
                async with session.post(API_URL, json=payload, headers=headers) as response:
                    if response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt * 5
                        print(f"Page {page}: Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status != 200:
                        text = await response.text()
                        print(f"Page {page}: HTTP {response.status} - {text[:100]}")
                        if attempt < retry_count - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return [], False, 0

                    data = await response.json()
                    dataset_list = data.get("datasetList", {})
                    items = dataset_list.get("items", [])
                    has_more = data.get("hasMore", False)
                    total_results = dataset_list.get("totalResults", 0)

                    extracted = [extract_dataset_info(item) for item in items]
                    print(f"Page {page}: fetched {len(extracted)} datasets")

                    return extracted, has_more, total_results

            except asyncio.TimeoutError:
                print(f"Page {page}: Timeout (attempt {attempt + 1}/{retry_count})")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"Page {page}: Error - {e} (attempt {attempt + 1}/{retry_count})")
                await asyncio.sleep(2 ** attempt)

        return [], False, 0


async def scrape_all_datasets(
    max_concurrent: int = 5,
    max_pages: int | None = None,
) -> list[dict]:
    """Scrape all datasets with pagination."""
    all_datasets = []
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60)

    # Create cookie jar to persist cookies
    cookie_jar = aiohttp.CookieJar(unsafe=True)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        cookie_jar=cookie_jar
    ) as session:
        # Initialize session to get cookies
        print("Initializing session...")
        xsrf_token = await init_session(session)

        # First, fetch page 1 to get total count
        datasets, has_more, total_results = await fetch_page(session, 1, semaphore, xsrf_token)
        all_datasets.extend(datasets)

        if not datasets:
            print("Failed to fetch first page. Check if the API requires authentication.")
            return all_datasets

        if not has_more:
            return all_datasets

        # Calculate total pages (20 items per page)
        items_per_page = 20
        total_pages = (total_results + items_per_page - 1) // items_per_page

        if max_pages:
            total_pages = min(total_pages, max_pages)

        print(f"\nTotal datasets: {total_results}")
        print(f"Total pages to fetch: {total_pages}")
        print("-" * 50)

        # Fetch remaining pages in batches
        page = 2
        while page <= total_pages:
            batch_size = min(max_concurrent, total_pages - page + 1)
            tasks = [
                fetch_page(session, p, semaphore, xsrf_token)
                for p in range(page, page + batch_size)
            ]

            results = await asyncio.gather(*tasks)

            for datasets, _, _ in results:
                all_datasets.extend(datasets)

            page += batch_size

            # Progress update
            progress = len(all_datasets) / total_results * 100
            print(f"Progress: {len(all_datasets)}/{total_results} ({progress:.1f}%)")

            # Small delay between batches
            await asyncio.sleep(0.3)

    return all_datasets


def save_to_csv(datasets: list[dict], filename: str) -> None:
    """Save datasets to CSV file."""
    if not datasets:
        print("No datasets to save")
        return

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(datasets)

    print(f"\nSaved {len(datasets)} datasets to {filename}")


async def main():
    print("=" * 60)
    print("Kaggle Dataset Scraper")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()

    # Set max_pages=None to scrape ALL pages, or set a limit for testing
    # With 611,907 datasets at 20 per page = ~30,596 pages
    datasets = await scrape_all_datasets(
        max_concurrent=10,
        max_pages=None,  # Set to None to scrape ALL pages (~30,600 pages)
    )

    # Remove duplicates based on dataset_id
    seen_ids = set()
    unique_datasets = []
    for ds in datasets:
        ds_id = ds.get("dataset_id")
        if ds_id and ds_id not in seen_ids:
            seen_ids.add(ds_id)
            unique_datasets.append(ds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kaggle_datasets_{timestamp}.csv"
    save_to_csv(unique_datasets, filename)

    print(f"\nFinished at: {datetime.now()}")
    print(f"Total unique datasets scraped: {len(unique_datasets)}")


if __name__ == "__main__":
    # Fix for Windows asyncio event loop cleanup warning
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
