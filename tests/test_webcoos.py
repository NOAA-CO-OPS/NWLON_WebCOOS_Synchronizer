"""
Tests related to specifics of the WebCOOS API, which is being deprecated
"""
import os
from glob import glob

import pytest
from tests.test_main import _get_key
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pywebcoos import API
from time import time



def create_session(max_workers: int) -> requests.Session:
    """
    Creates a thread-safe requests Session with retries and connection pooling.
    """
    session = requests.Session()

    # Configure automatic retries for transient server errors
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    # Match pool size to max_workers so threads don't block waiting for connections
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=max_workers,
        pool_maxsize=max_workers,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_file(
    session: requests.Session, url: str, dest_path: str
) -> tuple[str, bool, str]:
    """Downloads a single file using streaming to minimize RAM usage."""
    try:
        path = Path(dest_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # stream=True prevents loading the entire file into memory at once
        with session.get(url, stream=True, timeout=(10, 30)) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return (url, True, f"Saved to {dest_path}")
    except Exception as e:
        return (url, False, str(e))


def run_batch_download(batch: list[tuple[str, str]], max_workers: int = 8):
    session = create_session(max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks to the thread pool
        futures = {
            executor.submit(download_file, session, url, path): url
            for url, path in batch
        }

        # Process results as soon as each individual download finishes
        for future in as_completed(futures):
            url, success, message = future.result()
            status = "OK" if success else "FAILED"
            print(f"[{status}] {url} -> {message}")


def get_download_manifest(
    api: API,
    camera_name: str,
    product_name: str,
    start: str,
    stop: str,
    interval: int,
    feed_name: str = 'raw-video-data'
) -> list[dict]:
    """
    The WebCOOS client build the manifest and does download in one step,
    so we need to provide a break point after checking inputs and before
    initiating the actual download so that we can provide the manifest to
    our async batch download.
    """
    _start = api._local2ISO(start, camera_name)
    _stop = api._local2ISO(stop, camera_name)
    feeds = api._get_camera_feeds(camera_name, api.assets_json)
    products = api._get_camera_products(feed_name, feeds, camera_name)
    service_slug, _ = api._get_service_slug(product_name, products, feed_name, camera_name, api.api_base_url, api.HEADERS)
    return api._get_elements(service_slug, _start, _stop, interval, api.api_base_url, api.HEADERS)


@pytest.mark.parametrize("max_workers", [4, 8])
def test_webcoos_api_download_parallel(max_workers: int):
    """
    Perform the same download process in serial and parallel to compare performance.
    """
    _start = time()
    token = _get_key()
    api = API(token=token)
    filtered_elements = get_download_manifest(
        api=api,
        camera_name='Charleston Harbor, SC',
        product_name='one-minute-stills',
        start='202401011000',
        stop='202401011005',
        interval=1,
        feed_name='raw-video-data'
    )
    urls = [element['data']['properties']['url'] for element in filtered_elements]
    manifest = [(url, f"tmp/{url.split('/')[-1]}") for url in urls]
    print(f"Preprocessing took {time() - _start:.2f} seconds")
    _start = time()
    # doesn't write images if they already exist, but does download them anyway
    _ = api._download_imagery(filtered_elements, save_dir="tmp")
    for file in glob("tmp/*.jpg"):
        os.remove(file)
    serial = time() - _start
    print(f"Serial download took {serial:.2f} seconds")
    _start = time()
    run_batch_download(manifest, max_workers=max_workers)
    parallel = time() - _start
    print(f"Parallel download took {parallel:.2f} seconds")
    for file in glob("tmp/*.jpg"):
        os.remove(file)
    print(f"Files were cleaned up between downloads.")
    assert parallel < serial, "Parallel download should be faster than serial download"
