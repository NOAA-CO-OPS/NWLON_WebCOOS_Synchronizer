"""
Tests related to specifics of the WebCOOS API, which is being deprecated
"""
from tests.test_main import _get_key
from nwlon_webcoos_synchronizer.main import _save_frames, synch

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 1. Define your batch of (URL, local_filename) pairs
DOWNLOAD_BATCH = [
    ("https://example.com", "downloads/file1.pdf"),
    ("https://example.com", "downloads/file2.zip"),
    ("https://example.com", "downloads/image.png"),
]


def create_session(max_workers: int) -> requests.Session:
    """Creates a thread-safe requests Session with retries and connection pooling."""
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



def test_webcoos_downloads_serial():
    """
    Test serial vs. parallel download speed
    """
    token = _get_key()
    station = 8665530
    camera = 'Charleston Harbor, SC'
    source = synch(
        station=8665530,
        camera='Charleston Harbor, SC',
        data_product='water_level',
        camera_product='one-minute-stills',
        value='all',
        time_start='202401011000',
        time_end='202401011005',
        interval=1,
        cutoff=None,
        sep_model=None,
        token=token,
        save_dir='tmp'
    )
    datas_mov = _save_frames(source, camera, station, view_num=None)
    print(datas_mov)


def test_webcoos_downloads_parallel():
    """
    Test parallel download speed
    """
    run_batch_download(DOWNLOAD_BATCH, max_workers=8)


