"""
Tests related to specifics of the WebCOOS API, which is being deprecated
"""
from pathlib import Path
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
    """
    Downloads a single file using streaming to minimize RAM usage."""
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

        return (url, True, dest_path)
    except Exception as e:
        return (url, False, str(e))


def run_batch_download(
    manifest: list[tuple[str, str]], 
    max_workers: int = 4
) -> list[str]:
    """
    Runs a batch download of files specified in the manifest.
    """
    session = create_session(max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks to the thread pool
        futures = {
            executor.submit(download_file, session, url, path): url
            for url, path in manifest
        }
        # Process results as soon as each individual download finishes
        assets = []
        for future in as_completed(futures):
            url, success, filename = future.result()
            if success:
                assets.append(filename)
            status = "OK" if success else "FAILED"
            print(f"[{status}] {url} -> {filename}")
        return assets


class CameraFeed(API):
    """
    Extenson of the WebCOOS API interface to allow creating dry run manifests
    and doing batch downloads of images. The instance is locked to a single
    feed and can be reused for multiple interval downloads.

    Example usage

    ```    
    filelist = CameraFeed(
        token=token,
        camera='Charleston Harbor, SC',
        product='one-minute-stills',
        feed='raw-video-data'
    ).download_concurrent(
        start='202401011000',
        stop='202401011005',
        interval=1,
        max_workers=max_workers
    )
    ```
    """
    def __init__(
        self, 
        token: str,
        camera: str,
        feed: str = 'raw-video-data',
        product: str = 'one-minute-stills'
    ):
        super().__init__(token=token)
        self.camera = camera
        self.feed = feed
        self.product = product

    def get_asset_metadata(
        self,
        start: str,
        stop: str,
        interval: int,
    ) -> list[dict]:
        """
        The WebCOOS client build the manifest and does download in one step,
        so we need to provide a break point after checking inputs and before
        initiating the actual download so that we can provide the manifest to
        our async batch download.
        """
        _start = self._local2ISO(start, self.camera)
        _stop = self._local2ISO(stop, self.camera)
        feeds = self._get_camera_feeds(self.camera, self.assets_json)
        products = self._get_camera_products(self.feed, feeds, self.camera)
        service_slug, _ = self._get_service_slug(self.product, products, self.feed, self.camera, self.api_base_url, self.HEADERS)
        return self._get_elements(service_slug, _start, _stop, interval, self.api_base_url, self.HEADERS)

    @staticmethod
    def convert_assets_to_manifest(
        assets: list[dict],
        save_dir: str
    ) -> list[tuple[str, str]]:
        """
        Convert from WebCOOS metadata to pairs of source URL
        and IO destination. This assumes that all images are
        downloaded and save, to be later composited back into
        video.
        """
        manifest = []
        for element in assets:
            url = element['data']['properties']['url']
            filename = url.split('/')[-1]
            manifest.append((url, f"{save_dir}/{filename}"))
        return manifest

    def download_concurrent(
        self,
        start: str,
        stop: str,
        interval: int,
        max_workers: int = 8,
        save_dir: str = 'tmp'
    ) -> list[str]:
        """
        Performs a batch download of images based on the specified time range and interval.
        """
        filtered_elements = self.get_asset_metadata(
            start=start,
            stop=stop,
            interval=interval
        )
        manifest = self.convert_assets_to_manifest(filtered_elements, save_dir=save_dir)
        return run_batch_download(manifest, max_workers=max_workers)
