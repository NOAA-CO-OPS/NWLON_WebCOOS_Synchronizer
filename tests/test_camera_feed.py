"""
Tests related to specifics of the WebCOOS API, which is being deprecated
"""
import os
from glob import glob

import pytest
from tests.test_main import _get_key
from nwlon_webcoos_synchronizer.camera_feed import CameraFeed
from time import time


@pytest.mark.parametrize("max_workers", [4])
def test_camera_feed_download_concurrent(max_workers: int):
    """
    Perform the same download process in serial and parallel to compare performance.
    """
    _start = time()
    token = _get_key()
    api = CameraFeed(
        token=token,
        camera='Charleston Harbor, SC',
        product='one-minute-stills',
        feed='raw-video-data'
    )
    print(f"Preprocessing took {time() - _start:.2f} seconds")
    _ = api.download_concurrent(
        start='202401011000',
        stop='202401011005',
        interval=1,
        max_workers=max_workers
    )
    print(f"Parallel download took {time() - _start:.2f} seconds")
    for file in glob("tmp/*.jpg"):
        os.remove(file)
