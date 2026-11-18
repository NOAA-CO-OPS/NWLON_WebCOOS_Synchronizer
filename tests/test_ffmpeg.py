import subprocess
from nwlon_webcoos_synchronizer import synch
from nwlon_webcoos_synchronizer.main import _save_frames, _produce_movie
from time import time
from tests.test_main import _get_key

def test_ffmpeg_make_movie_baseline():
    start = time()
    key = _get_key()
    camera = 'Charleston Harbor, SC'
    station = 8665530
    data = synch(
        station=station,
        camera=camera,
        data_product='water_level',
        camera_product='one-minute-stills',
        value='all',
        time_start='202401011000',
        time_end='202401011005',
        interval=1,
        cutoff=None,
        sep_model=None,
        token=key,
        save_dir='tmp'
    )
    print(f"Data download took {time() - start} seconds")
    start = time()
    datas_mov = _save_frames(data, camera=camera, station=station, view_num=None)
    print(f"Frame rendering took {time() - start} seconds")
    start = time()
    _ = _produce_movie(datas_mov, fps=1)
    print(f"Movie creation took {time() - start} seconds")

def test_ffmpeg_make_movie_ffmpeg():
    prefix = "nwlon_charleston-2024-01-01"
    start = time()
    frames = f'tmp/{prefix}*_frame.png'
    video_file = f'tmp/{prefix}_ffmpeg.mp4'
    subprocess.run([
        'ffmpeg', '-y','-framerate', '1', '-pattern_type', 'glob', '-i', frames,
        '-c:v', 'libx264', '-r', '30', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', video_file
    ])
    print(f"FFmpeg movie creation took {time() - start} seconds")
