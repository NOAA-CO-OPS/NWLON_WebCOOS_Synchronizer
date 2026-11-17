import os
from nwlon_webcoos_synchronizer.main import make_movie
import requests
import json
import pytest
import warnings
from pandas import DataFrame
from pywebcoos import API
import nwlon_webcoos_synchronizer as synch
import math


def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees: (lat, lon)).
    Returns the distance in kilometers.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Radius of the Earth in kilometers (mean radius)
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Apply Haversine formula
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

warnings.filterwarnings('ignore')


# Input checking / error handling tests #
def test_invalid_station_raises_exception():
    key = _get_key()
    with pytest.raises(ValueError, match="Requested station ID is not a valid NWLON statation."):
        synch.synch(station=100, camera='Charleston Harbor, SC', data_product='water_level', camera_product='one-minute-stills', value='all',
                    time_start='202401011000', time_end='202401011010', interval=6, cutoff=None,
                    sep_model=None, token=key, save_dir='.')

        
def test_invalid_data_product_raises_exception():
    key = _get_key()
    with pytest.raises(ValueError, match="Requested NWLON data product is not available."):
        synch.synch(station=8665530, camera='Charleston Harbor, SC', data_product='Not a product', camera_product='one-minute-stills', value='all',
                    time_start='202401011000', time_end='202401011010', interval=6, cutoff=None,
                    sep_model=None, token=key, save_dir='.')

        
def test_invalid_date_format_raises_exception():
    key = _get_key()
    with pytest.raises(ValueError, match="Requested start date is of improper format. Format should be yyyymmddHHMM."):
        synch.synch(station=8665530, camera='Charleston Harbor, SC', data_product='water_level', camera_product='one-minute-stills', value='all',
                    time_start='01/01/2024 1000', time_end='202401011010', interval=6, cutoff=None,
                    sep_model=None, token=key, save_dir='.')    
    with pytest.raises(ValueError, match="Requested end date is of improper format. Format should be yyyymmddHHMM."):
        synch.synch(station=8665530, camera='Charleston Harbor, SC', data_product='water_level', camera_product='one-minute-stills', value='all',
                    time_start='202401011000', time_end='2024-01-01 10:10', interval=6, cutoff=None,
                    sep_model=None, token=key, save_dir='.')      

        
def test_date_out_of_range_raises_exception():
    key = _get_key()
    with pytest.raises(ValueError, match="At least one requested date bound is outside the range of available data for this product at this camera."):
        synch.synch(station=8665530, camera='Charleston Harbor, SC', data_product='water_level', camera_product='one-minute-stills', value='all',
                    time_start='190001011000', time_end='190001011010', interval=6, cutoff=None,
                    sep_model=None, token=key, save_dir='.')  

        
def test_invalid_value_argument_raises_exception():
    key = _get_key()
    with pytest.raises(ValueError, match="value argument must be either 'all' or a float/integer value."):
        synch.synch(station=8665530, camera='Charleston Harbor, SC', data_product='water_level', camera_product='one-minute-stills', value='Invalid input',
                    time_start='202401011000', time_end='202401011010', interval=6, cutoff=None,
                    sep_model=None, token=key, save_dir='.')   
    with pytest.raises(ValueError, match="value argument must be either 'all' or a float/integer value."):
        synch.synch(station=8665530, camera='Charleston Harbor, SC', data_product='water_level', camera_product='one-minute-stills', value=None,
                    time_start='202401011000', time_end='202401011010', interval=6, cutoff=None,
                    sep_model=None, token=key, save_dir='.')           


def test_invalid_local_file_extension_raises_synch_local_exception():
    with open('data/images/202401011200.txt', 'w'):
        pass
    with pytest.raises(ValueError, match="At least one file in the image directory does not have a png, jpg, or tif extension."):
        synch.synch_local(station=8774230, camera='Test camera', data_product='water_level', local_dir='data/images', 
                          value='all', time_start='202406201130', time_end='202406201200', interval=6, cutoff=None)
    os.remove('data/images/202401011200.txt')
    
    
def test_invalid_local_file_name_raises_synch_local_exception():
    with open('data/images/BadNameFormat.jpg', 'w'):
        pass
    with pytest.raises(ValueError, match="At least one file in the image directory is not named with the format yyyymmddHHMM.png/jpg/tif."):
        synch.synch_local(station=8774230, camera='Test camera', data_product='water_level', local_dir='data/images', 
                          value='all', time_start='202406201130', time_end='202406201200', interval=6, cutoff=None)
    os.remove('data/images/BadNameFormat.jpg')


# Unit tests #
def test_get_cameras():
    key = _get_key()
    cams = synch.get_cameras(key)
    assert len(cams) > 0 , 'Getting camera list failed'

    
def test_get_products():
    key = _get_key()
    prods = synch.get_products('Charleston Harbor, SC', key)
    assert len(prods) > 0 , 'Getting product list failed'    

    
def test_get_inventory():
    key = _get_key()
    inv = synch.get_inventory('Charleston Harbor, SC', 'video-archive', key)
    assert len(inv) > 0 , 'Getting product inventory failed' 


def test_synch_value_equals_all():
    key = _get_key()
    synchro = synch.synch(station=8665530,           
                          camera='Charleston Harbor, SC',  
                          data_product='water_level', 
                          camera_product='one-minute-stills',     
                          value='all',             
                          time_start='202401011000',   
                          time_end='202401011005',         
                          interval=1,  
                          cutoff=None,               
                          sep_model=None,        
                          token=key,             
                          save_dir='.')
    os.remove('nwlon_charleston-2024-01-01-150053Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150152Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150253Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150352Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150453Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150553Z.jpg')
    assert len(synchro['value'].dropna()) == len(synchro['image'].dropna()) , "Data synchronization failed - missing data or image"


def test_synch_value_equals_float():
    key = _get_key()
    synchro = synch.synch(station=8665530,           
                          camera='Charleston Harbor, SC',  
                          data_product='water_level', 
                          camera_product='one-minute-stills',     
                          value=-1.47,             
                          time_start='202401010500',   
                          time_end='202401010530',         
                          interval=6,  
                          cutoff=10,               
                          sep_model=None,        
                          token=key,             
                          save_dir='.')
    os.remove('nwlon_charleston-2024-01-01-100643Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-101242Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-102442Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-101843Z.jpg')
    assert len(synchro['value'].dropna()) == len(synchro['image'].dropna()) , "Data synchronization failed - missing data or image"


def test_synch_local():
    synchro = synch.synch_local(station=8774230, 
                                camera='Test camera', 
                                data_product='water_level', 
                                local_dir='data/images', 
                                value='all', 
                                time_start='202406201130', 
                                time_end='202406201200', 
                                interval=6, 
                                cutoff=None)
    assert len(synchro.iloc[1]['image']) > 0 , 'Local image synchronization failed.'


def test_make_movie():
    key = _get_key()
    synchro = synch.synch(station=8665530,           
                          camera='Charleston Harbor, SC',  
                          data_product='water_level', 
                          camera_product='one-minute-stills',     
                          value='all',             
                          time_start='202401011000',   
                          time_end='202401011005',         
                          interval=1,  
                          cutoff=None,               
                          sep_model=None,        
                          token=key,             
                          save_dir='.')
    synch.make_movie(synchro, camera='Charleston Harbor, SC', station=8665530)
    os.remove('nwlon_charleston-2024-01-01-150053Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150152Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150253Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150352Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150453Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150553Z.jpg')
    os.remove('nwlon_charleston-2024-01-01-150053Z.jpg_frame.png')
    os.remove('nwlon_charleston-2024-01-01-150152Z.jpg_frame.png')
    os.remove('nwlon_charleston-2024-01-01-150253Z.jpg_frame.png')
    os.remove('nwlon_charleston-2024-01-01-150352Z.jpg_frame.png')
    os.remove('nwlon_charleston-2024-01-01-150453Z.jpg_frame.png')
    os.remove('nwlon_charleston-2024-01-01-150553Z.jpg_frame.png')  
    assert os.path.exists('nwlon_charleston-2024-01-01-150053Z--nwlon_charleston-2024-01-01-150553Z--Video.mp4'), 'Movie creation failed - movie not created'
    os.remove('nwlon_charleston-2024-01-01-150053Z--nwlon_charleston-2024-01-01-150553Z--Video.mp4')


def test_build_virtual_station_camera_search():
    """
    Given a station """
    # station = 8418150  # Portland, ME
    my_lat = 44
    my_lon = -70
    api_token = _get_key()
    url = "https://app.webcoos.org/webcoos/api/v1/assets/"
    headers = {
        "Authorization": f"Token {api_token}",
    }
    response = requests.get(url, headers=headers, timeout=30)
    data = response.json()
    results = data.get('results', [])
    station_info = []
    for each in results:
        name = each['package']['data']['common']['label']
        lon, lat = each['package']['data']['properties']['location']['coordinates']
        dxy = haversine((lat, lon), (my_lat, my_lon))
        station_info.append({
            'uuid': each['uuid'],
            'name': name,
            'lon': lon,
            'lat': lat,
            'distance': dxy
        })
    df = DataFrame(station_info).sort_values(by='distance')
    # _next = data.get('next', None)
    print(df)

def test_build_virtual_station_station_search():
    pass

def test_build_virtual_station_end_to_end():
    """
    Confirm that geo-lookup matching based on station coordinates works correctly.

    - find stations near camera (N>0)
    - find cameras near station (N=1)
    """
    token = _get_key()  # Insert your WebCOOS API token here #
    station = 8418150  # Portland, ME
    camera = 'Camp Ellis Beach at Saco, ME - South view'
    print("Starting data sync...")
    synchro = synch.synch(
        station=station,
        camera=camera,
        data_product='water_level',
        camera_product='one-minute-stills',
        value='all',
        time_start='202512170800',
        time_end='202512171000',
        interval=6,
        cutoff=None,
        sep_model=None,
        token=token,
        save_dir='tmp'
    )
    print("Completed image sync, starting to render movie...")
    _ = synch.make_movie(synchro,camera,station)

def _get_key():
    key = os.getenv('API_KEY')
    if key[0] == "'":
        key = key[1:-1]
    return key
