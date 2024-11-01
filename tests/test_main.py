import os
import pytest
import nwlon_webcoos_synchronizer as synch

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

    
def _get_key():
    key = os.getenv('API_KEY')
    if key[0] == "'":
        key = key[1:-1]
    return key