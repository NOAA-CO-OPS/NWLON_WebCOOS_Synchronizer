import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../NWLON_WebCOOS_Synchronizer')))
import nwlon_webcoos_synchronizer as nws

token = ''  # Insert your WebCOOS API token here #
station = 9411340
camera = 'Carpinteria State Beach, CA'
erddap_dataset_id = 'ism-cencoos-edu_ucsd_cdip_flood_-3' # 'ONSET_gloucester_hourly_water_level'
dataset_id = erddap_dataset_id
synchro, metadata = nws.synch(dataset_id=dataset_id,        # If int --> the NWLON station ID from which to use data
                                                         # If str --> dataset id for ERDDAP data source to look for (e.g. ONSET_gloucester_hourly_water_level)
                    camera=camera,                       # The WebCOOS camera name #
                    data_product='water_surface_above_mllw_ipa_maximum_water_elevation_due_to_waves_and_tides' ,         # The ERDAPP/NWLON data product to make a timeseries of #
                    camera_product='one-minute-stills',  # The WebCOOS image product to use for the movie #
                    value='all',                         # Can be 'all' or a float value, depending on what you want.
                    time_start='202610270700',           # Start of the movie in local time at the camera location #
                    time_end='202610271500',             # End of the movie in local time at the camera location
                    interval=6,                          # Interval of data and imagery, in minutes. #
                    cutoff=None,                         # Make the movie of only this many data points. Use with value to get what you want.  #
                    sep_model=None,                      # A trained view separation model. Can be made yourself. Not needed for this non-PTZ camera.
                    token=token,                         # Your WebCOOS API Token
                    save_dir='media',                   # The directory in which to save the images and movie
                    )     

mov = nws.make_movie(synchro, metadata, camera,dataset_id)                   
