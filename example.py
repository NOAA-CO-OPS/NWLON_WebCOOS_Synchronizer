import nwlon_webcoos_synchronizer as nws
token = ''  # Insert your WebCOOS API token here #
station = 8665530
camera = 'North Inlet-Winyah Bay  National Estuarine Research Reserve Dock, Georgetown, SC'
erddap_link = "https://oceanwatch.pfeg.noaa.gov/noaa_wide/erddap/"
erddap_dataset_id = 'ism-aoos-kipnuk-alaska-water-lev' # 'ONSET_gloucester_hourly_water_level'

synchro = nws.synch(station=0,                           # The NWLON station ID (0 if using erdap dataset, ID if using NWLON station)
                    camera=camera,                       # The WebCOOS camera name #
                    # data_product='water_level',        # The CO-OPS data product to make a timeseries of #
                    data_product='water_surface_above_mhhw' ,         # The ERDAPP data product to make a timeseries of 
                    camera_product='one-minute-stills',  # The WebCOOS image product to use for the movie #
                    value='all',                         # Can be 'all' or a float value, depending on what you want.
                    time_start='202610170800',           # Start of the movie in local time at the camera location #
                    time_end='202610171000',             # End of the movie in local time at the camera location
                    interval=6,                          # Interval of data and imagery, in minutes. #
                    cutoff=None,                         # Make the movie of only this many data points. Use with value to get what you want.  #
                    sep_model=None,                      # A trained view separation model. Can be made yourself. Not needed for this non-PTZ camera.
                    token=token,                         # Your WebCOOS API Token
                    save_dir='data\\',                   # The directory in which to save the images and movie
                    erddap_link = erddap_link,           # URL to ERDDAP page (e.g. https://data.neracoos.org/erddap/)
                    erddap_dataset_id = erddap_dataset_id,  # dataset id for specific data source to look for (e.g. ONSET_gloucester_hourly_water_level)
                    )     

mov = nws.make_movie(synchro,camera,station)                   