## 
# FILENAME: CoopsApi.py
# CREATED:  2020/05/18
# 
# PURPOSE
# Class of data retrieval functions for the Center for Operational Oceanographic
# Products and Services (CO-OPS) APIs, including the Data API and Metadata API.
#
# This function will get data via the CO-OPS data API and return a Python Dictionary
# containing the downloaded data and metadata. More information about CO-OPS API for
# data retrieval can be found at https://tidesandcurrents.noaa.gov/api/.
#
# CO-OPS' data API call is limited to 31 days of 6-minute data so this function
# will send multiple successive requests for monthly time series data and then
# compile them all into a single output structure.
#
# PLEASE BE AWARE that sending too many API data requests at once can overload
# CO-OPS' data servers. To account for this, the time.sleep() function is implemented 
# to pause for 3 seconds between successive API requests. Users should also ensure
# that this function is implemented between API requests when augmenting this code
# for their own use.
#
# LICENSE
# Software code created by U.S. Government employees is not subject to 
# copyright in the United States (17 U.S.C.?105). The United States/
# Department of Commerce reserve all rights to seek and obtain copyright 
# protection in countries other than the United States for Software authored
# in its entirety by the Department of Commerce. To this end, the Department
# of Commerce hereby grants to Recipient a royalty-free, nonexclusive 
# license to use,copy, and create derivative works of the Software outside 
# of the United States.
#
# REVISION HISTORY
#
#   2018/02/05 - LL - Created
#
#   2020/05/18 - LA - Time.sleep() added & license & disclaimer info
#   
#   2020/07/17 - LA - Updated to Python 3 - replaced urllib to requests
#                     Retrieve data as json instead of csv
#   2020/08/05 - LA - Remove H and M codes for strftime
#                     API endpoint updated

    
import datetime as dt
import requests
import csv as csvy
import time as time
import pandas as pd
import numpy as np

# Class of data retrieval functions for CO-OPS APIs
class CoopsApi:

    def __init__(self):
        # URL information
        self.server = 'https://tidesandcurrents.noaa.gov/api/datagetter'
  
  
  
    # Get observations or product data for a given station, product type and date/time period.
    #
    # Input are as follows:
    # stationID - a string containing the digit water level station ID or the current meter designation (Required)
    # dateRange - a datetime list containing a starting and ending date for data retrieval
    #                     default is the previous 31 days from the time when the function is called
    # product   - the data product to be retrieved.  The options are:
    #   water_level             - Preliminary or verifed water levels, depending on availability
    #   air_temperature         - Air temperature as measured at the station
    #   water_temperature       - Water temperature as measured at the station
    #   wind                    - Wind speed, direction, and gusts as measured at the station
    #   air_pressure            - Barometric pressure as measured at the station
    #   air_gap                 - Air Gap (distance between a bridge and the water's surface) at the station
    #   conductivity            - The water's conductivity as measered at the station
    #   visibility              - visibility from the station's visibility sensor. A measure of atmopheric clarity
    #   humidity                - relative humidity as mesred at the station
    #   salinity                - Salinity and specific gravity data for the station
    #   hourly_height           - Verified hourly height water level data for the station
    #   high_low                - Verified high/low water level data for the station
    #   daily_mean              - Verified daily mean water level data for the station
    #   one_minute_water_level  - One minute water level data for the station
    #   predictions             - 6 minute predicted water level data for the station
    #   datums                  - datums data for the stations
    #   currents                - Currents data for the currents stations
    #   datum     - the datum bias for water level data.  Options are:
    #   MHHW    - Mean higher high water
    #   MHW     - Mean high water
    #   MSL     - Mean sea level
    #   MTL     - Mean tide level
    #   MLW     - Mean low water
    #   MLLW    - Mean lower low water
    #   NAVD    - North American Vertical Datum
    #   STND    - Station Datum (Default)
    #   IGLD    - International Great Lakes Datum
    #   CRD     - Columbia River Datum
    # units     - the units to return the data. Options are
    #   metric  - Metric (Celcius, meters) units
    #   english - English (farenheit, feet) units
    # timeZone  - The time zone to return the data.  Options are:
    #   gmt     - Greenwich mean time (default)
    #   lst     - local standard time.  Local to the station.
    #   lst_ldt - local standard/local daylight time. Local to the station.
    # interval  - the interval for which Meteorlogical data is returned. The defaults is 6 minute intervals
    #   and there is no need to specify it.  Only specify 'interval' for :
    #   h       - hourly Met data and predictions data
    #   hilo    - High/Low tide predictions for subordinate stations.
    # binNum    - the bin number for the spcified currenstation.  If not specified the returned data will correspond to the
    #   designated real-time bin. Example (bin='3')
    def get_data(self, stationID,
                dateRange = [dt.datetime.now() - dt.timedelta(days=31),
                             dt.datetime.now()],
                product   = 'water_level',
                datum     = 'STND',
                units     = 'metric',
                timeZone  = 'gmt',
                interval  = [],
                binNum    = []):


        ## Parse the dates and develop the starting and ending dates to call
        
        # define the time steps of the data
        if (product == 'one_minute_water_level'):
            step = dt.timedelta(days = 5)
        elif (product == 'hourly_height') or (product == 'high_low'):
            step = dt.timedelta(days = 365)
        elif (product == 'predictions') or (product == 'daily_mean') or (product == 'monthly_mean'):
            step = dt.timedelta(days = 10 * 365)
        else:
            step = dt.timedelta(days=31)

        # make the dictionary fields
        data = {'date_time':[],'value':[]}
        
        # create the startDate list.  This will be the information used in calling the
        # API recursively.  The end date of the request will be startDate + step - dt.timedelta(minutes=6)
        startDate = [dateRange[0]]

        while (startDate[-1] < dateRange[1]):
            startDate.append(startDate[-1] + step)

        if startDate[-1]>dateRange[1]:
            startDate.remove(startDate[-1])

        # The errorsFound variable will be returned as a list of errors, if any, found
        # during the calling of the URL for the data
        errorsFound = ['No errors were found']

        aDict =dict()

        # URL information
        server = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
        
        # call the API recursivly
        for d in startDate:
            # create endDate to call and check if it is greater than the requested end date
            d2 = d + step - dt.timedelta(minutes=6)
            if d2 > dateRange[1]:
                d2 = dateRange[1]

            urlRequest = ( server   +
                        '?begin_date='  + d.strftime('%Y%m%d')    +
                        '&end_date='    + d2.strftime('%Y%m%d')   +
                        '&station='     + stationID                     +
                        '&product='     + product                       +
                        '&datum='       + datum                         +
                        '&units='       + units                         +
                        '&time_zone='   + timeZone                      +
                        '&application=OD_python&format=json' )
            
            print("Querying the data API via " + urlRequest)
            
            # Append additional information to the url request if needed.
            if (product=='currents') and (len(binNum)):    
                urlRequest = ( urlRequest  + '&bin=' + binNum)
                
            elif len(interval) > 0:
                urlRequest = ( urlRequest + '&interval=' + interval )

            # Read the url response
            urlResponse = requests.get(urlRequest)
            content = urlResponse.json()
            
            # check for errors
            if 'error' in content:
                print(content['error'],['message'])
            else:
                # no errors found
                
                # populate the output dictionary with date/time and values
                k = content.keys()
                for aDict in content[list(k)[-1]]:
                    try:
                        date_time = pd.to_datetime(aDict['t'])
                    except ValueError:
                        date_time = pandas.NaT
                    try:
                        value = float(aDict['v'])
                    except ValueError:
                        value = np.NaN
                    data['date_time'].append(date_time)
                    data['value'].append(value)

            #change lists to numpy arrays
            output = pd.DataFrame(data)
                        
            # Pause for 3 seconds between API requests. Do not modify this.
            time.sleep(3)
            
        # Done cycling through 31 day intervals for requested date/time period
        
        # If errors were found, remove the 'no errors were found' part of the message
        if len(errorsFound)>1:
            errorsFound.remove(errorsFound[0])

 #       # If the request product is datum information, then clean up the output dictionary
 #       if product == 'datums':
 #           temp = dict()
 #           dCount = 0
 #           for d in aDict['Datum']:
 #               temp[d] = aDict['Value'][dCount]
 #               dCount += 1
 #           aDict = temp

            
        return output, errorsFound
        
