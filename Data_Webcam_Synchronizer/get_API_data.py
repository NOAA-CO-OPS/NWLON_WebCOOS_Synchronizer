import datetime
import numpy as np
import pandas as pd
import requests

class get_API_data:
    '''
    Class to download data from CO-OPS API. Class allows request of 
    multiple datasets over the same time period.

    Parameters
    ----------
    station: INT
        The NWLONS station ID
    begin_date: STR
        The begin date for data retrieval, in the format 'yyyymmdd' (e.g. '20200101')
    end_date: STR
        The end date for data retrieval, in the format 'yyyymmdd' (e.g. '20231231')
    product: STR or list of STR
        The water level, met, or oceanographic product of interst.
        The choices are:
        'water_level' - Prelim or verified water levels
        'air_temperature' - Air temp as measured
        'water_temperature' - Water temp as measured
        'wind' - Wind Speed, direction, and gusts as measured
        'air_pressure' - Barometric pressure as measured
        'air_gap' - Air Gap at the station
        'conductivity' - water conductivity
        'visibility' - Visibility
        'humidity' - relative humidity
        'hourly_height' - Verified hourly height data
        'high_low' - verified high/low water level data
        'daily_mean' - verified daily mean water level data
        'monthly_mean' - Verified monthly mean water level data
        'one_minute_water_level' - One minute water level data
        'predictions' - 6 minute predicted water level data
        'datums' - accepted datums for the station
        'currents' - Current data for thee current station
        DEFAULT = 'water_level'
    units : STR
        The type of units to use. Either 'english' or 'metric'
        DEFAULT = 'metric'
    datum_bias: STR
        The datum to which to bias the data to. Options are 
        'MHHW' - mean higher high water
        'MHW' - mean high water
        'MTL' - mean tide level
        'MSL' - mean sea level
        'MLW' - mean low water
        'MLLW' - mean lower low water
        'NAVD' - North American Veritcal Datum of 1988
        'STND' - station datum
        DEFAULT = 'MHHW'
    time_zone: STR
        The time zone for the data. Options are:
        'gmt' - Greenwich Mean Time
        'lst' - local standard time
        'lst_ldt' - Local Standard or Local daylight, depending on time of year
         DEFAULT = 'gmt'

    Returns
    -------
    None.

    '''
    def __init__(self,station,begin_date,end_date,product='water_level',units='metric',
                     datum_bias='MHHW',time_zone='gmt'):
        self.station = station
        self.begin_date = begin_date
        self.end_date = end_date
        self.product = product
        self.units = units
        self.datum_bias = datum_bias
        self.time_zone = time_zone
               
    def download_and_format(self):
        if not isinstance(self.product, list):
            self.product = [self.product]
            
        data_all = {}
        for prod in self.product:         
            # If requesting an hourly or 6-minute product, there is a 30 day max interval for retrieval. 
            # So need to loop through each month of the interval to download and then
            # smoosh it all together #
            cond1 = (prod=='water_level' or prod=='hourly_height' or prod=='predictions')
            cond2 = utils.datestr2dt(self.end_date)-utils.datestr2dt(self.begin_date)>datetime.timedelta(days=30)
            if cond1 and cond2:
                # Get the start and end datetimes #
                begin_dt = utils.datestr2dt(self.begin_date)
                end_dt = utils.datestr2dt(self.end_date)
                # Force the end datetime to be at the end of the requested day #
                if prod=='water_level':
                    end_dt = datetime.datetime(end_dt.year,end_dt.month,end_dt.day,23,54)
                else:
                    end_dt = datetime.datetime(end_dt.year,end_dt.month,end_dt.day,23,0)
                # Generate a list of interval datetimes between start and end #
                datetimes_list = []
                current_datetime = begin_dt
                while current_datetime <= end_dt:
                    datetimes_list.append(current_datetime)
                    current_datetime += datetime.timedelta(days=30)
                datetimes_list.append(end_dt)
                # Download data for each datetime interval and put into a DataFrame #
                for i in range(len(datetimes_list)-1):                 
                    begin_dt_interval = datetimes_list[i]
                    end_dt_interval = datetimes_list[i+1]
                    url = self.build_url_dapi(str(self.station),utils.dt2datestr(begin_dt_interval),
                                    utils.dt2datestr(end_dt_interval),product=prod,
                                    units=self.units,datum_bias=self.datum_bias,
                                    time_zone=self.time_zone)
                    content = self.request_data(url)
                    data1 = self.format_content_dapi(content)
                    if i==0:
                        data = data1
                    else:
                        data = pd.concat([data,data1],ignore_index=True)
                data = data.drop_duplicates(subset='time', keep='first')
                data = self.fill_gaps(data,begin_dt,end_dt)              
            else:
                if prod in ['datums','supersededdatums','harcon','sensors','details',
                                   'notices','disclaimers','benchmarks','tidepredoffsets',
                                   'floodlevels']:
                     url = self.build_url_mdapi(str(self.station),
                                                None,None,product=prod,units=self.units)
                     content = self.request_data(url)
                     data = self.format_content_mdapi(content)
                elif prod in ['sealvltrends']:
                     url = self.build_url_dpapi(str(self.station),
                                                None,None,product=prod)
                     content = self.request_data(url)
                     data = self.format_content_dpapi(content)
                else:
                    url = self.build_url_dapi(str(self.station),self.begin_date,
                                    self.end_date,product=prod,
                                    units=self.units,datum_bias=self.datum_bias,
                                    time_zone=self.time_zone)
                    content = self.request_data(url)
                    data = self.format_content_dapi(content)
                    data = self.fill_gaps(data,utils.datestr2dt(self.begin_date),utils.datestr2dt(self.end_date))
                     
            data_all[prod] = data
            
        return data_all
                               
    def run(self):
        data = self.download_and_format()
        return data


    @staticmethod
    def build_url_dapi(station,begin_date,end_date,product='water_level',units='metric',
                     datum_bias='MHHW',time_zone='gmt'):
        
        # CO-OPS API server #
        server = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?'
        
        if product=='predictions':
            url = (server + 'begin_date=' + begin_date +'&end_date=' + end_date +'&station=' + str(station) +
                 '&product=' + product +'&datum=' + datum_bias + '&time_zone=' + time_zone + '&units=' + 
                 units + '&format=json' +'&interval=h')
        else:
            url = (server + 'begin_date=' + begin_date +'&end_date=' + end_date +'&station=' + str(station) +
                 '&product=' + product +'&datum=' + datum_bias + '&time_zone=' + time_zone + '&units=' + 
                 units + '&format=json')
        
        return url

    @staticmethod
    def build_url_mdapi(station,begin_date=None,end_date=None,product='details',units='metric'):
        
        # CO-OPS metadata API server #
        server = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/'
        
        url = (server + str(station) + '/'+product+'.json?units='+units)
    
        return url  
    
    @staticmethod
    def build_url_dpapi(station,begin_date=None,end_date=None,product='sealvltrends'):
        
        # CO-OPS metadata API server #
        server = 'https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/product/'
        
        url = (server + product + ".json?station=" + str(station) + "&affil=us")
    
        return url   
    
    @staticmethod
    def request_data(url):
        content = requests.get(url).json()
        return content
    
    @staticmethod 
    def format_content_dapi(content):
        if len(content)>1 or list(content.keys())[0]=='predictions': # len=1 indicates no data was found (just an eror message returned) #
            data_raw = content[list(content.keys())[-1]]
            data1 = []
            for val in ['t','v']:
                data1.append([data_raw[i][val] for i in range(len(data_raw))])
            data_arr = np.array(data1).T
            data_dict = {'time':[utils.datestr2dt(data_arr[i,0]) for i in range(len(data_arr))],
                    'val':[float(data_arr[i,1]) if len(data_arr[i,1])>0 else np.nan for i in range(len(data_arr))] }
        else:
            data_dict = {'time':[],
                    'val':[]}
                   
        return pd.DataFrame(data_dict)
    
    @staticmethod 
    def format_content_mdapi(content):
        data_dict = {}
        for key in list(content.keys()):
            data_dict[key] = content[key]
        return data_dict

    @staticmethod 
    def format_content_dpapi(content):
        data_dict = content[list(content.keys())[-1]]
        return data_dict
    
    @staticmethod
    def fill_gaps(data,begin_date,end_date):
        def create_fillseries(tstart,tend,dt):
            fill = pd.date_range(start=tstart, end=tend, freq=dt)
            fill2 = pd.DataFrame({'time':fill,'val':np.empty(len(fill))*np.nan})
            return fill2          
        data = data.reset_index(drop=True)
        # Get the data time interval #
        dt = data['time'][1]-data['time'][0] # This assumes there is no missing data between the first two entries #
        # Find gaps as those where the time between two values is > dt
        tdif = [data['time'][i+1]-data['time'][i] for i in range(len(data['time'])-1)]
        jumps = np.where(np.array(tdif)!=dt)[0]
        # For each gap, create a dummy vector to fill the gap at dt spacing.
        # Save all of these to insert later #
        fill_all = []
        for jump in jumps:
            t1 = data['time'][jump]
            t2 = data['time'][jump+1]
            fill = create_fillseries(t1+dt,t2-dt,dt)
            fill_all.append(fill)
        # Insert the dummy fill dfs into the data df. Do this working last-to-first
        # so we don't have to worry about re-indexing after inserting #
        for i in np.arange(len(fill_all)-1,-1,-1):
            insert_index = jumps[i]+1
            data_before = data.iloc[:insert_index]
            data_after = data.iloc[insert_index:]
            data = pd.concat([data_before, fill_all[i], data_after], ignore_index=True)
        # Now also need to make sure the start and end points requested have data,
        # if not need to fill to there #
        tf_start = data['time'].iloc[0]==begin_date
        tf_end = data['time'].iloc[-1]==end_date
        if not tf_start:
            fill = create_fillseries(begin_date,data['time'].iloc[0]-dt,dt)
            data = pd.concat([fill,data], ignore_index=True)
        if not tf_end:
            fill = create_fillseries(data['time'].iloc[-1]+dt,end_date, dt)
            data = pd.concat([data,fill], ignore_index=True)     
            
        return data

    
class utils:
    @staticmethod
    def datestr2dt(datestr):
        if len(datestr)==8:
            return datetime.datetime(int(datestr[0:4]),int(datestr[4:6]),int(datestr[6:8]))
        elif len(datestr)==16:
            return datetime.datetime(int(datestr[0:4]),int(datestr[5:7]),int(datestr[8:10]),int(datestr[11:13]),int(datestr[14:16]))
        
    @staticmethod
    def dt2datestr(dt):
        mo = dt.month
        day = dt.day
        if mo<10:
            smo = '0'
        else:
            smo = ''
        if day<10:
            sday = '0'
        else:
            sday = ''
        return str(dt.year)+smo+str(dt.month)+sday+str(dt.day)   
    
    @staticmethod
    def dt2decimalyr(dt):
        yr_ref = datetime.datetime(dt.year,1,1)
        time_delta = dt-yr_ref
        frac1 = time_delta.days/365
        frac2 = time_delta.seconds/(60*60*24*365)
        frac = frac1+frac2
        decimalyr = dt.year+frac
        return decimalyr
