# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:32:29 2024

@author: Matthew.Conlin
"""

import datetime
import numpy as np
import os
import pandas as pd
import requests
import pytz

class WebcoosApi():

    def __init__(self,token):
        '''
        Class to interface with the WebCOOS API, written by Greg Dusek and Matt Conlin.

        Example usage:
        webcoos = WebcoosApi(token) # Must register on website to get a toeken
        webcoos.cameras 
        webcoos.download('Charleston Harbor, SC',201901011200 201901011300)
        '''
        # Establish the base URL and headers for requests #
        self.api_base_url = 'https://app.webcoos.org/webcoos/api/v1'
        self.HEADERS = {
            'Authorization': 'Token '+token,
            'Accept': 'application/json'
        }      
        #Access the json assets via the webcoos API and get the camera list
        self.assets_json = self._make_api_request(self.api_base_url,self.HEADERS)
        df_cams = self._get_camera_list(self.assets_json)
        self.cameras = df_cams
    
    def view_products(self,camera_name):
        '''
        Function to view the available products at a camera.
        '''
        feeds = self._get_camera_feeds(camera_name,self.assets_json)
        feed_name = 'raw-video-data'
        products = self._get_camera_products(feed_name,feeds,camera_name)
        print(f"Products for camera '{camera_name}' and feed '{feed_name}':")
        for product in products:
            print(product['data']['common']['label'])  # Printing the product label       
    
    def view_data(self,camera_name,product_name):
        '''
        Function to view available data for a product at a camera.
        '''
        feeds = self._get_camera_feeds(camera_name,self.assets_json)
        feed_name = 'raw-video-data'
        products = self._get_camera_products(feed_name,feeds,camera_name)
        service_slug,df_inv = self._get_service_slug(product_name,products,feed_name,camera_name,self.api_base_url,self.HEADERS)
        # How many days of data are there?
        total_days_with_data = df_inv['Has Data?'].sum()
        print(f"Total days of data: {total_days_with_data}")
        # What is the range of dates with data?
        min_date = df_inv['Bin Start'].min()
        max_date = df_inv['Bin End'].max()
        print(f"Data range: {min_date} to {max_date}")
        
    def download(self,camera_name,product_name,start,stop,interval,save_dir):
        start = self._local2ISO(start,camera_name)
        stop = self._local2ISO(stop,camera_name)
        feeds = self._get_camera_feeds(camera_name,self.assets_json)
        feed_name = 'raw-video-data'
        products = self._get_camera_products(feed_name,feeds,camera_name)
        service_slug,df_inv = self._get_service_slug(product_name,products,feed_name,camera_name,self.api_base_url,self.HEADERS)
        filtered_elements = self._get_elements(service_slug,start,stop,interval,self.api_base_url,self.HEADERS)
        filenames = self._download_imagery(filtered_elements,save_dir)
        return filenames
    
    def _make_api_request(self,api_base_url,HEADERS):
        '''
        Function to query the webcoos API for available assets
        '''
        response = requests.get(f"{api_base_url}/assets/", headers=HEADERS)

        # Check the status code of the response
        if response.status_code == 200:
            #Return the assets json
            return response.json()
        else:
            # Print error information if the request was not successful
            print(f"Failed to retrieve assets: {response.status_code} {response.text}")
            return None

      
    def _get_camera_list(self,assets_json):
        '''
        Function to get the camera list
        '''
        # Collecting camera names
        camera_names = [asset['data']['common']['label'] for asset in assets_json['results']]

        # Create pandas dataframe
        df_cams = pd.DataFrame(camera_names, columns=['Camera Name'])

        # Return the dataframe
        return df_cams
 
    
    def _get_camera_feeds(self,camera_name,assets_json):
        '''
        Function to get the camera feeds for a specific camera
        '''
        for asset in assets_json['results']:
            if asset['data']['common']['label'] == camera_name:
                # print(f"Feeds for camera '{camera_name}':")
                feeds = asset['feeds']
                for feed in feeds:
                    pass
                    # print(feed['data']['common']['label'])  # Printing the feed label
                break
        return feeds


    def _get_camera_products(self,feed_name,feeds,camera_name):
        '''
        Function to get the camera products available for that camera and feed
        '''
        for feed in feeds:
            if feed['data']['common']['label'] == feed_name:
                products = feed['products']
        return products
            
            
    def _get_service_slug(self,product_name,products,feed_name,camera_name,api_base_url,HEADERS):
        '''
        Function to get the service slug and print data inventory
        '''
        # go through and find the service_slug for the desired product
        for product in products:
            if product['data']['common']['label'] == product_name:
                print(f"Services for camera '{camera_name}', feed '{feed_name}' and product '{product_name}':")
                services = product['services']
                for service in services:
                    service_slug = service['data']['common']['slug']
                    print(service_slug)  # Printing the service slug
                break

        #Get the data inventory information for the service slug
        inv_url = f"{api_base_url}/services/{service_slug}/inventory/"
        response = requests.get(inv_url, headers=HEADERS)

        # Check the status code of the response and grab the inventory 
        if response.status_code == 200:
            # Grab the response JSON for the inventory
            inventory_json = response.json()
        else:
            # Print error information if the request was not successful
            print(f"Failed to retrieve assets: {response.status_code} {response.text}")

        #Put the data inventory into a dataframe for further analsyis and to provide some basic summary stats
        inventory_data = inventory_json['results'][0]['values']
        df_inv = pd.DataFrame(inventory_data, columns=['Bin Start', 'Has Data?', 'Bin End', 'Count', 'Bytes', 'Data Start', 'Data End'])

        return service_slug,df_inv

    
    def _get_elements(self,service_slug,start_time,end_time,interval_minutes,api_base_url,HEADERS):
        '''
        Function to create and view the download urls or elements
        '''

        params = {
            'starting_after': start_time,
            'starting_before': end_time,
            'service': service_slug
        }

        #Set the base_url now to avoid including elements in the paginated urls
        base_url = f'{api_base_url}/elements/'
        all_elements = []
        page = 1

        #Run through the response for each page, adding to all_elements with the additional results and print updates to the screen
        while True:
            print(f"Fetching page: {page}")
            response = requests.get(base_url, headers=HEADERS, params=params)
            if response.status_code != 200:
                print(f"Failed to fetch page {page}: {response.status_code}")
                break
            data = response.json()
            all_elements.extend(data['results'])
            print(f"Received {len(data['results'])} elements, total elements collected: {len(all_elements)}")

            # Update the URL for the next request or break the loop if no more pages
            next_url = data.get('pagination', {}).get('next')
            if not next_url:
                print("No more pages.")
                break
            base_url = next_url
            params = None  # Ensure subsequent requests don't duplicate parameters
            page += 1

        # Now use the interval_minutes specified to filter the returned elements to only grab the images on certain intervals
        filtered_elements = []
        print("Timestamps of filtered elements")
        for element in all_elements:
            timestamp_str = element['data']['extents']['temporal']['min']
            timestamp = datetime.datetime.fromisoformat(timestamp_str)
            if timestamp.minute % interval_minutes == 0:
                filtered_elements.append(element)
                print(timestamp)
        return filtered_elements


    def _download_imagery(self,filtered_elements,save_dir):
        '''
        Function to download the data.
        '''

        #Now generate the download ulrs for the images
        download_urls = []
        for element in filtered_elements:
            try:
                download_url = element['data']['properties']['url']
                download_urls.append(download_url)
            except KeyError:
                print("Unexpected element structure:", element)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Beginning imagery download")
        filenames = []
        for url in download_urls:
            filename = os.path.join(save_dir, os.path.basename(url))
            filenames.append(filename)
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            print(".", end='')
            
            if not os.path.exists(filename):
                with open(filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
        print()

        print(f"Download complete. Downloaded {len(download_urls)} images to {save_dir}")

        return filenames
    
    def _local2ISO(self,local_time,camera_name):
        # Get the time zone using the state the camera is in - need to search for the state name in the camera name #
        sbool = [' '+GeoDB().state_abbrevs[i] in camera_name for i in range(50)]
        state = np.array(GeoDB().state_abbrevs)[np.array(sbool)][0]
        tz = GeoDB().tzs[state]
        tz_formal = GeoDB().tz_formals[tz[0]][0]
        # Get the full datetime object and assign time zone #
        dt_local = datetime.datetime(int(local_time[0:4]),
                                     int(local_time[4:6]),
                                     int(local_time[6:8]),
                                     int(local_time[8:10]),
                                     int(local_time[10:12]))
        dt_local = pytz.timezone(tz_formal).localize(dt_local)
        # Convert to UTC and make ISO #
        dt_utc = dt_local.astimezone(pytz.timezone('UTC'))
        ISO = dt_utc.isoformat()
        return ISO


class GeoDB:
    def __init__(self):
        self.state_abbrevs = [
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
                                                                ]
        self.tzs = {'ME': ['Eastern'],
                    'NH': ['Eastern'],
                    'MA': ['Eastern'],
                    'RI': ['Eastern'],
                    'CT': ['Eastern'],
                    'NY': ['Eastern'],
                    'NJ': ['Eastern'],
                    'PA': ['Eastern'],
                    'DE': ['Eastern'],
                    'MD': ['Eastern'],
                    'VA': ['Eastern'],
                    'NC': ['Eastern'],
                    'SC': ['Eastern'],
                    'GA': ['Eastern'],
                    'FL': ['Eastern'],  # Florida spans two time zones
                    'AL': ['Central'],
                    'MS': ['Central'],
                    'LA': ['Central'],
                    'TX': ['Central'],
                    'WA': ['Pacific'],
                    'OR': ['Pacific'],
                    'CA': ['Pacific'],
                    'MI': ['Central'],  # Michigan spans two time zones (Great Lakes)
                    'IL': ['Central'],
                    'IN': ['Central'],  # Indiana spans two time zones
                    'OH': ['Eastern'],
                    'WI': ['Central']}
        
        self.tz_formals = {'Eastern':['America/New_York'],
                           'Central':['America/Chicago'],
                           'Pacific':['America/Los_Angeles']}