"""
Search utilities for NWLON and WebCOOS cameras.

Includes functions to calculate distances between coordinates,
retrieve station and camera metadata, and create distance tables.
"""
import math
import requests
from pandas import DataFrame


def haversine(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees: (lat, lon)).
    Returns the distance in kilometers.
    """
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    # Radius of the Earth in kilometers (mean radius)
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = r * c
    return distance


def get_station_metadata(station: int) -> dict:
    """
    Use a station ID to retriece metadata
    """
    url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}.json"
    response = requests.get(url, timeout=5)
    data = response.json()
    metadata = data.get('stations', [])
    if len(metadata) != 1:
        raise ValueError(f"No metadata found for station {station}")
    return metadata[0]

def get_all_cameras_with_metadata(api_token: str) -> list[dict]:
    """
    Get all camera metadata from WebCOOS API
    """
    url = "https://app.webcoos.org/webcoos/api/v1/assets/"
    headers = {
        "Authorization": f"Token {api_token}",
    }
    response = requests.get(url, headers=headers, timeout=30)
    data = response.json()
    return data.get('results', [])

def get_single_camera_with_metadata(uuid: str, api_token: str) -> dict:
    """
    If you know the camera ID, get the matching metadata
    """
    url = f"https://app.webcoos.org/webcoos/api/v1/assets/{uuid}/"
    headers = {
        "Authorization": f"Token {api_token}",
    }
    response = requests.get(url, headers=headers, timeout=30)
    data = response.json()
    return data


def get_nearby_stations(
    longitude: float,
    latitude: float,
    distance: float
):
    """
    Get nearby NWLON stations within a specified distance from a given point.
    """
    url = (
        "https://mapservices.weather.noaa.gov/static/rest/services/"
        "NOS_Observations/CO_OPS_Stations/FeatureServer/0/query"
    )
    params = {
        "where": "affil LIKE '%NWLON%'",
        "geometry": f"{longitude},{latitude}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "distance": distance,
        "units": "esriSRUnit_Kilometer",
        "outFields": "id,name,state,affil,latitude,longitude,metaapi",
        "returnGeometry": "true",
        "outSR": 4326,
        "f": "geojson",
    }
    response = requests.get(url, params=params, timeout=30)
    stations = response.json()
    # ArcGIS can return application errors with HTTP status 200.
    if "error" in stations:
        raise RuntimeError(f"ArcGIS query failed: {stations['error']}")
    return stations

def calculate_camera_distance_table(
    station_id: int,
    api_token: str,
    limit: int = 1
) -> DataFrame:
    """
    Create a table of cameras sort by distance from the station.
    Have to load all cameras to know which is closest because we
    only have GPS coordinates.
    """
    metadata = get_station_metadata(station_id)
    coordinates = (metadata['lng'], metadata['lat'])
    results = []
    for each in get_all_cameras_with_metadata(api_token):
        name = each['data']['common']['label']
        camera_coords = each['package']['data']['properties']['location']['coordinates']
        results.append({
            'uuid': each['uuid'],
            'name': name,
            'longitude': camera_coords[0],
            'latitude': camera_coords[1],
            'distance': haversine(camera_coords, coordinates)
        })
    return DataFrame(results).set_index('distance').sort_index().head(limit)

def calculate_station_distance_table(
    camera_id: str,
    api_token: str,
    limit: int = 1
) -> DataFrame:
    """
    Create a table of stations sort by distance from the given coordinates
    """
    camera = get_single_camera_with_metadata(camera_id, api_token)
    coordinates = camera['data']['properties']['location']['coordinates']
    name = camera['data']['common']['label']
    stations = get_nearby_stations(
        longitude=coordinates[0],
        latitude=coordinates[1],
        distance=200
    )
    station_info = []
    for feature in stations["features"]:
        station = feature["properties"]
        station_info.append({
            'uuid': station['id'],
            'name': station['name'],
            'lon': station["longitude"],
            'lat': station["latitude"],
            'distance': haversine(coordinates, (station["longitude"], station["latitude"])),
            'camera': name
        })
    return DataFrame(station_info).set_index('distance').sort_index().head(limit)
