from geopy.geocoders import Nominatim
from typing import Dict, Tuple, Optional
import time

class Geocoder:
    def __init__(self):
        self.geolocator = Nominatim(user_agent='data_agent')
    
    def get_bounding_box(self,city_name:str):
        """ Get bounding for a city name using OpenStreetMap Nominatim
        Params:
            city_name: Name of the city to get bounding_box
        """
        try:
            location = self.geolocator.geocode(city_name)
            if not location:
                return None
            detailed_location = self.geolocator.geocode(
                city_name,
                exactly_one=True,
                addressdetails=True
            )
            
            bbox = detailed_location.raw['boundingbox']
            if not bbox:
                raise ValueError("Bounding box is empty or invalid")
            return {
                "min_lat": float(bbox[0]),
                "max_lat": float(bbox[1]),
                "min_lon": float(bbox[2]),
                "max_lon": float(bbox[3]),
                "center_lat": location.latitude,
                "center_lon": location.longitude,
                "name": location.address
            }
        except Exception as e:
            print(f"[GEOCODING ERROR]: {e}")
            return None
    
    def bbox_to_geojson(self,bbox:Dict) -> Optional[Dict]:
        """ Convert bounding box to GeoJSON polygon """
        

        return {
            "type": "Polygon",
            "coordinates": [[
                [bbox["min_lon"], bbox["min_lat"]],
                [bbox["max_lon"], bbox["min_lat"]],
                [bbox["max_lon"], bbox["max_lat"]],
                [bbox["min_lon"], bbox["max_lat"]],
                [bbox["min_lon"], bbox["min_lat"]]
            ]]
        }

"""
Testing:-

geo = Geocoder()
bbox = geo.get_bounding_box(city_name='New York City')
print(bbox)
print("=="*10)
print(geo.bbox_to_geojson(bbox=bbox))

Output:
    {'min_lat': 40.476578, 'max_lat': 40.91763, 'min_lon': -74.258843, 'max_lon': -73.700233, 'center_lat': 40.7127281, 'center_lon': -74.0060152, 'name': 'City of New York, New York, United States of America'}
====================
    {'type': 'Polygon', 'coordinates': [[[-74.258843, 40.476578], [-73.700233, 40.476578], [-73.700233, 40.91763], [-74.258843, 40.91763], [-74.258843, 40.476578]]]}
"""