import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    EARTH_ENGINE_CREDENTIALS = os.getenv('EARTH_ENGINE_CREDENTIALS')
    
    # Data Sources
    NLCD_BASE_URL = "https://www.mrlc.gov/data"
    WORLDPOP_BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
    
    # Processing Settings
    DEFAULT_CRS = "EPSG:5070"
    DEFAULT_RESOLUTION = 30  # meters
    DEFAULT_YEAR = 2020
    
    # Output Settings
    OUTPUT_DIR = "outputs"
    TEMP_DIR = "temp"
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000

settings = Settings()
