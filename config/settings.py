import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    EARTH_ENGINE_CREDENTIALS = os.getenv('EARTH_ENGINE_CREDENTIALS')
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    # Data Sources
    NLCD_BASE_URL = "https://www.mrlc.gov/data"
    WORLDPOP_BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020"
    
    # Processing Settings
    DEFAULT_CRS = "EPSG:5070"
    DEFAULT_RESOLUTION = 30  # meters
    DEFAULT_YEAR = 2020
    
    # Performance Settings
    USE_EARTH_ENGINE_POPULATION = True  # Use EE for faster population downloads
    DOWNLOAD_TIMEOUT = 1200  # seconds (20 minutes) for large files
    MAX_PIXELS_PER_DOWNLOAD = 100_000_000  # 100 million pixels max (adjust for large cities)
    AUTO_SCALE_RESOLUTION = True  # Auto-reduce resolution for very large areas
    
    # Output Settings
    OUTPUT_DIR = "outputs"
    TEMP_DIR = "temp"
    
    # API Settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000

settings = Settings()
