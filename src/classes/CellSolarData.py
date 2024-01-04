from classes import Checkpoint
from astral import LocationInfo
from astral.sun import azimuth,elevation
from utils import solar_power_out
# from constants import CELL_AREA
from soc_estimation import CELL_AREA
import math
import datetime

class CellSolarData:
    def __init__(self, coord: Checkpoint, time: datetime.datetime, tilt: float):
        l = LocationInfo()
        l.name = f'{coord.lat},{coord.lon}'
        l.region = 'United States'
        l.timezone = 'US/Central' #update to be dynamic
        l.latitude = coord.lat
        l.longitude = coord.lon
        self.heading_azimuth_angle = coord.azimuth
        self.heading_azimuth_angle = 180 + self.heading_azimuth_angle if tilt < 0 else self.heading_azimuth_angle
        self.heading_azimuth_angle %= 360
        self.tilt = tilt * -1 if tilt < 0 else tilt
        self.lat = coord.lat
        self.lon = coord.lon
        self.elevation = coord.elevation / 1000 #convert to km
        self.time = time
        self.sun_elevation_angle = elevation(l.observer, time)
        self.sun_elevation_angle = max(0, self.sun_elevation_angle)
        self.sun_azimuth_angle = azimuth(l.observer, time)
        self.air_mass = 1/(math.cos(math.radians(90-self.sun_elevation_angle)) + 0.50572*(96.07995-(90-self.sun_elevation_angle))**-1.6364)
        self.incident_diffuse = 1.1*1.353*((1-0.14*(self.elevation / 1000))*0.7**self.air_mass**0.678 + 0.14*(self.elevation / 1000)) 
        self.cell_diffuse = self.incident_diffuse*(math.cos(math.radians(self.sun_elevation_angle))*math.sin(math.radians(self.tilt))*math.cos(math.radians(self.heading_azimuth_angle - self.sun_azimuth_angle)) + math.sin(math.radians(self.sun_elevation_angle))*math.cos(math.radians(self.tilt)))
        self.cell_irradiance = self.cell_diffuse * 1000 #convert to watts/m^2
        self.cloud_cover = coord.cloud_cover / 100
        cell_power_out = solar_power_out(self.cell_irradiance, self.cloud_cover) * CELL_AREA #watts
        if isinstance(self.incident_diffuse, complex):
            print(self.air_mass, self.incident_diffuse, self.sun_elevation_angle, self.tilt, self.heading_azimuth_angle, self.sun_azimuth_angle, time)
        self.cell_power_out = cell_power_out if cell_power_out > 0 else 0