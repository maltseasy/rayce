from __future__ import annotations

class Coordinate:
    def __init__(self, lon: float, lat: float, elevation: float):
        self.lon = lon
        self.lat = lat
        self.elevation = elevation

    def __str__(self):
        return f"Lat: {self.lat} | Lon: {self.lon} | Elevation: {self.elevation}"
    
    def __repr__(self):
        return f"Lat: {self.lat} | Lon: {self.lon} | Elevation: {self.elevation}"
    
class Placemark:
    def __init__(self, name: str, coords: list[Coordinate]):
        self.name = name
        self.coords = coords

    def __str__(self):
        return f"{self.name}: {self.coords}"
    
    def __repr__(self):
        return f"{self.name}: {self.coords}"
    
class Checkpoint:
    def __init__(self, lat: float, lon: float, distance: float, azimuth: float, elevation: float, cloud_cover: int, wind_dir: float, wind_speed: float):
        self.lat = lat
        self.lon = lon
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.cloud_cover = cloud_cover
        self.wind_dir = wind_dir
        self.wind_speed = wind_speed
    
    def __str__(self):
        return f"Lat: {self.lat} | Lon: {self.lon} | Distance: {round(self.distance/1000,1)} \n| Azimuth: {round(self.azimuth,1)} | Elevation: {round(self.elevation,1)} | Cloud Cover: {self.cloud_cover} \n| Wind Dir: {self.wind_dir} | Wind Speed: {self.wind_speed}"
    
    def __repr__(self):
        return f"Lat: {self.lat} | Lon: {self.lon} | Distance: {round(self.distance/1000,1)} \n| Azimuth: {round(self.azimuth,1)} | Elevation: {round(self.elevation,1)} | Cloud Cover: {self.cloud_cover} \n| Wind Dir: {self.wind_dir} | Wind Speed: {self.wind_speed}"
    
class Route:
    def __init__(self, name: str, checkpoints: list[Checkpoint]):
        self.name = name
        self.checkpoints = checkpoints
    def __str__(self):
        return f"{self.name}: {self.checkpoints}"
    
    def __repr__(self):
        return f"{self.name}: {self.checkpoints}"
    
def load_csv(name: str):
    import csv
    path = f"data/generated/{name}.csv"
    checkpoints: list[Checkpoint] = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        next(reader)
        for row in reader:
            checkpoint = Checkpoint(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5]), float(row[6]), float(row[7]))
            checkpoints.append(checkpoint)
    return Route(name, checkpoints)
    
def heversine_and_azimuth(lon1: float, lat1: float, lon2: float, lat2: float):
    import math
    R = 6371.0  #Radius of the Earth in km
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    azimuth = math.atan2(math.sin(dlon)*math.cos(lat2), math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dlon))
    
    azimuth %= math.pi*2

    return distance, azimuth

def solar_power_out(max_irradiance: float, cloud_cover: float):
    # from sklearn.linear_model import LinearRegression
    # import numpy as np

    # cell_power_x_data = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000]).reshape((-1,1))
    # cell_power_y_data = np.array([54, 83, 112, 141, 170, 200, 230, 260, 290])
    # irradiance_to_power_model = LinearRegression()
    # irradiance_to_power_model.fit(cell_power_x_data, cell_power_y_data)
    eff = 0.24
    solar_irradiance = (1 - cloud_cover) * max_irradiance
    power_out = solar_irradiance * eff
    return power_out