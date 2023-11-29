from __future__ import annotations
import math

class Vec:
    def __init__(self, degrees: float = 0, magnitude: float = 0):
        self.degrees = degrees
        self.magnitude = magnitude

    def components(self):
        x = self.magnitude * math.sin(math.radians(self.degrees))
        y = self.magnitude * math.cos(math.radians(self.degrees))
        return (x, y)
    
    def from_components(self, components: tuple[float, float]):
        self.magnitude = math.sqrt(components[0]**2 + components[1]**2)

        degree_offset = 0
        opposite = 0
        adjacent = 0
        
        if (components[0] * components[1]) > 0: # q1 or q3
            opposite = components[0]
            adjacent = components[1]
            if components[0] > 0: #q1
                degree_offset = 0
            else: #q3
                degree_offset = 180
        else: # q2 or q4
            opposite = components[1]
            adjacent = components[0]
            if components[0] > 0: #q4
                degree_offset = 90
            else: #q2
                degree_offset = 270
    
        if adjacent == 0:
            self.degrees = degree_offset + 90
        else:
            self.degrees = degree_offset + math.degrees(math.atan(abs(opposite/adjacent)))

    def add(self, vec: Vec):
        v1 = self.components()
        v2 = vec.components()
        v3 = Vec(0, 0)

        v3.from_components((v1[0] + v2[0], v1[1] + v2[1]))
        return v3

    def mult(self, scalar: float):
        out = Vec(self.degrees, self.magnitude)
        out.magnitude *= scalar
        return out
    
    def normalize(self):
        out = Vec(self.degrees, self.magnitude)
        out.magnitude = 1
        return out

    def dot(self, vec: Vec):
        v1 = self.components()
        v2 = vec.components()

        return v1[0]*v2[0] + v1[1]*v2[1]

    def proj(self, vec: Vec):
        alpha = abs(self.degrees-vec.degrees)

        if (alpha <= 90):
            return Vec(vec.degrees, self.dot(vec.normalize()))
        else:
            return Vec((vec.degrees+180)%360, self.dot(vec.normalize()))

    def __str__(self):
        return f"Dir: {self.degrees} | Mag: {self.magnitude}"
    
    def __repr__(self):
        return f"Dir: {self.degrees} | Mag: {self.magnitude}"

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
        return f"Lat: {self.lat} | Lon: {self.lon} | Distance: {round(self.distance,2)} \n| Azimuth: {round(self.azimuth,1)} | Elevation: {round(self.elevation,1)} | Cloud Cover: {self.cloud_cover} \n| Wind Dir: {self.wind_dir} | Wind Speed: {self.wind_speed}"
    
    def __repr__(self):
        return f"Lat: {self.lat} | Lon: {self.lon} | Distance: {round(self.distance,2)} \n| Azimuth: {round(self.azimuth,1)} | Elevation: {round(self.elevation,1)} | Cloud Cover: {self.cloud_cover} \n| Wind Dir: {self.wind_dir} | Wind Speed: {self.wind_speed}"
    
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

tilts = {
    "hood_front": {
        "left_stairs": [33.38, 27.64, 27.64, 27.63, 27.63, 27.63, 27.44, 27.44, 27.44],
        "left_center_3x4": [33.38, 33.38, 33.38, 27.64, 27.64, 27.64, 27.63, 27.63, 27.63, 27.44, 27.44, 27.44],
        "right_center_3x4": [33.38, 33.38, 33.38, 27.64, 27.64, 27.64, 27.63, 27.63, 27.63, 27.44, 27.44, 27.44],
        "right_stairs": [33.38, 27.64, 27.64, 27.63, 27.63, 27.63, 27.44, 27.44, 27.44]
    },
    "top_front": {
        "leftmost_top_3x2": [28.52, 28.52, 28.52, 22.16, 22.16, 22.16],
        "leftmost_center_3x2": [22.16, 22.16, 22.16, 22.16, 22.16, 22.16],
        "leftmost_bottom_3x2": [13.14, 13.14, 13.14, 13.14, 13.14, 13.14],
        "leftcenter_top_3x2": [28.52, 28.52, 28.52, 22.16, 22.16, 22.16],
        "leftcenter_center_3x2": [22.16, 22.16, 22.16, 22.16, 22.16, 22.16],
        "leftcenter_bottom_3x2": [13.14, 13.14, 13.14, 13.14, 13.14, 13.14],
        "rightcenter_top_3x2": [28.52, 28.52, 28.52, 22.16, 22.16, 22.16],
        "rightcenter_center_3x2": [22.16, 22.16, 22.16, 22.16, 22.16, 22.16],
        "rightcenter_bottom_3x2": [13.14, 13.14, 13.14, 13.14, 13.14, 13.14],
        "rightmost_top_4x3": [28.52, 28.52, 28.52, 28.52, 22.16, 22.16, 22.16, 22.16, 22.16, 22.16, 22.16, 22.16],
        "rightmost_bottom_4x3": [22.16, 22.16, 22.16, 22.16, 13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 13.14]
    },
    "top_back": {
        "leftmost_3x4": [13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 1.11, 1.11, 1.11, -3.64, -3.64, -3.64],
        "leftcenter_3x4": [13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 1.11, 1.11, 1.11, -3.64, -3.64, -3.64],
        "rightcenter_3x4": [13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 1.11, 1.11, 1.11, -3.64, -3.64, -3.64],
        "rightmost_4x4": [13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 13.14, 1.11, 1.11, 1.11, 1.11, -3.64, -3.64, -3.64, -3.64]
    },
    "back": {
        "leftmost_top_3x4": [-3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64],
        "leftcenter_top_3x4": [-3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64],
        "rightcenter_top_3x4": [-3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64],
        "rightmost_top_4x4": [-3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64],
        "leftmost_bottom_3x4": [-3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64],
        "leftcenter_bottom_3x4": [-3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64], 
        "rightcenter_bottom_3x4": [-3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64, -3.64],
        "rightmost_bottom-4x4": [-4, -4, -4, -4, -5, -5, -5, -5, -6, -6, -6, -6]
    }
}

#for independence to topeka, 2022
speed_limits = { #distance and speed in mph
    0: 20, #distance: speed limit
    1: 30,
    1.6: 35,
    7.1: 30,
    7.6: 25,
    7.9: 30,
    8.7: 40,
    16.7: 35,
    18.5: 45,
    20.1: 40,
    24.2: 35,
    24.5: 45,
    34.5: 50,
    38.5: 30,
    38.7: 50,
    43.3: 40,
    44.3: 30,
    44.6: 35,
    45.6: 40,
    46: 45,
    55.7: 55,
    60.4: 45,
    61.6: 55,
    69.6: 45,
    70.2: 60,
    76.6: 55,
    86.5: 45,
    87: 55,
    93.9: 45,
    94.9: 40,
    96: 30,
    97: 40,
    98: 30,
    98.25: 25
}