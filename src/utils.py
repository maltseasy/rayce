from classes import *
import xml.etree.ElementTree as ET
import folium
import pandas as pd
import csv
import math
import pandas as pd

def calc_distance(coords: list[Coordinate], current_coord: Coordinate):
    sum_distance = 0
    stop_index = coords.index(current_coord)
    for i in range(stop_index+1):
       if i > 0: sum_distance += heversine_and_azimuth(coords[i-1].lon, coords[i-1].lat, coords[i].lon, coords[i].lat)[0]
    return sum_distance * 1000 #convert to meters

# TODO: this should be a method the Coordinate class, either static or instance
def calc_azimuth(coords: list[Coordinate], current_coord: Coordinate):
   current_index = coords.index(current_coord)

   if(current_index < len(coords) - 1):
      next_coord = coords[current_index + 1]

      lat1 = math.radians(current_coord.lat)
      lon1 = math.radians(current_coord.lon)
      lat2 = math.radians(next_coord.lat)
      lon2 = math.radians(next_coord.lon)
      
      dlon = lon2 - lon1
      azimuth = math.degrees(math.atan2(math.sin(dlon)*math.cos(lat2), math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dlon)))
      azimuth %= 360
      return azimuth
   else:
      return 0

def parse_kml_file(filename):
    root = ET.parse(filename).getroot()
    placemarks = []
    for child in root[0]:
        if child.tag == '{http://www.opengis.net/kml/2.2}Placemark':
            # name tag
            name = child.find('{http://www.opengis.net/kml/2.2}name').text
            coords = []
            # coordinates tag
            p_coords = child.find('{http://www.opengis.net/kml/2.2}LineString').find('{http://www.opengis.net/kml/2.2}coordinates')
            for line in p_coords.text.split("\n"):
                line = line.strip() 
                if (len(line) == 0):
                    continue

                parts = line.split(",")
                coords.append(Coordinate(float(parts[0]),float(parts[1]),float(parts[2])))

            placemarks.append(Placemark(name, coords))


    return placemarks

def batch_elems(elems: list, size: int):
    if (len(elems) == 0):
        return []
    
    out = [[]]

    for elem in elems:
        if len(out[-1]) < size:
            out[-1].append(elem)
        else:
            out.append([elem])

    return out

def flat_2d(input: list[list]):
    out = []
    for i in input:
        out.extend(i)

    return out

def heversine_and_azimuth(lon1: float, lat1: float, lon2: float, lat2: float):
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

def heversine(lon1: float, lat1: float, lon2: float, lat2: float):
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
    
    return distance
   
# data: [(distance, elevation, cloud_cover_%, wind_dir, wind_speed)]
def write_csv(name: str, data: list[tuple[float, float, float, float, int, int, int]]):
    path = f"../data/generated/{name}.csv"

    with open(path, 'w') as csvfile:
        csvfile.write(f'{name}\nLat,Lon,Distance,Azimuth,Elevation,Cloud Cover,Wind Direction,Wind Speed\n')
        for row in data:
            csvfile.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]}\n")
   
def load_csv(name: str):
    path = f"../data/generated/{name}.csv"
    checkpoints: list[Checkpoint] = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        next(reader)
        for row in reader:
            checkpoint = Checkpoint(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5]), float(row[6]), float(row[7]))
            checkpoints.append(checkpoint)
    return Route(name, checkpoints)

def match_coords_to_checkpoint(route: Route, location: tuple[float,float]):
    closest_point = (0, math.inf)
    for (i, coord) in enumerate(route.checkpoints):
        new_distance,_ = heversine_and_azimuth(coord.lon, coord.lat, location[1], location[0])
        if new_distance < closest_point[1]:
            closest_point = (i,new_distance)
    
    return closest_point

def route_to_list(route: Route):
    coords: list[tuple[float, float]] = [] 
    for checkpoint in route.checkpoints:
        coords.append((checkpoint.lat, checkpoint.lon))
    return coords

def create_map(route: Route, random_coord: tuple[float, float]):
    coords = route_to_list(route)
    m = folium.Map()
    folium.Marker(random_coord, popup="random", draggable=True).add_to(m)
    for coord in coords[0::50]:
        folium.Marker(coord).add_to(m)
    folium.PolyLine(coords, weight=5, opacity=1).add_to(m)
    nearest_coord_index = match_coords_to_checkpoint(route, random_coord)[0]
    nearest_coord = (route.checkpoints[nearest_coord_index].lat, route.checkpoints[nearest_coord_index].lon)
    folium.PolyLine([random_coord, nearest_coord],color="#FF0000", weight=5, opacity=1,tooltip="Ok").add_to(m)
    df = pd.DataFrame(coords).rename(columns={0: 'lat', 1: 'lon'})
    print(df)
    sw = df[['lat', 'lon']].min().values.tolist()
    ne = df[['lat', 'lon']].max().values.tolist()
    m.fit_bounds([sw, ne])
    return m

def solar_power_out(max_irradiance: float, cloud_cover: float):
    eff = 0.24
    solar_irradiance = (1 - cloud_cover) * max_irradiance
    power_out = solar_irradiance * eff
    return power_out