from __future__ import annotations
from utils import *
from classes import *
import requests

API_KEY = "0AaKvkxb1Q452IX0NDW0zmN9TBKu1gquS5zoll4d5Ws"
API_URL = "https://atlas.microsoft.com/weather/route/json"
SPEED = 55

def weather_data_req(data: list[tuple[Coordinate, float]]):
    query = ":".join(map(lambda d: f"{d[0].lat},{d[0].lon},{d[1]}", data))
    
    params = {
        "api-version": "1.1",
        "subscription-key": API_KEY,
        "query": query
    }

    res = requests.get(API_URL, params)
    if not (res.status_code == 200):
        raise Exception(res)

    # json = res.json()
    # assert(len(json["waypoints"]) == len(data))
    return res

def get_weather_data(coords: list[Coordinate], time_offset: int = 0, speed: float = SPEED):
    if len(coords) == 0: return []

    coords_with_time: list[tuple[Coordinate, float]] = []
    running_distance = 0

    for i in range(len(coords)):
        c1 = coords[i]
        if i > 0:
            c1 = coords[i-1]
            
        c2 = coords[i]
        distance,_ = heversine_and_azimuth(float(c1.lon), float(c1.lat), float(c2.lon), float(c2.lat))
        running_distance = running_distance + distance
        
        coords_with_time.append(
            (
                c2,
                min((running_distance / speed * 60)+time_offset, 120) # max 120 minutes
            )
        )
    
    # list[list[tuple[Coordinate, float]]]
    # [
    #     [
    #         (coord, eta),
    #         (coord, eta),
    #         (coord, eta),
    #         ...
    #     ],
    #     [
    #         (coord, eta),
    #         (coord, eta),
    #         (coord, eta),
    #         ...
    #     ],
    #     ...
    # ]
    batched: list[list[tuple[Coordinate, float]]] = batch_elems(coords_with_time, 60)

    responses = []

    for batch in batched:
        res = weather_data_req(batch)
        if not (res.status_code == 200):
            print(res.json())
            raise Exception(res.json())
        responses.append(res.json())

    # return list of waypoints
    return flat_2d(map(lambda r: r["waypoints"], responses))
  
def generate_data(path_to_kml: str):
    print(f"[{path_to_kml}] Generating data...")
    placemarks = parse_kml_file(path_to_kml)

    for placemark in placemarks:
        print(f"[{placemark.name}] Fetching data...")
        #convert elevation to meters
        route_data: list[tuple[float, float, float, float, float]] = list(map(lambda p: (p.lat, p.lon, calc_distance(placemark.coords, p), calc_azimuth(placemark.coords, p), p.elevation * 0.3048), placemark.coords))
        weather_data: list[tuple[int, int ,int]] = list(map(lambda w: (w["cloudCover"], w["wind"]["direction"]["degrees"], w["wind"]["speed"]["value"] if w["wind"]["speed"]["unit"] == "km/h" else w["wind"]["speed"]["value"] * 1.60934), get_weather_data(placemark.coords)))

        assert(len(weather_data) == len(route_data))
        
        csv_data: list[tuple[float, float, float, float, float, int, int, int]] = []
        for i in range(len(route_data)):
            csv_data.append((route_data[i][0],route_data[i][1], route_data[i][2], route_data[i][3], route_data[i][4], weather_data[i][0], weather_data[i][1], weather_data[i][2]))
            
        write_csv(placemark.name, csv_data)
        print(f"[{placemark.name}] Created CSV!")

# generate_data("data/Main Route.kml")