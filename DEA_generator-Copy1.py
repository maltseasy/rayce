#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import requests
import json


#equation to calculate distance between coordinates 
def heversine(lon1, lat1, lon2, lat2):
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

#get data from the Microsoft weather API
def write_weather_data(loops):
  #https://atlas.microsoft.com/weather/route/json?api-version=1.1&query={query}
  #https://atlas.microsoft.com/weather/currentConditions/json
    base_url = "#https://atlas.microsoft.com/weather/route/json?api-version=1.1&query={query}"
    subscription_key = "0AaKvkxb1Q452IX0NDW0zmN9TBKu1gquS5zoll4d5Ws" #primary key

    #cloud_cover_data = []
    weather_data = []

    for i in loops:
        latitude, longitude = i
        params = {
             "api-version": "1.1",
             "subscription_key": subscription_key,
             "query": f"{latitude},{longitude}"
             }
        response = requests.get(base_url, params = params)
        if response.status_code == 200: #response code - OK
            weather_data = response.json() 
            cloudCover = weather_data.get("cloudCoverPercentage", "N/A") 
            windDirection = weather_data.get("windDirectionDegrees", "N/A")
            windSpeed = weather_data.get("windSpeedKph", "N/A")
            weather_data.append(cloudCover, windDirection, windSpeed)
        
        else:
            print("error response for {loops}: {response.text}")  
    return weather_data

            
#extract coordinates from kml 

def write_coords(condition,kml_filename):
    with open(kml_filename, 'r') as kml_file:
        kml_data = kml_file.read()

    loops = []
    start = 0
    while True:
        start = kml_data.find('<coordinates>', start)
        if start == -1:
            break  
        end = kml_data.find('</coordinates>', start)
        coord_text = kml_data[start+len('<coordinates>'):end].strip()

        lines = coord_text.split("\n")
        total_distance = 0.0 
        loop_coordinates = []

        #to get next coordinate
        for i in range(len(lines)):
            line = lines[i].strip()
            lon2_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            lat2_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

            if line and lon2_line and lat2_line:
                parts = line.split(",")
                lon2 = float(lon2_line.split(",")[0])
                lat2 = float(lat2_line.split(",")[1])

                #write distance and elevation into list
                if len(parts) == 3:
                    elevation = parts[2]
                    if i == 0:
                        distance = 0.0 #first distance is zero
                else:
                    distance = heversine(float(parts[0]), float(parts[1]), lon2, lat2)
                total_distance += distance
                coordinate = str(total_distance) + "," + elevation
                loop_coordinates.append(coordinate)
                    

        loops.append(loop_coordinates)
        start = end + len('</coordinates>') 
        
        
    speed_mph = 35 #assume 35mph average speed
    for i in range(len(loops)):
        for (j, pair) in enumerate(loops[i]):
            thisTuple = pair.split(",")
            distance = float(thisTuple[0])
            elevation = float(thisTuple[1])
            if j + 1 < len(loops[i]):
                nextTuple = loops[i][j+1].split(",")
                nextDistance = float(nextTuple[0])
                nextElevation = float(nextTuple[1])
                angle = math.degrees(math.atan((nextElevation - elevation) / (nextDistance*1000 - distance*1000)))
                time = distance / speed_mph # ETA 
                cloudCover, windDirection, windSpeed = write_weather_data(loops)
                row = str(distance) + "," + str(elevation) + "," + str(angle) + "," + str(time) + "," + str(cloudCover) + "," + str(windDirection) + "," + str(windSpeed) + "\n"
            else:
                row = str(distance) + "," + str(elevation) + "," + "0" + "," + str(time) + "," + str(cloudCover) + str(windDirection) + "," + str(windSpeed) + "\n"
            loops[i][j] = row
            

    #put data into csv files
    #optional loops
    if condition == True:
        loop_names = ['Topeka Loop','Grand Island Loop','Casper Loop','Montpelier Loop','Lander Loop','Pocatello Loop']
        loop_filenames = ['Topeka Loop.csv','Grand Island Loop.csv','Casper Loop.csv','Montpelier Loop.csv','Lander Loop.csv','Pocatello Loop.csv']
        
    #main route
    if condition == False:
        loop_names = ['A. Independence to Topeka','B. Topeka to Grand Island','C. Grand Island to Gering','D. Gering to Casper','E. Casper to Lander','F. Lander to Montpelier','G. Montpelier to Pocatello','H. Pocatello to Twin Falls']
        loop_filenames =  ['Independence to Topeka.csv','Topeka to Grand Island.csv','Grand Island to Gering.csv','Gering to Casper.csv','Casper to Lander.csv','Lander to Montpelier.csv','Montpelier to Pocatello.csv','Pocatello to Twin Falls.csv']
            
    loop_index = 0
    for loop_coordinates in loops:
        loop_name = loop_names[loop_index]
        filename = "data/generated/"+loop_filenames[loop_index]

        with open(filename, 'w') as csvfile:
            csvfile.write(loop_name+'\n'+'Distance (km), Elevation (m), Angle (deg), Time (min), Cloud Cover (%), Wind Dir (deg), Wind Speed (km/h) \n')
            for coord in loop_coordinates:
                csvfile.write(coord)
        loop_index += 1

write_coords(True, 'data/Optional Loops.kml')
write_coords(False, 'data/Main Route.kml')


# In[ ]:




