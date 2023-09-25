import math

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
                    coordinate = str(total_distance) + "," + elevation + '\n'
                    loop_coordinates.append(coordinate)

        loops.append(loop_coordinates)
        start = end + len('</coordinates>') 


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
        filename = loop_filenames[loop_index]

        with open(filename, 'w') as csvfile:
            csvfile.write(loop_name+'\n'+'Distance (km), Elevation (m)\n')
            for coord in loop_coordinates:
                csvfile.write(coord)
        loop_index += 1

write_coords(True, 'data/Optional Loops.kml')
write_coords(False, 'data/Main Route.kml')
