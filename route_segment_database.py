import sqlite3
import csv

#Change value of variable 'filename' to the desired segment file
filename = "AL. Topeka Loop.csv"

#Reads each row of data in the csv file and outputs a list of tuples containing data from each row
def data_rows(filename):
    rows = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for index, row in enumerate(csv_reader):
            if index >= 2:
                rows.append(tuple(row))
    return rows

#Stores the list in a variable 'data'
data = data_rows(filename)

#Establish a connection to 'data.sqilte' database
connection = sqlite3.connect("data.sqlite")

#Creates a cursor object used to execute SQL queries
cursor = connection.cursor()

#Initializes the table with column headers and their data types
table = """
create table Data (
Lat float,
Lon float,
Distance float,
Azimuth float,
Elevation float,
Cloud_Cover float,
Wind_Direction float,
Wind_Speed float
);
"""

#Creates the table in the database file
cursor.execute(table)

#Inserts the data values from the chosen csv file into the database
cursor.executemany("insert into Data values (?,?,?,?,?,?,?,?)", data)

connection.commit()

connection.close()