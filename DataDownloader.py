import urllib.request
import csv
import numpy as np
import datetime


# TODO
# Add a method to check for updates

def DownloadData(confirmed_url, deaths_url, recovered_url):
    print("Starting data download...")

    urllib.request.urlretrieve(confirmed_url, './Confirmed_cases.csv')
    urllib.request.urlretrieve(deaths_url, './Deaths_cases.csv')
    urllib.request.urlretrieve(recovered_url, './Recovered_cases.csv')

    print('Download finished.')


# Read case Time Series from a file
def readFromFile(name):
    Countries = []
    TS = []
    # Open csv and process it
    with open(name, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')

        for row in reader:
            Countries.append(row[1])
            TS.append(row[4:])

        # Add same country data (compact)
        CompactCountries = []
        CompactCases = []

        # TODO
        # MAKE cases integers
        # testlist=list(map(list1,testlist))
        j = -1
        for i in range(1, len(Countries)):
            country = Countries[i]
            previous = Countries[i - 1]
            if country != previous:
                j += 1
                CompactCountries.append(country)
                CompactCases.append(np.array(TS[i]).astype(int))
            else:
                CompactCases[j] = np.add(CompactCases[j], np.array(TS[i]).astype(int))
        FinalData = {CompactCountries[i]: CompactCases[i] for i in range(len(CompactCountries))}
        return FinalData

# URLS AND FILENAMES
confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
confirmed_filename = './Confirmed_cases.csv'

deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
deaths_filename = './Deaths_cases.csv'

recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
recovered_filename = './Recovered_cases.csv'

#checkForUpdates('./update_checker.log',confirmed_url,deaths_url,recovered_url)

#DownloadData(confirmed_url, deaths_url, recovered_url)

print('Reading from Confirmed cases: (' + confirmed_url + ')')
Confirmed = readFromFile(confirmed_filename)

print('Reading from Death cases: (' + deaths_url + ')')
Deaths = readFromFile(deaths_filename)

print('Reading from Recovered cases: (' + recovered_url + ')')
Recovered = readFromFile(recovered_filename)


import os
os.system("pause")







""""
TODO
2. CREATE CONNECTIONS BETWEEN COUNTRIES
    2.1 DOWNLOAD DATA FOR AIRLINES , MARINE TRAFFIC AND ROAD CONNECTIONS
    2.2 CREATE THE GRAPH USING POPULATION DATA AND THE DATA BEFORE
"""

"""
def checkForUpdates(date_file,confirmed_url,deaths_url, recovered_url):
    with open(date_file,'r') as datefile:
        #read date
        old_date=datefile.read()
        curr_date=datetime.date.today()

    if (old_date<curr_date) or (old_date==None):
        with open(date_file,'w') as datefile:
            DownloadData(confirmed_url, deaths_url, recovered_url)
            datefile.write(curr_date)
"""

