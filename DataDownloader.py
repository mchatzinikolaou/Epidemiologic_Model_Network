import urllib.request
import csv
#Add a method to check for updates



def DownloadData(confirmed_url,deaths_url,recovered_url):
    print("Starting data download...")

    urllib.request.urlretrieve(confirmed_url, './Confirmed_cases.csv')
    urllib.request.urlretrieve(deaths_url, './Deaths_cases.csv')
    urllib.request.urlretrieve(recovered_url, './Recovered_cases.csv')

    print('Download finished.')

confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
deaths_url    = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
recovered_url  = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'



print('Reading from Confirmed cases: (' + confirmed_url +')')

print('Reading from Death cases: (' + deaths_url +')')

print('Reading from Recovered cases: (' + recovered_url +')')






#Create a dictionary of countries(nodes) and time-series of cases deaths etc.