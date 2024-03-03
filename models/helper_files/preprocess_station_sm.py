import csv
import math
import os
import urllib.request

import matplotlib.pyplot as plt
import mercantile,fiona
import netCDF4 as nc
from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd

class QuadhashHelper:
    def get_quad_tile(self, lat, lon, precision):
        ret = mercantile.tile(lon,lat,precision)
        return ret

    def get_quad_key_from_tile(self,x, y, zoom):
        return mercantile.quadkey(x, y, zoom)

    # GIVEN A QUAD_KEY, GET THE CORRESPONDING QUAD TILE
    def get_tile_from_key(self,key):
        return mercantile.quadkey_to_tile(key)

    # GET QUADHASH STRING OF A GIVEN COORDINATE
    def get_quad_key(self,lat, lon, zoom):
        tile = self.get_quad_tile(lat, lon, precision=zoom)
        return self.get_quad_key_from_tile(tile.x, tile.y, tile.z)

class CRNProcessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj

    def download_crn_raw_files(self):
        path = "/s/chopin/a/grad/paahuni/cl_3.8/soil_moisture_project/stationist.csv"
        with open(path, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/daily01/2019/" + row[0]
                destination_path =  "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/crn/raw/" + row[0]

                urllib.request.urlretrieve(url, destination_path)

    def split_daily_soil_moisture(self):
        input_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/crn/raw/"
        mystaion = set()
        for station in os.listdir(input_path):
            with open(input_path + station, "r") as file:
                for line in file:

                    splited = line.strip().split(" ")
                    data = [x for x in splited if x != '']
                    date = data[1]
                    lon = float(data[3])
                    lat = float(data[4])

                    mystaion.add((lat, lon))

                    soil_moisture_5_cm = float(data[18])
                    soil_moisture_10_cm = float(data[19])
                    soil_moisture_20_cm = float(data[20])

                    quad = self.quadHelper.get_quad_key(lat, lon, 14)

                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm) + "\n")
                        file.close()

                    if soil_moisture_10_cm >= 0 and soil_moisture_10_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/10/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_10_cm) + "\n")
                        file.close()

                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm) + "\n")
                        file.close()
        print("No. of stations in USCRN network: ", len(mystaion))

class SoilScapeProcessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj

    def convert_DatetimeGregorian_to_string(self, date_time_obj):
        year = date_time_obj.year
        month = date_time_obj.month
        day = date_time_obj.day
        hour = date_time_obj.hour
        minute = date_time_obj.minute
        second = date_time_obj.second
        datetime_obj = datetime(year, month, day, hour, minute, second)
        date_time_str = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        return date_time_str

    def convert_soilscape_time_to_datetime(self, time_var):
        time_unit = time_var.getncattr('units')
        time_cal = time_var.getncattr('calendar')
        local_time = nc.num2date(time_var[:], units=time_unit, calendar=time_cal) #cftime._cftime.DatetimeGregorian
        return local_time

    def plot_daily_sm_soilscape(self, dataset, sm_df_daily, station, start_date, end_date):
        ylabel_name = dataset.variables["soil_moisture"].getncattr('long_name') + ' (' + \
                      dataset.variables["soil_moisture"].getncattr('units') + ')'  # Label for y-axis
        series_name = dataset.variables["depth"].getncattr('long_name') + ' (' + \
                      dataset.variables["depth"].getncattr('units') + ')'  # Legend title
        # plot
        sm_df_daily.plot()
        plt.legend(title=series_name)
        plt.ylabel(ylabel_name)
        plt.title("Data from : " + start_date + " to " + end_date)
        plt.savefig("./ss/sm_"+ station + ".png")
        plt.close()

    def daily_files_soilscape(self):
        path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/soilscape/raw/daily/"
        outpath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/soilscape/merged_daily/"

        for f in os.listdir(path):
                dataset = nc.Dataset(path + f)
                soil_moisture = dataset.variables['soil_moisture'][:]
                depths = dataset.variables['depth'][:]
                time_var = dataset.variables['time']
                lat = dataset.variables['lat'][:]
                lon = dataset.variables['lon'][:]


                local_time = self.convert_soilscape_time_to_datetime(time_var)
                for s in range(lat.shape[0]):
                    sm_df = pd.DataFrame(soil_moisture[s], columns=depths, index=local_time.tolist())
                    converted_time = pd.to_datetime([dt.strftime('%Y-%m-%d %H:%M:%S') for dt in local_time])
                    sm_df.index = converted_time
                    filtered_df_2019 = sm_df[sm_df.index.year == 2019]

                    start_date = sm_df.index[0].date().strftime("%Y-%m-%d %H:%M:%S")
                    end_date = sm_df.index[-1].date().strftime("%Y-%m-%d %H:%M:%S")

                    # plot_daily_sm_soilscape(dataset, sm_df_daily, f.split("_")[3] + "_" +
                    #               f.split("_")[4] + "_" + f.split("_")[5] , start_date, end_date)

                    quad = self.quad_obj.get_quad_key(lat[s], lon[s], 8)

                    for index, row in filtered_df_2019.iterrows():
                        date = str(index)[:10]
                        depths = filtered_df_2019.columns
                        for depth in depths:
                            if np.isnan(row[depth]) == True:
                                continue

                            if depth > 25:
                                continue

                            os.makedirs(outpath + str(depth), exist_ok=True)
                            os.makedirs(outpath + str(depth) + "/" + quad, exist_ok=True)
                            file = open(outpath + str(depth) + "/" + quad + "/" + date.strip()  + ".txt", "a")

                            if row[depth] <= 100.0 and row[depth] >= 0.0:
                                file.write(str(lat[s]) + "," + str(lon[s]) + "," + str(row[depth]) + "\n")
                                file.close()

    def split_daily_soil_moisture(self):
        path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/soilscape/raw/new/"
        count = 0
        mystation = set()
        for f in os.listdir(path):
            count+=1
            input_path = path + f
            dataset = nc.Dataset(input_path)
            soil_moisture = dataset.variables['soil_moisture'][:]
            time_var = dataset.variables['time']
            depths = dataset.variables['depth'][:]
            lat = dataset.variables['lat'][:]
            lon = dataset.variables['lon'][:]


            local_time = self.convert_soilscape_time_to_datetime(time_var)
            sm_df = pd.DataFrame(soil_moisture, columns=depths, index=local_time.tolist())
            converted_time = pd.to_datetime([dt.strftime('%Y-%m-%d %H:%M:%S') for dt in local_time])
            sm_df.index = converted_time
            sm_df_daily = sm_df.resample('D').apply(np.nanmean)
            filtered_df_2019 = sm_df_daily[sm_df_daily.index.year == 2019]

            start_date = sm_df_daily.index[0].date().strftime("%Y-%m-%d %H:%M:%S")
            end_date = sm_df_daily.index[-1].date().strftime("%Y-%m-%d %H:%M:%S")

            if end_date[:4] in ["2011", "2012", "2013", "2014", "2015", "2016", "2017",
                                "2018"]:
                os.remove(path+f)
            dataset.close()


            mystation.add((float(lat), float(lon)))
            # self.plot_daily_sm_soilscape(dataset, sm_df_daily, f.split("_")[3] + "_" +
            #               f.split("_")[4] + "_" + f.split("_")[5] , start_date, end_date)

            quad = self.quadHelper.get_quad_key(lat, lon, 14)
            outpath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/"

            for index, row in filtered_df_2019.iterrows():
                date = str(index)[:10].replace("-", "")
                depths = filtered_df_2019.columns
                for depth in depths:
                    if np.isnan(row[depth]) == True:
                       continue

                    if depth > 25:
                        continue

                    if row[depth] <= 100.0 and row[depth] >= 0.0:
                        val = row[depth]
                        if depth == 15:
                            depth = 10

                        os.makedirs(outpath + str(depth), exist_ok=True)
                        os.makedirs(outpath + str(depth) + "/" + quad, exist_ok=True)

                        file = open(outpath + str(depth) + "/" + quad + "/" + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(val/100) + "\n")
                        file.close()
        print("No. of station in soilscape: ", len(mystation))

class ScanSnotelProcessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/snotelScan/raw/"

    def split_daily_soil_moisture(self):
        my_stations = set()
        for state in os.listdir(self.rawInputPath):
            data_frame = pd.read_csv(self.rawInputPath + state, header=None)
            num_columns = data_frame.shape[1]

            for ind, row in data_frame.iterrows():
                date = row[0].replace("-", "")
                if num_columns == 6:
                    lat = float(row[1])
                    lon = float(row[2])
                    soil_moisture_5_cm = float(row[3])
                    soil_moisture_10_cm = float(row[4])
                    soil_moisture_20_cm = float(row[5])

                elif num_columns == 8:
                    lat = float(row[3])
                    lon = float(row[4])

                    soil_moisture_5_cm = float(row[5])
                    soil_moisture_10_cm = float(row[6])
                    soil_moisture_20_cm = float(row[7])

                quad = self.quadHelper.get_quad_key(lat, lon, 14)

                if not math.isnan(soil_moisture_5_cm):
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 101:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm / 100) + "\n")
                        file.close()

                        my_stations.add((lat, lon))

                if not math.isnan(soil_moisture_10_cm):
                    if soil_moisture_10_cm >= 0 and soil_moisture_10_cm <= 101:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/10/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_10_cm / 100) + "\n")
                        file.close()

                if not math.isnan(soil_moisture_20_cm):
                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 101:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm / 100) + "\n")
                        file.close()

        print("No. of stations in ScanSnotel: ", len(my_stations))

class MontanaMesonetProcessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.latlondict = self.get_station_lat_lon()
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/montana_mesonet/raw/"

    def get_station_lat_lon(self):
        mydict = {'wsrabsaw': [-109.61, 45.56],
                  'blm1arge': [-112.88, 45.25],
                  'blmbattl': [-103.78, 45.07],
                  'blmbelfr': [-108.9, 45.18],
                  'mdabench': [-109.97, 47.09],
                  'bentlake': [-111.47, 47.69],
                  'wsrbigtm': [-109.83, 45.74],
                  'blmbroad': [-105.36, 45.44],
                  'wsrbroad': [-108.76, 46.13],
                  'mbmgtech': [-112.53, 46.002],
                  'blmcapit': [-104.14, 45.32],
                  'mdacasca': [-111.62, 47.28],
                  'dmacharl': [-114.17, 46.16],
                  'blmhavre': [-109.05, 48.87],
                  'mdachine': [-109.16, 48.58],
                  'mdachote': [-112.09, 47.89],
                  'churchil': [-111.304, 45.75],
                  'ebarllob': [-113.37, 46.96],
                  'turekran': [-110.13, 47.39],
                  'conradmt': [-111.93, 48.31],
                  'wsrboydw': [-109.26, 45.47],
                  'corvalli': [-114.09, 46.33],
                  'crowagen': [-107.44, 45.59],
                  'mdadagma': [-104.25, 48.66],
                  'mdadillo': [-112.59, 45.25],
                  'mdaedgar': [-108.86, 45.46],
                  'ftbentcb': [-110.48, 47.79],
                  'arskeogh': [-105.95, 46.41],
                  'arskeose': [-105.83, 46.35],
                  'arskeosw': [-105.98, 46.3],
                  'mdafroid': [-104.48, 48.35],
                  'mdagildf': [-110.28, 48.7],
                  'mdaglasw': [-106.72, 48.23],
                  'blmglend': [-104.75, 46.92],
                  'mdahardi': [-107.6, 45.87],
                  'blmhardi' : [-103.73, 45.17],
                  'havrenmt': [-109.8, 48.49],
                  'mdahogel': [-108.56, 48.77],
                  'namlower': [-114.51, 47.7],
                  'namupper': [-114.54, 47.72],
                  'huntleys': [-108.233, 45.919],
                  'sevnoner': [-107.22, 46.66],
                  'kalispel': [-114.14, 48.19],
                  'blm5kidd': [-112.72, 44.81],
                  'mdaledge': [-111.82, 48.34],
                  'lololowr': [-114.13, 46.75],
                  'lomawood': [-110.53, 47.89],
                  'lubrecht': [-113.44, 46.89],
                  'usfsmacp': [-112.31, 46.56],
                  'mdamalta': [-108, 48.35],
                  'mdamanha': [-111.32, 45.79],
                  'blm3mcca': [-112.6, 45.55],
                  'wsrmelvi': [-109.84, 46],
                  'mdamiles': [-105.76, 46.22],
                  'moccasin': [-109.95, 47.06],
                  'moltwest': [-109.09, 45.82],
                  'mdaoilmo': [-111.6, 48.78],
                  'blmplevn': [-104.43, 46.44],
                  'mdapower': [-111.71, 47.66],
                  'blmpumpk': [-105.7, 46.2],
                  'raplejen': [-109.24, 46.04],
                  'reedpoin': [-109.43, 45.8],
                  'wsrreeds': [-109.56, 45.61],
                  'wrsround': [-108.42, 46.58],
                  'blmround': [-108.74, 46.58],
                  'blmroyno': [-108.95, 47.49],
                  'mdascoby': [-105.44, 48.88],
                  'sidneymt': [-104.23, 47.752],
                  'blmterry': [-105.26, 46.84],
                  'mdatwinb': [-112.36, 45.47],
                  'blmglnor': [-106.91, 48.51],
                  'blm2virg': [-111.88, 45.27],
                  'blmwarre': [-108.44, 45.04],
                  'whitshaw': [-114.44, 48.48],
                  'blmmatad': [-108.32, 47.89]
                  }
        return mydict

    def split_daily_soil_moisture(self):
        mystations = set()
        for f in os.listdir(self.rawInputPath):
            data_frame = pd.read_csv(self.rawInputPath + f)
            first_row = data_frame.iloc[0]

            lon = float(self.latlondict[first_row['station_key']][0])
            lat = float(self.latlondict[first_row['station_key']][1])
            quad = self.quadHelper.get_quad_key(lat, lon, 14)

            for index, row in data_frame.iterrows():
                date = str(row['datetime']).replace("-", "")
                if 'Soil VWC @ 0 in. (m³/m³)' in data_frame.columns:
                    soil_moisture_5_cm = float(row['Soil VWC @ 0 in. (m³/m³)'])
                    if math.isnan(soil_moisture_5_cm):
                        continue
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 2:
                        mystations.add((lat, lon))
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm) + "\n")
                        file.close()

                if 'Soil VWC @ 4 in. (m³/m³)' in data_frame.columns:
                    soil_moisture_10_cm = float(row['Soil VWC @ 4 in. (m³/m³)'])
                    if math.isnan(soil_moisture_10_cm):
                        continue
                    if soil_moisture_10_cm >= 0 and soil_moisture_10_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/10/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_10_cm) + "\n")
                        file.close()

                if 'Soil VWC @ 8 in. (m³/m³)' in data_frame.columns:
                    soil_moisture_20_cm = float(row['Soil VWC @ 8 in. (m³/m³)'])
                    if math.isnan(soil_moisture_20_cm):
                        continue
                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm) + "\n")
                        file.close()
        print("No. of stations in Monatana:" , len(mystations))

class NoahHMTProcessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/noaa_hmt/raw/"
        self.out_path_hourly = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/noaa_hmt/merged/"
        self.latlondict = self.get_station_lat_lon()
        self.columnIndex = self.getColumnIndex()
        self.possible_columns = {0: [14, 15, 17],
          1: [None, 10, 11],
          2: [None, 11, 12],
          3: [None, 17, 18],
          4: [22, 23, 25],
          5: [21, 22, 24],
          6: [None, 15, 16],
          7: [13, 14, 16],
          8: [None, 19, 21],
          9: [None, 18, 19],
          10: [None, 14, 15],
          11: [14, 19, None],
          12: [13, 14, 15],
          13: [14, 15, 16]}

    def getColumnIndex(self):
        return {'pvc': 0, 'idp': 1, 'rod': 2, 'stm': 1, 'nvc': 1, 'bbd': 3, 'hbk': 1, 'cmn': 1,
     'wls': 2, 'lgs': 1, 'czc': 4, 'hbg': 2, 'bcc': 5, 'ons': 1, 'hld': 2, 'pvw': 0, 'snf': 1,
     'pck': 1, 'ptv': 2, 'lwd': 1, 'rvw': 0, 'ork': 1, 'lsn': 2 , 'spv': 1,
     'sld': 6 , 'leg': 1, 'wvv': 1, 'bbl': 1,'and': 1, 'gry': 1, 'hdc': 5, 'mgc': 1,
     'cyn': 1, 'skg': 7, 'ata': 1, 'sky': 5, 'str': 8, 'blu': 9,
     'rve': 0, 'mdt': 5, 'wdg': 5, 'frc': 5, 'lve': 10, 'drw': 5, 'bve': 1,
     'cfx': 1, 'rvn': 0, 'hdm': 1, 'pvn': 5, 'ncm': 5, 'boc': 11, 'elg': 12, 'fbk': 13,
                       'fms': 12, 'wse': 7}

    def get_station_lat_lon(self):
        return {
            'pvc': [39.320940, -123.102750],
            'idp': [36.798900, -118.195000],
            'rod': [38.507300, -122.956500],
            'mta': [36.738998, -120.357002],
            'stm': [35.380920, -120.188850],
            'nvc': [39.385300, -120.978200],
            'std': [40.715800, -122.429300],
            'bbd': [39.309000, -120.518],
            'hbk': [41.904300, -122.5693],
            'cmn': [38.735300, -120.6644],
            'wls': [39.346300, -123.3166],
            'lgs': [37.261590, -122.132870],
            'czc': [38.610700, -123.215200],
            'ove': [39.511930, -121.629200],
            'hbg': [38.653000, -122.873200],
            'ctl': [36.297358, -121.714894],
            'bcc': [39.340530, -123.163490],
            'slr': [37.061100, -121.0668],
            'ons': [37.203800, -119.570100],
            'hcp': [41.790600, -123.385400],
            'svc': [37.860001, -122.220001],
            'hld': [39.003000, -123.1209],
            'boc': [31.556800, -110.544200],
            'elg': [31.590700, -110.509200],
            'fbk': [31.721400, -110.188800],
            'fms': [31.565600, -110.546300],
            'wse': [31.685000, -110.281500],
            'pvw': [39.320430, -123.180160],
            'lso': [35.304401, -120.860001],
            'snf': [38.538930, -122.233800],
            'brg': [38.669998, -123.230003],
            'tci': [38.096890, -121.650200],
            'pck': [40.332720, -121.924250],
            'sbo': [34.203200, -117.335300],
            'cfc': [39.079500, -120.937900],
            'ptv': [39.335700, -123.138300],
            'lwd': [35.937080, -121.108030],
            'ner': [37.597200, -120.2775],
            'omm': [37.610001, -119.0],
            'log': [36.302101, -121.051003],
            'acv': [40.972000, -124.11],
            'rvw': [39.301410, -123.260110],
            'ork': [41.223200, -124.054000],
            'lsn': [38.718700, -123.053700],
            'prv': [36.027401, -119.063004],
            'jcb': [32.616501, -116.169998],
            'spv': [36.192200, -118.802400],
            'pts': [36.304175, -121.888054],
            'sld': [36.461040, -121.381170],
            'wcc': [37.799999, -120.639999],
            'leg': [39.876367, -123.719920],
            'wvv': [40.676930, -122.830450],
            'mck': [37.470001, -122.360001],
            'sth': [38.554501, -122.485001],
            'bbl': [39.813100, -122.369300],
            'hmb': [40.876301, -124.074997],
            'and': [38.235000, -120.364000],
            'gry': [37.071800, -121.478800],
            'hdc': [39.267714, -123.147305],
            'mgc': [40.868180, -121.886360],
            'cyn': [37.898640, -121.860020],
            'skg': [39.991000, -105.263600],
            'ata': [39.198300, -120.815500],
            'sba': [34.429470, -119.846820],
            'sky': [39.470969, -121.091673],
            'lcd': [37.099998, -121.650002],
            'ovl': [39.531800, -121.487600],
            'str': [38.515400, -122.802200],
            'sms': [34.263000, -119.096001],
            'blu': [39.275900, -120.709000],
            'rve': [39.314270, -123.186900],
            'mdt': [38.745630, -122.711200],
            'ffm': [38.470001, -121.650002],
            'lqt': [33.575001, -116.226997],
            'dsc': [32.979599, -115.487999],
            'knv': [35.754200, -118.419500],
            'pfd': [36.830100, -119.332400],
            'wdg': [39.234370, -123.004950],
            'vdj': [34.188900, -114.598999],
            'frc': [39.945873, -120.969701],
            'lve': [39.184400, -122.436000],
            'drw': [39.197720, -123.159920],
            'mhl': [38.299999, -122.739998],
            'gmp': [33.051102, -114.827003],
            'bve': [40.473869, -123.792820],
            'cna': [33.858002, -117.609001],
            'cfx': [39.090900, -120.948300],
            'rvn': [39.340650, -123.229730],
            'hdm': [37.796400, -119.858600],
            'pvn': [39.361260, -123.113210],
            'pan': [38.930000, -123.730003],
            'pld': [37.349998, -120.199997],
            'ncm': [39.179600, -123.080000]}

    def julian_day_to_date(self, day):
        date = datetime.strptime(str(day) + ' 2019', '%j %Y')
        formatted_date = date.strftime('%Y%m%d')
        return formatted_date

    def retrive_hourly_sm(self):
        count = 0
        for station in os.listdir(self.rawInputPath):
            count+=1
            newp = self.rawInputPath + station + "/2019/"
            lat, lon = self.latlondict[station]
            daily_averages_5cm, daily_averages_10cm, daily_averages_20cm = [], [], []
            sm_5, sm_10, sm_20 = self.possible_columns[self.columnIndex[station]]
            file_path = self.out_path_hourly + "merged_station.csv"
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file, delimiter=',')

                for day in os.listdir(newp):
                    day_averages_5cm, day_averages_10cm, day_averages_20cm = 0.0, 0.0, 0.0
                    DATE = self.julian_day_to_date(day)
                    hourlies = os.listdir(newp + day)
                    total_hrs = len(hourlies)
                    for hr in hourlies:
                        data_frame = pd.read_csv(newp + day + "/" + hr)

                        if sm_5 is not None:
                            average_moisture_5cm = np.nanmean(data_frame.iloc[:, sm_5])
                            day_averages_5cm += average_moisture_5cm
                        if sm_10 is not None:
                            average_moisture_10cm = np.nanmean(data_frame.iloc[:, sm_10])
                            day_averages_10cm += average_moisture_10cm
                        if sm_20 is not None:
                            average_moisture_20cm = np.nanmean(data_frame.iloc[:, sm_20])
                            day_averages_20cm += average_moisture_20cm

                    if day_averages_5cm != 0.0 or not math.isnan(day_averages_5cm):
                        day_averages_5cm /= total_hrs
                        daily_averages_5cm.append(day_averages_5cm)
                    else:
                        day_averages_5cm = ''

                    if daily_averages_10cm != 0.0 or not math.isnan(day_averages_10cm):
                        day_averages_10cm /= total_hrs
                        daily_averages_10cm.append(day_averages_10cm)
                    else:
                        day_averages_10cm = ''

                    if daily_averages_20cm != 0.0 or not math.isnan(day_averages_20cm):
                        day_averages_20cm /= total_hrs
                        daily_averages_20cm.append(day_averages_20cm)
                    else:
                        day_averages_20cm = ''

                    writer.writerow([DATE, lat, lon, day_averages_5cm, day_averages_10cm, day_averages_20cm])

    def split_daily_soil_moisture(self):
        count = 0
        mysttaions = set()
        with open(self.out_path_hourly + "merged_station.csv", "r") as file:
            for line in file:
                splited = line.strip().split(",")
                count +=1
                date = splited[0]
                lon = float(splited[2])
                lat = float(splited[1])

                soil_moisture_5_cm = float(splited[3])
                soil_moisture_10_cm = float(splited[4])
                soil_moisture_20_cm = float(splited[5])

                quad = self.quadHelper.get_quad_key(lat, lon, 14)

                if math.isnan(soil_moisture_5_cm) or soil_moisture_5_cm == 0.0:
                   pass
                else:
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 150.0:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm/100.0) + "\n")
                        file.close()
                        mysttaions.add((lat, lon))

                if math.isnan(soil_moisture_10_cm) or soil_moisture_10_cm == 0.0:
                   pass
                else:
                    if soil_moisture_10_cm >= 0 and soil_moisture_10_cm <= 150.0:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/10/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_10_cm/100.0) + "\n")
                        file.close()

                if math.isnan(soil_moisture_20_cm) or soil_moisture_20_cm == 0.0:
                    pass
                else:
                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 150.0:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm/100.0) + "\n")
                        file.close()
            print("No. of stations in NoahHMT:", len(mysttaions))

class TexasMesonetProcessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/texas_mesonet/raw/"

    def split_daily_soil_moisture(self):
        self.split_daily_soil_moisture_st(network='texas_mesonet')
        time.sleep(5)
        self.split_daily_soil_moisture_st(network='eea')
        time.sleep(5)
        self.split_daily_soil_moisture_st(network='twdb')

    def split_daily_soil_moisture_st(self, network = 'eea'):
        my_stat = set()
        for station in os.listdir(self.rawInputPath + network):
            df = pd.read_csv(self.rawInputPath + network + "/" + station)
            lat = float(df[' Latitude'][0])
            lon = float(df[' Longitude'][0])
            df[' Date_Time (UTC)'] = pd.to_datetime(df[' Date_Time (UTC)'])
            df.set_index(' Date_Time (UTC)', inplace=True)
            daily_avg = df.resample('D').apply(np.nanmean)
            new_columns = {column: column.strip() for column in daily_avg.columns}
            daily_avg.rename(columns=new_columns, inplace=True)
            comns = daily_avg.columns
            quad = self.quadHelper.get_quad_key(lat, lon, 14)

            for index, row in daily_avg.iterrows():
                date = row.name.strftime('%Y-%m-%d').split("T")[0].replace("-","")
                soil_moisture_5_cm, soil_moisture_10_cm, soil_moisture_20_cm = math.nan, math.nan, math.nan

                if 'Soil Volumetric Water Content 5 cm (%)' in comns:
                    soil_moisture_5_cm = row['Soil Volumetric Water Content 5 cm (%)']
                    if math.isnan(soil_moisture_5_cm):
                        continue
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 150:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm/100.0) + "\n")
                        file.close()
                        my_stat.add((lat, lon))

                if 'Soil Volumetric Water Content 10 cm (%)' in comns:
                    soil_moisture_10_cm = row['Soil Volumetric Water Content 10 cm (%)']
                    if math.isnan(soil_moisture_10_cm):
                        continue
                    if soil_moisture_10_cm >= 0 and soil_moisture_10_cm <= 150:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/10/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_10_cm/100.0) + "\n")
                        file.close()

                if 'Soil Volumetric Water Content 20 cm (%)' in comns:
                    soil_moisture_20_cm = row['Soil Volumetric Water Content 20 cm (%)']
                    if math.isnan(soil_moisture_20_cm):
                        continue
                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 150:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm/100.0) + "\n")
                        file.close()

                # print(lat, lon, date, soil_moisture_5_cm, soil_moisture_10_cm, soil_moisture_20_cm)
        print("No. of stations in texas: ", network, len(my_stat))

def remove_duplicate_rows(level=5):
    file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/" + str(level)+ "/"
    out_file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/split/" + str(level) + "/"

    for quad in os.listdir(file_path):
        os.makedirs(out_file_path + quad, exist_ok=True)
        for d in os.listdir(file_path + quad):
            unique_rows = set()
            input_file = os.path.join(file_path, quad, d)
            with open(input_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    lat, lon, moisture = row[0], row[1], row[2]
                    unique_rows.add((lat, lon, moisture))

            with open(out_file_path + quad + "/" + d.split(".txt")[0].strip() + '.txt', 'w', newline='') as file:
                writer = csv.writer(file)
                for row in unique_rows:
                    writer.writerow(row)

class Okhlahoma_preprocessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj

    def split_daily_soil_moisture(self):
        input_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/okhla_mesonet/"
        mystations = set()
        with open(input_path + 'data.csv', "r") as file:
            for line in file:

                    splited = line.strip().split(",")
                    data = [x for x in splited if x != '']
                    date = data[1].split("T")[0].replace("-", "")

                    lon = float(data[3])
                    lat = float(data[2])

                    mystations.add((lat, lon))

                    soil_moisture_5_cm = float(data[4])
                    soil_moisture_20_cm = float(data[5])

                    quad = self.quadHelper.get_quad_key(lat, lon, 14)

                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip() + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm) + "\n")
                        file.close()

                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm) + "\n")
                        file.close()
        print("No. of stations in Okhlahoma: " , len(mystations))

class Missouri_preprocessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj

    def split_daily_soil_moisture(self):
        input_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/missouri_agr/raw/"

        mystations = set()

        with open(input_path + 'missouri_data.csv', "r") as file:
            for line in file:
                    splited = line.strip().split(",")
                    data = [x for x in splited if x != '']
                    date = data[5]+data[3]+data[4]

                    lon = float(data[2])
                    lat = float(data[1])

                    mystations.add((lat, lon))
                    soil_moisture_5_cm = float(data[6])

                    quad = self.quadHelper.get_quad_key(lat, lon, 14)

                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 110:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm/100.0) + "\n")
                        file.close()
        print("No. of stations in Missouri: ", len(mystations))

class NycProcessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/nyc_mesonet/raw/"

    # 1                                    time
    # 2                latitude [degrees_north]
    # 3                longitude [degrees_east]
    # 10       soil_moisture_05cm_avg [m^3/m^3]
    # 13       soil_moisture_25cm_avg [m^3/m^3]


    def split_daily_soil_moisture(self):
        mystations = set()
        for f in os.listdir(self.rawInputPath):
            data_frame = pd.read_csv(self.rawInputPath + f)
            first_row = data_frame.iloc[1]

            lon = float(first_row['longitude [degrees_east]'])
            lat = float(first_row['latitude [degrees_north]'])
            quad = self.quadHelper.get_quad_key(lat, lon, 14)

            for index, row in data_frame.iterrows():
                if index == 0:
                    continue
                if 'EST' in str(row['time']):
                    date = str(row['time']).split("EST")[0].replace("-", "")
                if 'EDT' in str(row['time']):
                    date = str(row['time']).split("EDT")[0].replace("-", "")

                if 'soil_moisture_05cm_avg [m^3/m^3]' in data_frame.columns:
                    soil_moisture_5_cm = float(row['soil_moisture_05cm_avg [m^3/m^3]'])
                    if math.isnan(soil_moisture_5_cm):
                        continue
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm) + "\n")
                        file.close()
                        mystations.add((lat, lon))

                if 'soil_moisture_25cm_avg [m^3/m^3]' in data_frame.columns:
                    soil_moisture_20_cm = float(row['soil_moisture_25cm_avg [m^3/m^3]'])
                    if math.isnan(soil_moisture_20_cm):
                        continue
                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm) + "\n")
                        file.close()
        print("No, of stations in NYC: ", len(mystations))

class IllinoisPreprocessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/illinois_climate/"
        self.latlondict = self.get_station_lat_lon()

    def get_station_lat_lon(self):
        data =  [["bbc", "Big Bend", 41.63, -90.04],
                       ["brw", "Brownstown", 38.95, -88.96],
                       ["bvl", "Bondville", 40.05, -88.37],
                        ["cmi", "Champaign", 40.08, -88.24],
                        ["dek", "DeKalb", 41.84, -88.85],
                        ["dxs", "Dixon Springs", 37.44, -88.67],
                        ["fai", "Fairfield", 38.38, -88.39],
                        ["fre", "Freeport", 42.28, -89.67],
                        ["frm", "Belleville", 38.52, -89.84],
                        ["icc", "Peoria", 40.71, -89.51],
                        ["llc", "Springfield", 39.73, -89.61],
                        ["mon", "Monmouth", 40.93, -90.72],
                        ["oln", "Olney", 38.74, -88.10],
                        ["orr", "Perry", 39.81, -90.82],
                        ["rnd", "Rend Lake", 38.14, -88.92],
                        ["siu", "Carbondale", 37.70, -89.24],
                        ["sni", "Snicarte", 40.11, -90.18],
                        ["stc", "St Charles", 41.90, -88.36],
                        ["ste", "Stelle", 40.95, -88.16]]

        return {station[0]: [station[2], station[3]] for station in data}

    def calculate_daily_averages(self, data):
        daily_averages = {}
        for entry in data:
            station = entry['station']
            date = entry['timestamp'].date()
            moisture_values = entry['moisture_values']

            if (station, date) not in daily_averages:
                daily_averages[(station, date)] = [moisture_values, 1]
            else:
                daily_averages[(station, date)][0] = [sum(x) for x in
                                                      zip(daily_averages[(station, date)][0], moisture_values)]
                daily_averages[(station, date)][1] += 1

        for key, value in daily_averages.items():
            average_values = [round(val / value[1], 4) for val in value[0]]
            daily_averages[key] = average_values

        return daily_averages

    def merge_hourlies(self):
        input_file = self.rawInputPath + 'raw/illionois.csv'
        data = []
        with open(input_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                station = row[0].strip()
                timestamp = datetime.strptime(row[1].strip(), '%m/%d/%y %H:%M')
                moisture_values = [float(val.strip()) for val in row[2:]]
                data.append({'station': station, 'timestamp': timestamp, 'moisture_values': moisture_values})

        daily_averages = self.calculate_daily_averages(data)

        output_file = self.rawInputPath + 'merged/merged.csv'
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for (station, date), values in daily_averages.items():
                writer.writerow([station, self.latlondict[station][0] ,self.latlondict[station][1], date.strftime('%Y%m%d')] + values)


    def split_daily_soil_moisture(self):
        count = 0
        mystation = set()
        with open(self.rawInputPath + "merged/merged.csv", "r") as file:
            for line in file:
                splited = line.strip().split(",")
                count +=1

                date = splited[3]
                lon = float(splited[2])
                lat = float(splited[1])

                soil_moisture_5_cm = float(splited[4])
                soil_moisture_10_cm = float(splited[5])
                soil_moisture_20_cm = float(splited[6])

                quad = self.quadHelper.get_quad_key(lat, lon, 14)

                if math.isnan(soil_moisture_5_cm) or soil_moisture_5_cm == 9999.0:
                   pass
                else:
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm) + "\n")
                        file.close()
                        mystation.add((lat, lon))

                if math.isnan(soil_moisture_10_cm) or soil_moisture_10_cm == 9999.0:
                   pass
                else:
                    if soil_moisture_10_cm >= 0 and soil_moisture_10_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/10/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_10_cm) + "\n")
                        file.close()

                if math.isnan(soil_moisture_20_cm) or soil_moisture_20_cm == 9999.0:
                    pass
                else:
                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm) + "\n")
                        file.close()
        print("No. of stations in Illinois: ", len(mystation))

class DelawarePreprocessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/delware_environment/"
        self.latlondict = self.get_station_lat_lon()

    def get_station_lat_lon(self):
        station_dict = {}
        with open(self.rawInputPath + 'soil_metadata_dalaware.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                station_name = row[0].lower()
                lat = float(row[1])
                lon = float(row[2])
                station_dict[station_name] = [lat, lon]
        return station_dict

    def merge_hourlies(self):
        for station in os.listdir(self.rawInputPath + "raw/"):
            print("Station: ", station)
            input_file = self.rawInputPath + 'raw/' + station
            daily_averages = {}
            daily_counts = {}
            st_name = station.split('.')[0].lower()
            lat, lon = self.latlondict[st_name]
            with open(input_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    timestamp_str, sm, _ = row
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    date = timestamp.date()
                    sm = float(sm)
                    daily_averages[date] = daily_averages.get(date, 0) + sm
                    daily_counts[date] = daily_counts.get(date, 0) + 1

            for date in daily_averages:
                daily_averages[date] /= daily_counts[date]

            output_file = self.rawInputPath + 'merged/merged.csv'
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for date, average in daily_averages.items():
                    writer.writerow([lat, lon, date.strftime('%Y%m%d'), average])

    def split_daily_soil_moisture(self):
        count = 0
        mystation = set()
        with open(self.rawInputPath + "merged/merged.csv", "r") as file:
            for line in file:
                splited = line.strip().split(",")
                count +=1

                date = splited[2]
                lon = float(splited[1])
                lat = float(splited[0])

                soil_moisture_5_cm = float(splited[3])

                quad = self.quadHelper.get_quad_key(lat, lon, 14)

                if math.isnan(soil_moisture_5_cm) or soil_moisture_5_cm == 9999.0:
                   pass
                else:
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm) + "\n")
                        file.close()
                        mystation.add((lat, lon))

        print("No. of stations in Delaware: ", len(mystation))

class NebraskaPreprocessing:
    def __init__(self, quad_obj):
        self.quadHelper = quad_obj
        self.rawInputPath = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/nebraska_mesonet/"
        self.latlondict = self.get_station_lat_lon()

    def get_station_lat_lon(self):
        station_dict = {'Alliance': [42.1843056, -102.9221667],
                        'Arthur': [41.452662, -101.713468],
                        'Central City': [41.112545,-98.04952],
                        'Decatur': [41.906727, -96.263428],
                        'Dickens': [40.83324,-100.978402],
                        'Hayes Center': [40.55357,-101.03472],
                        'Indianola': [40.134659, -100.478704],
                        'Leigh': [41.712838, -97.263538],
                        'Lexington': [40.72501, -99.751222],
                        'Lincoln': [40.8300556, -96.6569444],
                        'Memphis': [41.14565, -96.440526],
                        'North Platte': [41.085025, -100.774853],
                        'Oshkosh': [41.492588, -102.345758],
                        'Pierce': [42.173854, -97.545847],
                        'Plattsmouth': [40.977161, -95.880736],
                        'Rulo': [40.021621, -95.504868],
                        'Whitman': [42.08154, -101.44906],
                        'Winslow': [41.629981, -96.382077]}

        return station_dict
        # with open(self.rawInputPath + 'soil_metadata_dalaware.csv', newline='') as csvfile:
        #     reader = csv.reader(csvfile)
        #     for row in reader:
        #         station_name = row[0].lower()
        #         lat = float(row[1])
        #         lon = float(row[2])
        #         station_dict[station_name] = [lat, lon]
        # return station_dict

    def merge_hourlies(self):
        for station in os.listdir(self.rawInputPath + "raw/"):
            input_file = self.rawInputPath + 'raw/' + station
            daily_averages = {}
            daily_counts = {}
            st_name = station.split('-')[0]
            if st_name in self.latlondict.keys():
                lat, lon = self.latlondict[st_name]
            else:
                continue
            print("Station: ", station)
            with open(input_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if row[0][:4] != '2019':
                        continue
                    timestamp_str, mV, _, _ = row
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    date = timestamp.date()
                    mV = float(mV)
                    sm = (mV * 0.05) - 5
                    daily_averages[date] = daily_averages.get(date, 0) + sm
                    daily_counts[date] = daily_counts.get(date, 0) + 1

            for date in daily_averages:
                daily_averages[date] /= daily_counts[date]

            output_file = self.rawInputPath + 'merged/merged.csv'
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for date, average in daily_averages.items():
                    if average <= 150 and average >= 0.0:
                        writer.writerow([lat, lon, date.strftime('%Y%m%d'), average/100.0])

    def split_daily_soil_moisture(self):
        count = 0
        mystation = set()
        with open(self.rawInputPath + "merged/merged.csv", "r") as file:
            for line in file:
                splited = line.strip().split(",")
                count +=1

                date = splited[2]
                lon = float(splited[1])
                lat = float(splited[0])

                soil_moisture_5_cm = float(splited[3])

                quad = self.quadHelper.get_quad_key(lat, lon, 14)
                if math.isnan(soil_moisture_5_cm) or soil_moisture_5_cm == 9999.0:
                   pass
                else:
                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 2:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip()  + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm) + "\n")
                        file.close()
                        mystation.add((lat, lon))

        print("No. of stations in Nebaraska: ", len(mystation))


def create_dataset_for_merged():
    ss_obj = TexasMesonetProcessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = DelawarePreprocessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = NebraskaPreprocessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = Okhlahoma_preprocessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = Missouri_preprocessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = CRNProcessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = SoilScapeProcessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = ScanSnotelProcessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = MontanaMesonetProcessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = NoahHMTProcessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = NycProcessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ss_obj = IllinoisPreprocessing(quad_obj)
    ss_obj.split_daily_soil_moisture()

    ob = Kentucky_preprocessing(quad_obj)
    ob.split_daily_soil_moisture()

    # New Jersey

class Kentucky_preprocessing:
        def __init__(self, quad_obj):
            self.quadHelper = quad_obj

        def getmonth(self, number):
                return str(number).zfill(2)

        def split_daily_soil_moisture(self):
            input_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/kentucky/"

            mystations = set()

            with open(input_path + 'kentucky.csv', "r") as file:
                for line in file:
                    splited = line.strip().split(",")
                    data = [x for x in splited if x != '']
                    date = '2019' + self.getmonth(data[3]) + self.getmonth(data[4])

                    lon = float(data[2])
                    lat = float(data[1])

                    mystations.add((lat, lon))
                    soil_moisture_5_cm = float(data[5])
                    soil_moisture_10_cm = float(data[6])
                    soil_moisture_20_cm = float(data[7])

                    quad = self.quadHelper.get_quad_key(lat, lon, 14)

                    if soil_moisture_5_cm >= 0 and soil_moisture_5_cm <= 110:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/5/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip() + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_5_cm / 100.0) + "\n")
                        file.close()

                    if soil_moisture_10_cm >= 0 and soil_moisture_10_cm <= 110:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/10/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip() + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_10_cm / 100.0) + "\n")
                        file.close()

                    if soil_moisture_20_cm >= 0 and soil_moisture_20_cm <= 110:
                        file_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/station_data/merged/20/" + quad + "/"
                        os.makedirs(file_path, exist_ok=True)
                        file = open(file_path + date.strip() + ".txt", "a")
                        file.write(str(lat) + "," + str(lon) + "," + str(soil_moisture_20_cm / 100.0) + "\n")
                        file.close()

            print("No. of stations in Kentucky: ", len(mystations))


if __name__ == '__main__':

    quad_obj = QuadhashHelper()
    create_dataset_for_merged()

    # remove_duplicate_rows(level=5)
    # remove_duplicate_rows(level=10)
    # remove_duplicate_rows(level=20)


