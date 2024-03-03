import socket
import gdal
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import random
import geopandas as gpd
from skimage import exposure
import json
import matplotlib
# from pykrige.ok import OrdinaryKriging

class Dataloader:
    def __init__(self, height=64, width=64, season=[3], batch_size=32, isTrainingSet=True):
        self.train=[]
        self.test=[]
        self.height = height
        self.width = width
        self.input_conditions_size = 6

        self.input_quadhashes_14_path = "/s/chopin/a/grad/paahuni/cl_3.8/soil_moisture_project/filter_quads_se.txt"
        self.all_quadhased_14 = self.load_list()

        self.polaris_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/polaris/merged_14/"
        self.nlcd_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/nlcd/merged_14/"
        self.landsat8_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/landsat8L2/merged_14/"
        self.gNATSGO_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/gNATSGO/merged_14/"
        self.dem_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/dem/merged_14/"
        self.koppen_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/koppen_climate/merged_14/"
        self.hru_corrected_path_model_1 = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/merged_corrected_14/"
        self.hru_sm_path_model_1 = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/merged_14/"
        self.hru_sm_path_model_1_monthly = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/raw/"
        self.hru_corrected_path_model_1_monthly = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/corrected/"

        self.dic = self.load_dic()

        self.smap9km_path_split = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/smap_9km/split_14/"
        self.gridmet_path_split = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/gridmet/split_14/"
        self.lai_path_split = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/MCD15A3H.061/split_14/"
        self.daymet_path_split = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/daymet/split_14/"
        self.landsat8_path_split = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/landsat8L2/split_14/"
        self.dem_path_split = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/dem/split_14/"
        self.nlcd_path_split = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/nlcd/split_14/"
        self.hru_corrected_path_model_2 = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_corrected_14/"
        self.hru_sm_path_model_2 = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_14/"
        self.model1_prediction_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/1/tifs/"

    def scale_image_polaris(self, data_array):
        data_array[data_array == -9999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)

        masked_data_array[:, :, 0:3] = masked_data_array[:, :, 0:3] / 100.0
        masked_data_array[:, :, 3:5] = masked_data_array[:, :, 3:5] / 2.0
        masked_data_array[:, :, 5] = (masked_data_array[:, :, 5] + 2) / 4.0
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_image_gnatsgo_30m(self, data_array):
        data_array[data_array == -9999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)

        masked_data_array[:, :, 0] = masked_data_array[:, :, 0] / 270.0
        masked_data_array[:, :, 1] = masked_data_array[:, :, 1] / 50.0
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_image_gnatsgo_90m(self, data_array):
        data_array[data_array == -9999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
        min_val, max_val = 200, 800
        masked_data_array[:, :, 0] = (masked_data_array[:, :, 0] - min_val)/ (max_val-min_val)
        min_val, max_val = 0, 300
        masked_data_array[:, :, 1] = (masked_data_array[:, :, 1] - min_val)/ (max_val-min_val)
        min_val, max_val = 0, 500
        masked_data_array[:, :, 2] = (masked_data_array[:, :, 2] - min_val) / (max_val - min_val)
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_nlcd(self, data_array):
        x_min = 11.0
        x_max = 95.0
        data_array = data_array.astype(np.float64)

        data_array[data_array == 0.0] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
        masked_data_array = (masked_data_array - x_min) / (x_max - x_min)
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_gridmet(self, data):
        data[data == 32767.0] = np.nan
        new_data = []
        mins = [0,     0,         0, 1.05, 0,  0,     225.54, 233.08, 0,   0]
        maxes= [27.02, 17.27, 690.44,100,  100,455.61,314.88, 327.14, 9.83,131.85]

        for i in range(data.shape[-1]):
            new_data.append((data[:,:,i] - mins[0])/(maxes[i] - mins[i]))
        data = np.nan_to_num(data, nan=-1)
        return np.array(new_data).reshape((data.shape))

    def scale_koppen(self, data_array):
        x_min = 1
        x_max = 30.0
        data_array = data_array.astype(np.float64)

        data_array[data_array == 0.0] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
        masked_data_array = (masked_data_array - x_min) / (x_max - x_min)
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)
        return masked_data_array

    def scale_dem(self, data_array):
        x_max = 6500

        data_array[data_array == -999999] = np.nan
        masked_data_array = np.ma.masked_invalid(data_array)
        masked_data_array = masked_data_array / x_max
        masked_data_array = np.nan_to_num(masked_data_array, nan=-1)

        return masked_data_array

    def scale_hru(self, data_array):
        data_array[data_array == -9999] = np.nan
        data_array[data_array < 0] = 0
        data_array[data_array > 1] = 1
        data_array = np.nan_to_num(data_array, nan=-1)
        return data_array

    def scale_image_smap(self, data_array):
        data_array[data_array == -9999] = np.nan
        data_array[data_array<0] = 0
        data_array[data_array>1] = 1
        data_array = np.nan_to_num(data_array, nan=-1)
        return data_array

    def scale_image_landsat_30m(self, img):
        def normalize(band):
            return ((band - 1) / (60000.0 - 1))

        def scale_image_landsat_30m(imgs):
            imgs = imgs.astype(np.float32)
            imgs[imgs == 0] = np.nan
            imgs = normalize(imgs)
            imgs[imgs < 0] = 0
            imgs = np.nan_to_num(imgs, nan=-1)
            return imgs

        img = exposure.rescale_intensity(img,
                                             out_range=(1, 50000)).astype(np.int32)

        img = img.astype(np.float32)
        img[img == 0] = np.nan
        img = scale_image_landsat_30m(img)
        # img[img == -1] = np.nan
        # ndvi = np.where((img[:, :, -1] + img[:, :, 0]) == 0., 0,
        #                     (img[:, :, -1] - img[:, :, 0]) / (img[:, :, -1] + img[:, :, 0]))

        return img[:,:,:3]

    def scale_image_lai(self, data_array):
        data_array  = data_array.astype(np.float64)
        data_array[data_array<0] = np.nan
        data_array[data_array>100] = np.nan
        data_array = data_array * 0.1
        data_array = data_array/10
        data_array = np.nan_to_num(data_array, nan=-1)
        return data_array

    def resize_image(self, data, height=64, width=64, isNLCD=False):
        resized_images = []

        for i in range(data.shape[-1]):
            current_slice = data[:, :, i]
            if isNLCD:
                resized_slice = cv2.resize(current_slice, (height, width), interpolation=cv2.INTER_NEAREST)
            else:
                resized_slice = cv2.resize(current_slice, (height, width), interpolation=cv2.INTER_AREA)

            resized_images.append(resized_slice)

        resized_array = np.array(resized_images)
        resized_array = np.reshape(np.array(resized_array), (height, width, data.shape[-1]))

        return resized_array

    def load_model_1_datasets_30m(self, quad):
        if quad is None or len(quad)!=14:
            return None

        # out_path = "/s/lattice-180/b/nobackup/galileo/paahuni/half_conus/model_1_pred/2/" + quad + "/30m.tif"
        #
        # if os.path.exists(out_path):
        #     return None

        if True:
            if os.path.exists(self.polaris_path + quad + "/0_5_merged.tif"):
                image_polaris = gdal.Open(self.polaris_path + quad + "/0_5_merged.tif").ReadAsArray()[[0,1,2,3,5,6], :, :]
                image_polaris = self.scale_image_polaris(self.resize_image(np.transpose(image_polaris, (1, 2, 0))))
            else:
                return None
        if True:
            if os.path.exists(self.nlcd_path + quad + "/nlcd.tif"):
                image_nlcd = gdal.Open(self.nlcd_path + quad + "/nlcd.tif").ReadAsArray()
                image_nlcd = image_nlcd.reshape(1, image_nlcd.shape[0], image_nlcd.shape[1])
                image_nlcd = self.scale_nlcd(self.resize_image(np.transpose(image_nlcd, (1, 2, 0)), isNLCD=True))
            else:
                return None

        if True:
            if os.path.exists(self.gNATSGO_path + quad + "/30m.tif"):
                image_gnatsgo_30m = gdal.Open(self.gNATSGO_path + quad + "/30m.tif").ReadAsArray()
                if image_gnatsgo_30m.shape[0] == 9:
                    image_gnatsgo_30m = image_gnatsgo_30m[[0,3], :, :]
                image_gnatsgo_30m = self.scale_image_gnatsgo_30m(self.resize_image(np.transpose(image_gnatsgo_30m, (1, 2, 0))))
            else:
                return None

        if True:
            if os.path.exists(self.landsat8_path + quad + "/landsat_final_30m.tif"):
                image_landsat_30m = gdal.Open(self.landsat8_path + quad + "/landsat_final_30m.tif").ReadAsArray()
                image_landsat_30m = np.reshape(self.scale_image_landsat_30m(self.resize_image(np.transpose(image_landsat_30m, (1, 2, 0)))), (64,64,3))
            else:
                return None

        if True:
            if os.path.exists(self.gNATSGO_path + quad + "/90m.tif"):
                image_gnatsgo_90m = gdal.Open(self.gNATSGO_path + quad + "/90m.tif").ReadAsArray()
                image_gnatsgo_90m = self.scale_image_gnatsgo_90m(self.resize_image(np.transpose(image_gnatsgo_90m[:3, :, :], (1, 2, 0))))
            else:
                return None

        if True:
            if os.path.exists(self.dem_path + quad + "/final_elevation_30m.tif"):
                image_dem = gdal.Open(self.dem_path + quad + "/final_elevation_30m.tif").ReadAsArray()
                image_dem = image_dem.reshape(1, image_dem.shape[0], image_dem.shape[1])
                image_dem = self.scale_dem(self.resize_image(np.transpose(image_dem, (1, 2, 0))))
            else:
                return None

        if True:
            if os.path.exists(self.koppen_path + quad + "/1km.tif"):
                image_koppen = gdal.Open(self.koppen_path + quad + "/1km.tif").ReadAsArray()
                image_koppen = image_koppen.reshape(1, image_koppen.shape[0], image_koppen.shape[1])
                image_koppen = self.scale_koppen(self.get_max_occuring_val_in_array(np.transpose(image_koppen, (1, 2, 0))))
                image_koppen = np.full((64,64,1), image_koppen[0], dtype=np.uint8)
            else:
                return None
        # merged_image = np.concatenate((image_polaris, image_nlcd, image_gnatsgo_30m, image_dem, image_landsat_30m), axis=2)

        merged_image = np.concatenate((image_polaris, image_gnatsgo_30m, image_gnatsgo_90m, image_koppen, image_dem, image_nlcd, image_landsat_30m), axis=2)
        return merged_image

    def get_max_occuring_val_in_array(self, arr):
        values, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        most_frequent = values[max_count_index]
        return np.array([most_frequent])

    def load_dic(self):
        with open("./filter_quads_and_neighbors.json", "r") as file:
            loaded_dict = json.load(file)
        return loaded_dict

    def find_keys_by_value(self, quad):
        keys_with_value = []
        for key, value_list in self.dic.items():
            if quad in value_list or quad == key:
                keys_with_value.append(key)
        return keys_with_value

    def load_model_2_datasets(self, quad, date, load_daymet=True):
        if quad is None or len(quad) != 14:
            return None

        if os.path.exists(self.smap9km_path_split + quad[:12] + "/" + date):
                img_smap = gdal.Open(self.smap9km_path_split + quad[:12] + "/" + date).ReadAsArray()
                img_smap = img_smap.reshape(1, img_smap.shape[0], img_smap.shape[1])
                img_smap = self.scale_image_smap(self.resize_image(np.transpose(img_smap, (1, 2, 0))))
        else:
                return None

        if os.path.exists(self.gridmet_path_split + quad[:12] + "/" + date):
                img_grid = gdal.Open(self.gridmet_path_split + quad[:12] + "/" + date).ReadAsArray()[[0,3,4,5,6,8,10,11,12], :,:]
                img_grid = self.scale_gridmet(self.resize_image(np.transpose(img_grid, (1, 2, 0))))
        else:
                return None

        if os.path.exists(self.lai_path_split + quad + "/" + date):
                img_lai = gdal.Open(self.lai_path_split + quad + "/" + date).ReadAsArray()
                img_lai = img_lai.reshape(1, img_lai.shape[0], img_lai.shape[1])
                img_lai = self.scale_image_lai(self.resize_image(np.transpose(img_lai, (1, 2, 0))))
        else:
                return None

        if os.path.exists(self.nlcd_path_split + quad + "/nlcd.tif"):
                image_nlcd = gdal.Open(self.nlcd_path_split + quad + "/nlcd.tif").ReadAsArray()
                image_nlcd = image_nlcd.reshape(1, image_nlcd.shape[0], image_nlcd.shape[1])
                image_nlcd = self.scale_nlcd(self.resize_image(np.transpose(image_nlcd, (1, 2, 0)), isNLCD=True))
        else:
                return None

        if os.path.exists(self.dem_path_split + quad + "/final_elevation_30m.tif"):
            image_dem = gdal.Open(self.dem_path_split + quad + "/final_elevation_30m.tif").ReadAsArray()
            image_dem = image_dem.reshape(1, image_dem.shape[0], image_dem.shape[1])
            image_dem = self.scale_dem(self.resize_image(np.transpose(image_dem, (1, 2, 0))))
        else:
            return None

        # if load_daymet:
        #     if os.path.exists(self.daymet_path_split + quad + "/" + date + ".tif"):
        #         img_daymet = gdal.Open(self.daymet_path_split + quad + "/" + date + ".tif").ReadAsArray()
        #         img_daymet = img_daymet.reshape(1, img_daymet.shape[0], img_daymet.shape[1])
        #         img_daymet = self.scale_image_daymet(self.resize_image(np.transpose(img_daymet, (1, 2, 0))))
        #     else:
        #         return None

        if os.path.exists(self.landsat8_path_split + quad + "/landsat_final_30m.tif"):
            image_landsat_30m = gdal.Open(self.landsat8_path_split + quad + "/landsat_final_30m.tif").ReadAsArray()
            image_landsat_30m = np.reshape(self.scale_image_landsat_30m(self.resize_image(np.transpose(image_landsat_30m, (1, 2, 0)))), (64,64,3))
        else:
            return None

        quadn = self.find_keys_by_value(quad)[0]

        season = date[4:6]
        if os.path.exists(self.model1_prediction_path + quadn + "_" + season + ".npy"):
            sm_predicted_model_1 = np.load(self.model1_prediction_path + quadn + "_" + season + ".npy")
            sm_predicted_model_1 = sm_predicted_model_1.reshape((64, 64, 1))
        else:
            return None

        merged_image = np.concatenate((img_smap, img_grid, img_lai, sm_predicted_model_1, image_dem, image_nlcd, image_landsat_30m), axis=2)
        return merged_image

    def perform_inferences_stations(folder, training=False, one_per_quad=False, iscorrected=False):
        model2 = load_model_weights(folder)
        model2 = model2.to(device).float()
        model2 = model2.eval()
        if iscorrected:
            inp_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_corrected_14/"
        else:
            inp_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/split_14/"

        output_geotiff_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/" + str(
            folder) + "/tiffs/"
        os.makedirs(output_geotiff_path, exist_ok=True)

        batch_size = 32
        test_dataset = QuadhashDataset_model_2(training=training, one_per_quad=one_per_quad, corrected_hru=iscorrected,
                                               cluster_no=11)
        dataloader_new = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        all_err, all_stations, all_out_vals = [], [], []

        with torch.no_grad():
            for (input_sample, target_img, allquads, dates) in dataloader_new:
                input_sample = input_sample.to(torch.float32).to(device)
                output_sample = model2(input_sample).cpu().numpy()

                for i in range(output_sample.shape[0]):
                    d = dates[i]
                    q = allquads[i]
                    lat_s, lon_s, sm_s = load_station(q, d)
                    if sm_s is not None:
                        if os.path.exists(inp_path + q + "/" + d):
                            sm_hru = gdal.Open(inp_path + q + "/" + d)

                            if sm_hru is None:
                                continue
                            geotransform = sm_hru.GetGeoTransform()
                            projection = sm_hru.GetProjection()
                            resized_out_image = cv2.resize(output_sample[i][0],
                                                           (sm_hru.RasterXSize, sm_hru.RasterYSize))
                            driver = gdal.GetDriverByName('GTiff')
                            b_dataset = driver.Create(output_geotiff_path + q + "_" + d, sm_hru.RasterXSize,
                                                      sm_hru.RasterYSize, 1,
                                                      gdal.GDT_Float32)  # Change data type as needed

                            b_dataset.SetProjection(projection)
                            b_dataset.SetGeoTransform(geotransform)
                            band = b_dataset.GetRasterBand(1)
                            band.WriteArray(resized_out_image)
                            b_dataset = None
                            resized_out_image[resized_out_image == -1] = np.nan
                            errs, station_vals, out_vals = get_closest_station_window(lon_s, lat_s, sm_s, sm_hru,
                                                                                      resized_out_image)

                            all_err.extend(errs)
                            all_stations.extend(station_vals)
                            all_out_vals.extend(out_vals)
                            sm_hru = None

                        else:
                            continue

        print("len: ", len(all_stations), len(all_out_vals))
        print("Average error SM: ", np.mean(np.array(all_err)), " average std: ", np.std(np.array(all_err)),
              " on sample count: ", len(all_err))

    def load_target_data_model_2(self, quad, no_of_samples_per_quad=None, corrected_hru=True, training=True):
        dataset_hru = []
        dates_r = []
        if corrected_hru:
            dates = os.listdir(self.hru_corrected_path_model_2 + quad)
        else:
            dates = os.listdir(self.hru_sm_path_model_2 + quad)

        random.seed(42)
        random.shuffle(dates)

        # if len(dates) > 70:
        #     dates = dates[:70]

        if no_of_samples_per_quad is not None:
                filtered_dates = [date for date in dates if self.checkseason(date)]
                random_dates = random.sample(filtered_dates, no_of_samples_per_quad)
                dates_r = random_dates
        else:
            for d in dates:
                if len(d) > 13:
                    continue
                if training:
                    if d[4:6] in ['04', '06', '07', '09']:
                            dates_r.append(d)
                            if corrected_hru:
                                hru_file = gdal.Open(self.hru_corrected_path_model_2 + quad + "/" + d).ReadAsArray()
                            else:
                                hru_file = gdal.Open(self.hru_sm_path_model_2 + quad + "/" + d).ReadAsArray()

                            hru_file = hru_file.reshape(1, hru_file.shape[0], hru_file.shape[1])
                            hru_file = self.scale_hru(self.resize_image(np.transpose(hru_file, (1, 2, 0))))
                            dataset_hru.append(hru_file)
                else:
                        if d[4:6] in ['05', '10']:
                            dates_r.append(d)
                            if corrected_hru:
                                hru_file = gdal.Open(self.hru_corrected_path_model_2 + quad + "/" + d).ReadAsArray()
                            else:
                                hru_file = gdal.Open(self.hru_sm_path_model_2 + quad + "/" + d).ReadAsArray()

                            hru_file = hru_file.reshape(1, hru_file.shape[0], hru_file.shape[1])
                            hru_file = self.scale_hru(self.resize_image(np.transpose(hru_file, (1, 2, 0))))
                            dataset_hru.append(hru_file)
        return np.array(dataset_hru), dates_r

    def perform_hru_monthly_average(self, data):
        data[data == -1] = np.nan
        average_array = np.nanmean(data, axis=0)
        average_array = np.nan_to_num(average_array, nan=-1)
        return average_array

    def clip_image(self, data):
        return np.clip(data, 0, 1)

    def check_month(self, date, month):
        if date[4:6] == month:
            return True

    def load_target_data_model_1(self, quad, no_of_samples_per_quad=None, training=True, corrected_hru=True, one_per_quad=False, saved=True):
        if corrected_hru:
            os.makedirs("/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/corrected/" + quad, exist_ok=True)
            out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/corrected/" + quad + "/"
        else:
            os.makedirs("/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/raw/" + quad, exist_ok=True)
            out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/raw/" + quad + "/"

        dataset_hru,dates_r = [],[]
        if saved and no_of_samples_per_quad == 12:
            for m in os.listdir(out_path):
                if os.path.exists("/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/corrected/" + quad + "/" + m):
                    dates_r.append(m.split(".")[0])
                    dataset_hru.append(np.load(out_path + m))
                else:
                    return None, None
            return np.array(dataset_hru), dates_r

        if corrected_hru:
            dates = os.listdir(self.hru_corrected_path_model_1 + quad)
        else:
            dates = os.listdir(self.hru_sm_path_model_1 + quad)

        if no_of_samples_per_quad == 1:
                filtered_dates = [date for date in dates if self.checkseason(date)]
                random_dates = random.sample(filtered_dates, 1)
                dates_r = random_dates

        elif no_of_samples_per_quad == 12:
            dates_r = []
            if training:
                look_for_months = ['03', '04', '05', '06', '07', '08', '09']
            else:
                look_for_months = ['04', '05', '06']

            for month in look_for_months:
                    filtered_dates = [date for date in dates if self.check_month(date, month)]

                    monthly_data = []
                    for d in filtered_dates:
                        if corrected_hru:
                            hru_file = gdal.Open(self.hru_corrected_path_model_1 + quad + "/" + d).ReadAsArray()
                        else:
                            hru_file = gdal.Open(self.hru_sm_path_model_1 + quad + "/" + d).ReadAsArray()

                        hru_file = hru_file.reshape(1, hru_file.shape[0], hru_file.shape[1])
                        hru_file = self.scale_hru(self.resize_image(np.transpose(hru_file, (1, 2, 0))))
                        monthly_data.append(hru_file)

                    if len(monthly_data) == 0:
                        continue

                    # dates_r.append(month)
                    #
                    # dataset_hru.append()
                    if not saved:
                        np.save(out_path + month + '.npy', np.round(self.perform_hru_monthly_average(np.array(monthly_data)), 4))

                    if one_per_quad and month in ['04', '05', '06', '07', '08','09']:
                        return np.array(dataset_hru), dates_r
        else:
            for d in dates:
                if self.checkseason(d):
                    if corrected_hru:
                        hru_file = gdal.Open(self.hru_corrected_path_model_1 + quad + "/" + d).ReadAsArray()
                    else:
                        hru_file = gdal.Open(self.hru_sm_path_model_1 + quad + "/" + d).ReadAsArray()

                    hru_file = hru_file.reshape(1, hru_file.shape[0], hru_file.shape[1])
                    hru_file = self.scale_hru(self.resize_image(np.transpose(hru_file, (1, 2, 0))))
                    dataset_hru.append(hru_file)
            dates_r = dates
        return np.array(dataset_hru), dates_r

    def load_list(self):
        loaded_list = []
        with open(self.input_quadhashes_14_path, "r") as file:
            for line in file:
                loaded_list.append(str(line.strip()))
        return loaded_list

    def generate_model_1_train_test_data(self,data_quad_list,
                                         no_of_samples_per_quad=None, corrected_hru=False, one_per_quad=False,
                                         training = True):

        os.makedirs("/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/",
                    exist_ok=True)
        os.makedirs("/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/corrected/",
                    exist_ok=True)
        os.makedirs("/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru_monthly_av/raw/",
                    exist_ok=True)

        all_imgs_30m, all_target_sm, all_quads, all_months = [], [], [], []
        count = 0
        for quad in data_quad_list:
            count += 1
            img_30m = self.load_model_1_datasets_30m(quad)
            if img_30m is None:
                continue

            target_sm, months = self.load_target_data_model_1(quad, no_of_samples_per_quad=no_of_samples_per_quad,training=training,
                                                                    corrected_hru=corrected_hru, one_per_quad=one_per_quad)
            if target_sm is None:
                continue

            for d in range(target_sm.shape[0]):
                all_imgs_30m.append(img_30m)
                all_target_sm.append(target_sm[d])
                all_quads.append(quad)
                all_months.append(months[d])

        all_imgs_30m = np.array(all_imgs_30m)
        all_target_sm = np.array(all_target_sm)
        all_quads = np.array(all_quads)
        all_months = np.array(all_months)
        return all_imgs_30m, all_target_sm, all_quads, all_months

    def generate_model_1_only_infer_data(self,data_quad_list, no_of_samples_per_quad=None, corrected_hru=False):
        quadhashes = gpd.read_file('/s/chopin/a/grad/paahuni/cl_3.8/soil_moisture_project/states_shapefile_okhla_filtered/quadhash.shp')
        needed = quadhashes['Quadkey'].tolist()

        all_imgs_30m, all_imgs_koppen, all_imgs_90m, all_quads = [], [], [], []
        count = 0
        for quad in needed:
            count += 1

            if os.path.exists("/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/model_1_pred/2/" + quad + "/30m.tif"):
                continue

            img_30m = self.load_model_1_datasets_30m(quad)
            if img_30m is None:
                continue

            all_imgs_30m.append(img_30m)
            all_quads.append(quad)

        all_imgs_30m = np.array(all_imgs_30m)
        all_quads = np.array(all_quads)

        return all_imgs_30m, all_quads

    def checkseason(self, date_str):
        # Return True for spring or summer
        out = self.get_season(date_str)
        if out == 1 or out == 2 or out == 3:
            return True
        else:
            return False

    def get_season(self, date_str):
        month = int(date_str[4:6])
        day = int(date_str[6:8])

        if (month == 3 and day >= 20) or (month > 3 and month < 6) or (month == 6 and day <= 20):
            return 1  # Spring
        elif (month == 6 and day >= 21) or (month > 6 and month < 9) or (month == 9 and day <= 21):
            return 2  # "Summer"
        elif (month == 9 and day >= 22) or (month > 9 and month < 12) or (month == 12 and day <= 20):
            return 3  # "Autumn/Fall"
        else:
            return 4  # "Winter"

    def generate_model_2_train_test_data(self, data_quad_list, no_of_samples_per_quad=None, corrected_hru=False,
                                         training = True, one_per_quad=False):
        all_imgs_30m, all_target_sm, all_quads, all_dates = [], [], [], []
        count = 0

        for quad in data_quad_list:
            count += 1

            target_sm, dates_av_hru = self.load_target_data_model_2(quad, no_of_samples_per_quad=no_of_samples_per_quad,
                                                         corrected_hru=corrected_hru, training=training)
            if len(dates_av_hru) == 0:
                continue

            for i in range(len(dates_av_hru)):
                img_30m = self.load_model_2_datasets(quad, dates_av_hru[i])
                if img_30m is None:
                    continue

                if target_sm[i] is None:
                    continue

                all_imgs_30m.append(img_30m)
                all_target_sm.append(target_sm[i])
                all_quads.append(quad)
                all_dates.append(dates_av_hru[i])

                if one_per_quad:
                    break

        all_imgs_30m = np.array(all_imgs_30m)
        all_target_sm = np.array(all_target_sm)
        all_quads = np.array(all_quads)
        all_dates = np.array(all_dates)

        return all_imgs_30m, all_target_sm, all_quads, all_dates

# def load_cluster_by_no():

class QuadhashDataset(Dataset):
    def __init__(self, training=True, one_per_quad=False, corrected_hru=True, cluster_no = 1):
        self.data_loader = Dataloader()

        # random_seed = 42
        # random.seed(random_seed)
        if corrected_hru:
            self.quadhashes = os.listdir(self.data_loader.hru_corrected_path_model_1_monthly)
        else:
            self.quadhashes = os.listdir(self.data_loader.hru_sm_path_model_1_monthly)

        self.training = training
        random.shuffle(self.quadhashes)
        split_index = int(0.2 * len(self.quadhashes))

        if self.training is True:
            self.data_quad_list = self.quadhashes
        else:
            self.data_quad_list = self.quadhashes[:split_index]

        self.all_input, self.all_target, self.all_quads, self.months = self.data_loader.generate_model_1_train_test_data(
            self.data_quad_list, no_of_samples_per_quad=12, corrected_hru=corrected_hru, one_per_quad=False, training = training)

        print("No. of samples returning: ", len(self.all_quads))

    def __len__(self):
        return len(self.all_quads)

    def __getitem__(self, index):

        all_input_1 = self.all_input[index]
        all_target = self.all_target[index]
        all_quads = self.all_quads[index]
        all_months = self.months[index]

        all_input_1 = torch.tensor(all_input_1).permute(2, 0, 1)
        all_target = torch.tensor(all_target).permute(2, 0, 1)
        return all_input_1, all_target, all_quads, all_months


class QuadhashDataset_model_2(Dataset):
    def load_cluster(self):
        data = {}
        cluster_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/static_properties/mycluster_fin.txt"
        with open(cluster_path, 'r') as file:
            for line in file:
                hash_value, cluster = line.strip().split(',')
                cluster = int(cluster)
                if cluster in data.keys():
                    prev = data.get(cluster)
                    prev.append(hash_value)
                    data[cluster] = prev
                else:
                    data[cluster] = [hash_value]
        return data

    def __init__(self,  cluster_no=12, training=True, one_per_quad=False, corrected_hru=False):
        self.data_loader = Dataloader()
        cluster_assgn = self.load_cluster()

        paren_q = cluster_assgn[cluster_no]

        random_seed = 42
        random.seed(random_seed)
        if corrected_hru:
            mypath = self.data_loader.hru_corrected_path_model_2
        else:
            mypath = self.data_loader.hru_sm_path_model_2

        self.quadhashes = []
        for q in paren_q:
            if os.path.exists(mypath + q):
                self.quadhashes.append(q)
            for chd in self.data_loader.dic.get(q):
                if os.path.exists(mypath + chd):
                    self.quadhashes.append(chd)

        print("cluster_no: ", cluster_no, len(self.quadhashes))
        # self.quadhashes = os.listdir(mypath)
        random.shuffle(self.quadhashes)
        if training is False:
            self.quadhashes = self.quadhashes[:300]
        #     print("Found testing parents: ", len(paren_q), "and total quad: ", len(self.quadhashes))
        # else:
        #     self.quadhashes = self.quadhashes[:500]
        #     print("Found training parents: ", len(paren_q), "and total quad: ", len(self.quadhashes))

        self.training = training

        self.all_input, self.all_target, self.all_quads, self.all_dates = self.data_loader.generate_model_2_train_test_data(
            self.quadhashes, no_of_samples_per_quad=None, corrected_hru=corrected_hru, training=training,
        one_per_quad=one_per_quad)

        num_samples = self.all_target.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        print("Found: ", len(self.all_quads))
        selected_indices = indices[:13000]

        if len(self.all_quads) > 13000:
            self.all_input = self.all_input[selected_indices, :, :, :]
            self.all_target = self.all_target[selected_indices,: ,:, :]
            self.all_quads = self.all_quads[selected_indices]
            self.all_dates = self.all_dates[selected_indices]

        # print("No. of samples out of: ", len(all_quads), all_input.shape, all_target.shape)
        print("Returning: ", len(self.all_quads), self.all_input.shape, self.all_target.shape)

    def __len__(self):
        return len(self.all_quads)

    def __getitem__(self, index):
        all_input_1 = self.all_input[index]
        all_target = self.all_target[index]
        all_quads = self.all_quads[index]
        all_input_1 = torch.tensor(all_input_1).permute(2, 0, 1)
        all_target = torch.tensor(all_target).permute(2, 0, 1)
        all_dates = self.all_dates[index]
        return all_input_1, all_target, all_quads, all_dates

class QuadhashDataset_ONLY_INFER(Dataset):
    def __init__(self, training=False):
        self.data_loader = Dataloader()
        file_path = './okhla_quad_needed.txt'

        with open(file_path, 'r') as file:
            self.data_quad_list = [line.strip() for line in file]
        print("Returning :", len(self.data_quad_list))
        self.training = training

        self.all_input, self.all_quads = self.data_loader.generate_model_1_only_infer_data(
            self.data_quad_list, no_of_samples_per_quad=None, corrected_hru=False)
        print("No. of samples returning: ", len(self.all_quads))

    def __len__(self):
        return len(self.all_quads)

    def __getitem__(self, index):

        # all_input_1 = self.all_input[0][index]
        # all_input_2 = self.all_input[1][index]
        # all_input_3 =self.all_input[2][index]
        all_input_1 = self.all_input[index]
        all_quads = self.all_quads[index]

        # all_input_1 = torch.tensor(all_input_1)
        # all_input_2 = torch.tensor(all_input_2)
        # all_input_3 = torch.tensor(all_input_3)
        all_input_1 = torch.tensor(all_input_1).permute(2, 0, 1)

        # return [all_input_1, all_input_2, all_input_3], all_target, all_quads
        return all_input_1, all_quads

def resize_image(data):
    resized_images = []

    for i in range(data.shape[-1]):
        current_slice = data[:, :, i]
        resized_slice = cv2.resize(current_slice, (64, 64), interpolation=cv2.INTER_AREA)
        resized_images.append(resized_slice)

    resized_array = np.reshape(np.array(resized_images), (64,64, data.shape[-1]))
    print(resized_array.shape)
    return resized_array

def plot_target():
    count = 0
    inpu_p = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/hru/merged_corrected_14/"
    for q in os.listdir(inpu_p):
        for f in os.listdir(inpu_p + q):

            count +=1
            im = gdal.Open(inpu_p + q + "/" + f).ReadAsArray()


            im = np.array(im)
            im = im.reshape(im.shape[0], im.shape[1], 1)
            im = resize_image(im)

            im[im<0] = np.nan


            plt.imshow(im[:,:,0])
            # plt.imshow(im)

            plt.savefig("./plots/" + str(count) + "_" + q + ".png")
            plt.close()
            break

def load_cluster():
        data = {}
        cluster_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/static_properties/mycluster_fin.txt"
        with open(cluster_path, 'r') as file:
            for line in file:
                hash_value, cluster = line.strip().split(',')
                cluster = int(cluster)
                if cluster in data.keys():
                    prev = data.get(cluster)
                    prev.append(hash_value)
                    data[cluster] = prev
                else:
                    data[cluster] = [hash_value]
        return data

def plot_no_of_clusters():
    cluster_assgn = load_cluster()
    # CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
    #                   '#f781bf', '#a65628', '#984ea3',
    #                   '#999999', '#e41a1c', '#dede00',
    #                   '#377eb8', '#ff7f00', '#4daf4a',
    #                   '#f781bf', '#a65628', '#984ea3',
    #                   '#999999', '#e41a1c', '#dede00',
    #                   '#377eb8', '#ff7f00', '#4daf4a',
    #                   '#f781bf', '#a65628', '#984ea3',
    #                   '#999999', '#e41a1c', '#dede00'
    #                   ]

    sorted_clusters = dict(sorted(cluster_assgn.items()))
    # num_colors_needed = len(sorted_clusters)
    # chosen_colors = random.sample(CB_color_cycle, num_colors_needed)

    clusters = list(sorted_clusters.keys())
    sample_counts = [len(samples) for samples in sorted_clusters.values()]
    # plt.grid(True)
    colormap = matplotlib.cm.tab20b.colors
    num_colors_to_select = 19
    chosen_colors = random.sample(colormap, num_colors_to_select)

    bars = plt.bar(clusters, sample_counts, color=chosen_colors, alpha=0.9)
    plt.xlabel('Cluster Number')
    plt.ylabel('Number of Samples')
    plt.title('Number of Samples per Cluster')
    plt.xticks(clusters)
    i = 0
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color='black',
            # weight='bold'
        )
        i+=1

    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)

    plt.savefig("./clustering_samples_per_cluster.png")
    plt.close()

def plot_kmeans():
    def perform_kringing(longs, lats, cluster_no,
                         gridx, gridy):
        longs = np.array(longs)
        lats = np.array(lats)
        cluster_no = np.array(cluster_no)
        try:
            OK = OrdinaryKriging(longs, lats, cluster_no, variogram_model='gaussian',
                                 verbose=False, enable_plotting=False)
            zstar, _ = OK.execute("grid", xpoints=gridx, ypoints=gridy)
        except Exception as e:
            print("\nAn error occurred during kriging:", str(e))
            return None
        return zstar

    quad_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/koppen_climate/merged_14/"
    cluster_assgn = load_cluster()
    cluster_no, lats, longs = [], [], []
    for k in cluster_assgn.keys():
        for q in cluster_assgn[k]:
            newp = quad_path + q
            sample = os.listdir(newp)[0]
            raster = gdal.Open(newp + "/" + sample)
            geotransform = raster.GetGeoTransform()

            center_x = geotransform[0] + geotransform[1] * raster.RasterXSize / 2
            center_y = geotransform[3] + geotransform[5] * raster.RasterYSize / 2
            longs.append(center_x)
            lats.append(center_y)
            cluster_no.append(k)

    indices = np.arange(len(cluster_no))
    np.random.shuffle(indices)
    plt.figure(dpi=300)  # Set the DPI value to 300 (adjust as needed)

    gridx = np.arange(-125.9051, -66.9727, 0.055666666666667, dtype="float64")
    gridy = np.arange(24.6071, 48.9780, 0.055666666666667, dtype="float64")
    new_clusters = perform_kringing(longs, lats, cluster_no, gridx, gridy)
    print(new_clusters.shape)
    print(np.min(new_clusters), np.max(new_clusters))
    print(min(cluster_no), max(cluster_no))
    cax2 = plt.imshow(new_clusters, extent=(gridx[0], gridx[-1], gridy[0],gridy[-1]), origin='lower', cmap='winter')
    plt.scatter(longs, lats, c='k', marker='.', s=1)
    # plt.colorbar(cax2)
    plt.xticks([])  # Hide x ticks
    plt.yticks([])  # Hide y ticks

    plt.title("Hydrologically Similar Clusters")
    plt.savefig("./6_Estimated_Clustering_Impact" + ".png",  dpi=300)
    plt.close()

def get_no_of_missing_pixels_and_water():
    non_ava_pixels = 0
    iswater = 0
    total_pixels = 0
    input_path_nlcd = "/s/" + socket.gethostname() + "/b/nobackup/galileo/paahuni/half_conus/nlcd/split_14/"
    for i in range(0, 19):
        train_dataset = QuadhashDataset_model_2(training=True, corrected_hru=False, cluster_no=i, one_per_quad=True)
        train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for _, test_target_sm, test_quads,_ in train_dataset:
            for s in range(test_target_sm.shape[0]):
                non_ava_pixels += np.count_nonzero(test_target_sm[s] == -1)
                # nlc = gdal.Open(input_path_nlcd + test_quads[s] + "/nlcd.tif").ReadAsArray()
                # values_to_count = [11, 12, 90, 95]
                # iswater += np.count_nonzero(np.isin(nlc, values_to_count))
                # print(test_target_sm[s].shape, non_ava_pixels)
                total_pixels+= test_target_sm[s].shape[0]*test_target_sm[s].shape[1]*test_target_sm[s].shape[2]
    print("Found total missing pixels in HydroBlocks:", non_ava_pixels, "/", total_pixels)

if __name__ == '__main__':
    batch_size = 1
    # plot_kmeans()
    # get_no_of_missing_pixels_and_water()


    # train_dataset = QuadhashDataset(training=True, corrected_hru=True)
    # train_dataset = QuadhashDataset_model_2(training=False, corrected_hru=False, cluster_no=17)
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # train_features, train_target_sm = next(iter(train_dataloader))
    # for test_features, test_quads in test_dataloader:
    #     continue
    # test_features, test_target_sm, test_quads = next(iter(train_dataloader))
    # print(test_features.shape, test_target_sm.shape, len(test_quads))



    # print(cluster_assgn)



