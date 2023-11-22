import gdal
import numpy as np
import mercantile,fiona
import rasterio as rio
from rasterio import mask as msk
import random
import geopy.distance
import os, osr
import geopandas as gpd
import shutil
import pickle
import subprocess
import socket
import json

def chop_in_quadhash():
    input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/koppen/raw/koppen_map.tif"
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/koppen/split_14/"

    root_path = "/s/chopin/a/grad/paahuni/cl_3.8/deepSoil/dataset_preprocessing/"
    quadhash_dir = next(
        d for d in os.listdir(root_path) if os.path.isdir(root_path + d) and d.startswith("quadshape_14_"))

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, 'quadhash.shp'))

    for index, row in quadhashes.iterrows():
        poly, quadf = row['geometry'], row['Quadkey']

        if os.path.exists(out_path + quadf + "/1km.tif"):
            continue

        os.makedirs(out_path + quadf, exist_ok=True)
        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])
        gdal.Translate(out_path + quadf + "/1km.tif", input_path, projWin = window)
        a = gdal.Open(out_path + quadf + "/1km.tif").ReadAsArray()
        print(index, "/", len(quadhashes))
        if np.min(a) == np.max(a) == 0.0:
            os.remove(out_path + quadf + "/1km.tif")
    remove_empty_folders()

def remove_empty_folders():
    path_dir = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/koppen/split_14/"
    tot = len(os.listdir(path_dir))
    count = 0
    for q in os.listdir(path_dir):
        if len(os.listdir(path_dir + q)) == 0:
            print("No files in :", q)
            count += 1
            os.rmdir(path_dir + q)
    print(count, "/", tot)

if __name__ == '__main__':
    chop_in_quadhash()
