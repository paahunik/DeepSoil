import gdal
import numpy as np
import os
import geopandas as gpd
import socket

def chop_in_quadhash():
    root_path = "/s/chopin/a/grad/paahuni/cl_3.8/deepSoil/dataset_preprocessing/"
    quadhash_dir = next(d for d in os.listdir(root_path) if os.path.isdir(root_path + d) and d.startswith("quadshape_14_"))

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, 'quadhash.shp'))

    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/nlcd/raw/1.tif"
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/nlcd/split_14/"
    count = 0
    total = len(quadhashes)

    for ind, row in quadhashes.iterrows():
        poly, qua = row["geometry"], row["Quadkey"]

        count += 1
        print("Splitting: ", count, "/", total)
        if os.path.exists(out_path + qua + '/nlcd.tif'):
            continue

        os.makedirs(out_path + qua, exist_ok=True)

        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

        gdal.Translate(out_path + qua + '/nlcd.tif', in_path, projWin=window)
        if not os.path.exists(out_path + qua + "/nlcd.tif"):
            continue
        x = gdal.Open(out_path + qua + '/nlcd.tif').ReadAsArray()
        if np.min(x) == np.max(x) == 0:
            os.remove(out_path + qua + '/nlcd.tif')

def remove_empty_folders():
    path_dir = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/nlcd/split_14/"
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
    remove_empty_folders()
