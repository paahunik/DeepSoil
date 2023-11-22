import rasterio
from dem_stitcher.stitcher import stitch_dem
import numpy as np
import socket
import os
import geopandas as gpd

mins = 100000
maxs = -9999

def download_dem(out_path, poly):
    # bounds = [-106.0016667, 36.9983333, -104.9983333, 38.0016667]
    global mins
    global maxs

    X, p = stitch_dem(list(poly.bounds),
                      dem_name='glo_30',
                      dst_ellipsoidal_height=False,
                      dst_area_or_point='Area')
    if np.min(X) < mins:
        mins = np.min(X)
    if np.max(X) > maxs:
        maxs = np.max(X)

    if os.path.exists(out_path):
        return

    save_as_tif(X, p, out_path)

def save_as_tif(X, p, out_path):
    with rasterio.open(out_path, 'w', **p) as ds:
        ds.write(X, 1)
        ds.update_tags(AREA_OR_POINT='Area')

def chop_in_quadhash():
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/dem/split_14/"

    root_path = "/s/chopin/a/grad/paahuni/cl_3.8/deepSoil/dataset_preprocessing/"
    quadhash_dir = next(
        d for d in os.listdir(root_path) if os.path.isdir(root_path + d) and d.startswith("quadshape_14_"))

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, 'quadhash.shp'))

    total = len(quadhashes)
    count = 0

    for ind, row in quadhashes.iterrows():
        poly, qua = row["geometry"], row["Quadkey"]
        os.makedirs(out_path + qua, exist_ok=True)
        count += 1
        print("Dem Processing: ", count, "/", total, qua, flush=True)
        download_dem(out_path + qua + "/final_elevation_30m.tif", poly)

    print("-------  Min dem value in state: ", mins, "------ max: ", maxs)

def remove_empty_folders():
    path_dir = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/dem/split_14/"
    tot = len(os.listdir(path_dir))
    count = 0
    for q in os.listdir(path_dir):
        if len(os.listdir(path_dir + q)) == 0:
            count += 1
            os.rmdir(path_dir + q)
    print("No data in: ", count, "/", tot, "quadhashes")

if __name__ == '__main__':
    chop_in_quadhash()
    remove_empty_folders()