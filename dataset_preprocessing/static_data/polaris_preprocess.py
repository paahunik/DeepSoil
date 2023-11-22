import csv
import socket
import subprocess
import os
from gdalconst import GA_ReadOnly
import gdal
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import box

def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)

def download_tif_for_param(param, files, out_path):
    d = '0_5'
    out_path = out_path + param + "/"
    for f in files:
        if os.path.exists(out_path + param + "/" + f):
            continue
        print("Downloading param: ", param, f)
        download_link = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/" + param + "/mean/" + d + "/" + f
        runcmd("wget -P " + out_path + " " + download_link, verbose=False)

def load_csv_of_tif_names():
    list_of_files = []
    with open('./polaris_tif_names.csv', newline='\n') as csvfile:
        data = list(csv.reader(csvfile))
    for i in data:
        list_of_files.append(i[0])
    return list_of_files

def get_state_name():
    root_path = "/s/chopin/a/grad/paahuni/cl_3.8/deepSoil/dataset_preprocessing/"
    quadhash_dir = next(
        d for d in os.listdir(root_path) if os.path.isdir(root_path + d) and d.startswith("quadshape_14_"))

    state = quadhash_dir.split("_14")[1].replace("_", " ").strip()
    states = gpd.read_file('../cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    states = states.to_crs("EPSG:4326")
    state_info = states[states['NAME'] == state]["geometry"].iloc[0]
    return state_info

def download_for_each_param():
    state_info = get_state_name()
    files = load_csv_of_tif_names()
    parameter_name = ['silt', 'sand', 'clay', 'bd', 'theta_s', 'theta_r', 'ksat', 'ph', 'om', 'n']
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/polaris/raw/"

    filtered_files = []
    for f in files:
        min_lat = float(f.split('_')[0][3:5]) - 0.1
        max_lat = float(f.split('_')[0][5:]) + 0.1
        min_lon = -1 * (float(f.split('_lon')[1].split("-")[1]) + 0.2)
        max_lon = -1 * (float(f.split('_lon')[1].split("-")[2].split(".tif")[0]) - 0.2)

        smaller_region_box = box(min_lon, min_lat, max_lon, max_lat)

        if state_info.intersects(smaller_region_box) or state_info.contains(smaller_region_box):
            filtered_files.append(f)

    print("downloading: ", len(filtered_files))

    for p in parameter_name:
        os.makedirs(out_path + p, exist_ok=True)
        download_tif_for_param(p, filtered_files, out_path)

def merge_geotiffs_for_each_band():
        input_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/polaris/raw/"
        output_path_merged = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/polaris/merged/"

        os.makedirs(output_path_merged, exist_ok=True)
        parameter_name = ['silt', 'sand', 'clay', 'bd', 'theta_s', 'theta_r', 'ksat', 'ph', 'om', 'n']
        no_data_value = -9999
        count = 0

        total = len(os.listdir(input_path + "silt"))
        for f in os.listdir(input_path + "silt"):
            output_file = output_path_merged + f
            if os.path.exists(output_file):
                continue

            count += 1
            print("Processing: ", count, "/", total)
            input_files = []
            for i in parameter_name:
                input_files.append(input_path + i + "/" + f)

            if len(input_files) != 10:
                print("Skipping tiles, as band missing: ", f)

            command = ['gdal_merge.py', '-separate', '-o', output_file, '-a_nodata', str(no_data_value)] + input_files
            subprocess.call(command)

            for infile in input_files:
                os.remove(infile)

def chop_in_quadhash():
    depths = '0_5'
    root_path = "/s/chopin/a/grad/paahuni/cl_3.8/deepSoil/dataset_preprocessing/"
    quadhash_dir = next(
        d for d in os.listdir(root_path) if os.path.isdir(root_path + d) and d.startswith("quadshape_14_"))

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, 'quadhash.shp'))

    in_path_dir = "/s/" + socket.gethostname()+ "/b/nobackup/galileo/sm_predictions/input_datasets/polaris/merged/"
    out_path = "/s/" + socket.gethostname()+ "/b/nobackup/galileo/sm_predictions/input_datasets/polaris/split_14/"

    count = 0
    total = len(os.listdir(in_path_dir))

    for f in os.listdir(in_path_dir):
        count += 1
        print("processing:  ", count, "/", total, flush=True)
        for ind, row in quadhashes.iterrows():
                poly, qua = row["geometry"], row["Quadkey"]

                new_path = depths + "_depth_" + f

                if os.path.exists(out_path + qua + '/' + new_path):
                    continue

                os.makedirs(out_path + qua, exist_ok=True)
                bounds = list(poly.exterior.coords)
                window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

                gdal.Translate(out_path + qua + '/' + new_path,  in_path_dir + f,
                               projWin=window)
                x = gdal.Open(out_path + qua + '/' + new_path).ReadAsArray()
                if os.path.exists(out_path + qua + '/' + new_path):
                    if np.min(x[0]) == np.max(x[0]) == -9999.0:
                        os.remove(out_path + qua + '/' + new_path)

def merge_geotiffs_for_quadhash():
    input_path = "/s/" + socket.gethostname()+ "/b/nobackup/galileo/sm_predictions/input_datasets/polaris/split_14/"

    depths = '0_5'
    no_data_value = -9999

    for qua in os.listdir(input_path):
        input_files = []
        for f in os.listdir(input_path+qua):
            if depths in f[:5]:
                input_files.append(input_path + qua + "/" + f)

        output_file = input_path + qua + "/" + depths + "_merged" + ".tif"
        if len(input_files) == 1:
            command = ['mv', input_files[0], output_file]
            subprocess.call(command)
            continue

        print("\nMerging for quad:", qua, " files: ", len(input_files), flush=True)
        command = ['gdal_merge.py', '-init', str(no_data_value) , '-n', str(no_data_value), '-o', output_file] + input_files
        subprocess.call(command)

        for each_inp_file in input_files:
            os.remove(each_inp_file)

    remove_empty_folders()

def remove_empty_folders():
    path_dir = "/s/" + socket.gethostname()+ "/b/nobackup/galileo/sm_predictions/input_datasets/polaris/split_14/"
    tot = len(os.listdir(path_dir))
    count = 0
    for q in os.listdir(path_dir):
        if len(os.listdir(path_dir + q)) == 0:
            print("No files in :", q)
            count += 1
            os.rmdir(path_dir + q)
    print(count, "/", tot)

if __name__ == '__main__':
    download_for_each_param()
    merge_geotiffs_for_each_band()
    chop_in_quadhash()
    merge_geotiffs_for_quadhash()
