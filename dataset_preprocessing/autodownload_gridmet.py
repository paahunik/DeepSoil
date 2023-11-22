import subprocess
import socket
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
import datetime

def download_nc_file():
    year = datetime.datetime.now().year
    print(subprocess.run(["./download_daily_gridmet.sh " + str(year)], shell=True))
    print(subprocess.run(["mv *_{}.nc /s/{}/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/raw/".format(year, socket.gethostname())], shell=True))

def day_of_year_to_date(day_of_year, year):
     date = datetime.date.fromordinal(datetime.date(year, 1, 1).toordinal() + day_of_year - 1)
     return date.strftime('%Y%m%d')

def convert_to_tif():
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/raw/"
    for f in os.listdir(in_path):
        if not f.endswith('.nc'):
            continue

        ds = gdal.Open(in_path + f)
        num_bands = ds.RasterCount
        out_tif = in_path + f.split(".nc")[0] + '.tif'
        last_time_slice_band = ds.GetRasterBand(num_bands)

        gtiff_driver = gdal.GetDriverByName("GTiff")
        output_ds = gtiff_driver.Create(out_tif, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float64)
        output_band = output_ds.GetRasterBand(1)
        band = last_time_slice_band.ReadAsArray()
        band = band.astype(float)
        output_band.WriteArray(band)
        output_ds.SetProjection(ds.GetProjectionRef())
        output_ds.SetGeoTransform(ds.GetGeoTransform())
        output_band = None
        output_ds = None
        last_time_slice_band = None
        ds = None

        with open(in_path + "last_updated.txt", 'w') as file:
            file.write(str(num_bands))

        print("Converted ", f)
        os.remove(in_path + f)

def merge_bands_gridmet():
    year = str(datetime.datetime.now().year)
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/raw/"
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/merged/"

    etr = gdal.Open(in_path + 'etr_' + year + ".tif").ReadAsArray()
    etr[etr != 32767.0] = etr[etr != 32767.0] * 0.1

    pet = gdal.Open(in_path + 'pet_' + year + ".tif").ReadAsArray()
    pet[pet != 32767.0] = pet[pet != 32767.0] * 0.1

    pr = gdal.Open(in_path + 'pr_' + year + ".tif").ReadAsArray()
    pr[pr != 32767.0] *= 0.1

    rmax = gdal.Open(in_path + 'rmax_' + year + ".tif").ReadAsArray()
    rmax[rmax != 32767.0] *= 0.1

    rmin = gdal.Open(in_path + 'rmin_' + year + ".tif").ReadAsArray()
    rmin[rmin != 32767.0] *= 0.1

    srad = gdal.Open(in_path + 'srad_' + year + ".tif").ReadAsArray()
    srad[srad != 32767.0] *= 0.1

    tmmn = gdal.Open(in_path + 'tmmn_' + year + ".tif").ReadAsArray()
    tmmn[tmmn != 32767.0] = tmmn[tmmn != 32767.0] * 0.1 + 210

    tmmx = gdal.Open(in_path + 'tmmx_' + year + ".tif").ReadAsArray()
    tmmx[tmmx != 32767.0] = tmmx[tmmx != 32767.0] * 0.1 + 220

    vpd = gdal.Open(in_path + 'vpd_' + year + ".tif").ReadAsArray()
    vpd[vpd != 32767.0] *= 0.01

    raster = gdal.Open(in_path + 'vpd_' + year + ".tif")
    transform = raster.GetGeoTransform()
    prj = raster.GetProjection()
    srs = osr.SpatialReference(wkt=prj)

    with open(in_path + 'last_updated.txt', 'r') as file:
        date = int(file.read())

    date = day_of_year_to_date(date, int(year))
    new_o = out_path + date + ".tif"
    nx = vpd.shape[0]
    ny = vpd.shape[1]
    dst_ds = gdal.GetDriverByName('GTiff').Create(new_o, ny, nx, 9, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(transform)
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(etr)
    dst_ds.GetRasterBand(2).WriteArray(pet)
    dst_ds.GetRasterBand(3).WriteArray(pr)
    dst_ds.GetRasterBand(4).WriteArray(rmax)
    dst_ds.GetRasterBand(5).WriteArray(rmin)
    dst_ds.GetRasterBand(6).WriteArray(srad)
    dst_ds.GetRasterBand(7).WriteArray(tmmn)
    dst_ds.GetRasterBand(8).WriteArray(tmmx)
    dst_ds.GetRasterBand(9).WriteArray(vpd)
    ds = None

def remove_empty_folders():
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/split_14/"
    tot = len(os.listdir(in_path))
    count = 0
    for q in os.listdir(in_path):
        if len(os.listdir(in_path + q)) == 0:
            print("No files in :", q)
            count += 1
            os.rmdir(in_path + q)
    print(count,"/",tot)

def transfer_tifs_to_all_machines():
    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/merged/"
    for send_to_lattice in range(176, 224):
        if send_to_lattice == int(socket.gethostname().split("-")[1]):
            continue
        out_path = "/s/lattice-" + str(send_to_lattice) + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/merged/"

        for file_name in os.listdir(in_path):
            source_file = os.path.join(in_path, file_name)
            command = ['scp', source_file, 'paahuni@lattice-' + str(send_to_lattice) + ":" + out_path]
            subprocess.run(command)

        print("Sent to lattice-", send_to_lattice)

def chop_in_quadhash():
    quadhash_dir = next(d for d in os.listdir() if os.path.isdir(d) and d.startswith("quadshape_12_"))
    quadhashes = gpd.read_file(os.path.join(quadhash_dir, 'quadhash.shp'))

    in_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/merged/"
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gridmet/split_14/"

    count = 0
    total = len(quadhashes)
    for ind, row in quadhashes.iterrows():
        count+=1
        print("processing:  ", count, "/", total)
        poly,qua = row["geometry"], row["Quadkey"]
        os.makedirs(out_path + qua, exist_ok=True)
        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])

        for f in os.listdir(in_path):
            gdal.Translate(out_path + qua + '/' + f,  in_path + f, projWin=window)

            x = gdal.Open(out_path + qua + '/' + f).ReadAsArray()
            if np.min(x[0]) == np.max(x[0]) == 32767.0:
                os.remove(out_path + qua + '/' + f)
    remove_empty_folders()

if __name__ == '__main__':
    download_nc_file()
    convert_to_tif()
    merge_bands_gridmet()
    chop_in_quadhash()
