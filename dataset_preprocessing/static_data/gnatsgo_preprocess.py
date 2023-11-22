import pygeohydro as gh
from rasterio.enums import Resampling
from pathlib import Path
import numpy as np
import socket
from pyproj import CRS
import pygeoutils as geoutils
from pynhd import NLDI
import matplotlib.pyplot as plt
import geopandas as gpd
import pyproj
import pandas as pd
from geopy.distance import geodesic
import math
import pickle
import shapely
import rioxarray
import xarray as xr
import pickle
import subprocess
import rasterio
from rasterio.mask import mask
import os
import gdal
from osgeo import ogr, osr
import json

def chop_10m(poly, qua, out_path):
    crs = CRS.from_epsg(4326)

    out_tmp_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gnatsgo/tmp/"
    os.makedirs(out_tmp_path, exist_ok=True)

    if os.path.exists(out_path + qua + "/30m.tif"):
        print("File exists: ", qua, flush=True)
        return

    o_path = out_path + qua + "/10m.tif"
    o_tmp_path = out_tmp_path + qua + "/10m.tif"

    try:
        aws0_5 = gh.soil_gnatsgo("aws0_5", poly, crs).aws0_5
        aws0_5 = aws0_5.where(aws0_5 < 2e6, drop=False) * 10
        aws0_5 = aws0_5.rio.write_nodata(-9999)
        aws0_5 = aws0_5.rio.reproject('EPSG:4326', resampling=5, nodata=aws0_5.rio.nodata)
        if aws0_5.min().item() == aws0_5.max().item() == -9999.0:
            return

        tk0_5a = gh.soil_gnatsgo("tk0_5a", poly, crs).tk0_5a
        tk0_5a = tk0_5a.where(tk0_5a < 2e6, drop=False) * 10
        tk0_5a = tk0_5a.rio.write_nodata(-9999)
        tk0_5a.attrs["units"] = "mm"
        tk0_5a = tk0_5a.rio.reproject('EPSG:4326', resampling=5, nodata=tk0_5a.rio.nodata)
        if tk0_5a.min().item() == tk0_5a.max().item() == -9999.0:
            return

        rasters = xr.concat([aws0_5, tk0_5a], dim="band")

        os.makedirs(out_tmp_path + qua, exist_ok=True)
        rasters.rio.to_raster(o_tmp_path, indexes=[1, 2])

        bounds = list(poly.exterior.coords)
        window = (bounds[0][0], bounds[0][1], bounds[2][0], bounds[2][1])
        os.makedirs(out_path + qua, exist_ok=True)
        gdal.Translate(o_path, o_tmp_path, projWin=window)
        os.remove(o_tmp_path)
        os.removedirs(out_tmp_path + qua)

        rasters.close()
        aws0_5.close()
        tk0_5a.close()

        ds = gdal.Open(os.path.join(out_path, qua, "10m.tif"))
        gdal.Warp(os.path.join(out_path, qua, "30m.tif"),
                            ds, xRes=0.000277777777775, yRes=-0.000277777777775)
        os.remove(os.path.join(out_path, qua, "10m.tif"))
    except:
        return


def chop_90m(poly, qua, out_path):
    crs = CRS.from_epsg(4326)

    if os.path.exists(out_path + qua + "/90m.tif"):
        print("File exists: ", qua, flush=True)
        return

    o_path = out_path + qua + "/90m.tif"
    try:
        porosity = gh.soil_properties("por").porosity
        porosity = geoutils.xarray_geomask(porosity, poly, crs)
        porosity = porosity.where(porosity > porosity.rio.nodata)
        porosity = porosity.rio.write_nodata(-9999)
        if porosity.min().item() == porosity.max().item() == -9999.0:
            return

        awc = gh.soil_properties("awc").awc
        awc = geoutils.xarray_geomask(awc, poly, crs)
        awc = awc.where(awc > awc.rio.nodata)
        awc = awc.rio.write_nodata(-9999)
        if awc.min().item() == awc.max().item() == -9999.0:
            return

        fc = gh.soil_properties("fc").fc
        fc = geoutils.xarray_geomask(fc, poly, crs)
        fc = fc.where(fc > fc.rio.nodata)
        fc = fc.rio.write_nodata(-9999)
        if fc.min().item() == fc.max().item() == -9999.0:
            return

        rasters = xr.concat([porosity, awc, fc], dim="band")

        os.makedirs(out_path + qua, exist_ok=True)
        rasters.rio.to_raster(o_path, indexes=[1, 2, 3])

        rasters.close()
        porosity.close()
        awc.close()
        fc.close()
    except:
        return


def chop_in_quadhash():
    out_path = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gnatsgo/split_14/"
    os.makedirs(out_path, exist_ok=True)

    root_path = "/s/chopin/a/grad/paahuni/cl_3.8/deepSoil/dataset_preprocessing/"
    quadhash_dir = next(
        d for d in os.listdir(root_path) if os.path.isdir(root_path + d) and d.startswith("quadshape_14_"))

    quadhashes = gpd.read_file(os.path.join(root_path, quadhash_dir, 'quadhash.shp'))

    total = len(quadhashes)
    count = 0

    for ind, row in quadhashes.iterrows():
        count += 1
        poly, qua = row["geometry"], row["Quadkey"]
        print("Processing: ", count, "/", total, qua, flush=True)

        chop_90m(poly, qua, out_path)
        chop_10m(poly, qua, out_path)


def remove_empty_folders():
    path_dir = "/s/" + socket.gethostname() + "/b/nobackup/galileo/sm_predictions/input_datasets/gnatsgo/split_14/"
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