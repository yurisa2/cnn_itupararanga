import os
import pandas as pd
import rasterio
import numpy as np
from PIL import Image


dir = "C:/Users/Administrator/Documents/rPDI/artigo/sat/represa"

all_files = os.listdir(dir)

os.chdir(dir)


all_files_df = pd.DataFrame(all_files)

all_files_string = [t[:34] for t in all_files]
all_files_satno = [t[:4][-2:] for t in all_files]
all_files_band = [t[t.find('B'):t.find('.tif')] for t in all_files]

all_files_df["img_name"] = all_files_string
all_files_df["sat_no"] = all_files_satno
all_files_df["band"] = all_files_band

grn_band_files_l08 =  all_files_df[(all_files_df.sat_no == "08") & (all_files_df.band == "B3")]
grn_band_files_l07 =  all_files_df[(all_files_df.sat_no == "07") & (all_files_df.band == "B2")]
grn_band_files_l05 =  all_files_df[(all_files_df.sat_no == "05") & (all_files_df.band == "B2")]
grn_band_files_l04 =  all_files_df[(all_files_df.sat_no == "04") & (all_files_df.band == "B2")]

grn_band_files = grn_band_files_l08
grn_band_files = grn_band_files.append(grn_band_files_l07)
grn_band_files = grn_band_files.append(grn_band_files_l05)
grn_band_files = grn_band_files.append(grn_band_files_l04)

red_band_files_l08 =  all_files_df[(all_files_df.sat_no == "08") & (all_files_df.band == "B4")]
red_band_files_l07 =  all_files_df[(all_files_df.sat_no == "07") & (all_files_df.band == "B3")]
red_band_files_l05 =  all_files_df[(all_files_df.sat_no == "05") & (all_files_df.band == "B3")]
red_band_files_l04 =  all_files_df[(all_files_df.sat_no == "04") & (all_files_df.band == "B3")]

red_band_files = red_band_files_l08
red_band_files = red_band_files.append(red_band_files_l07)
red_band_files = red_band_files.append(red_band_files_l05)
red_band_files = red_band_files.append(red_band_files_l04)

nir_band_files_l08 =  all_files_df[(all_files_df.sat_no == "08") & (all_files_df.band == "B5")]
nir_band_files_l07 =  all_files_df[(all_files_df.sat_no == "07") & (all_files_df.band == "B4")]
nir_band_files_l05 =  all_files_df[(all_files_df.sat_no == "05") & (all_files_df.band == "B4")]
nir_band_files_l04 =  all_files_df[(all_files_df.sat_no == "04") & (all_files_df.band == "B4")]

nir_band_files = nir_band_files_l08
nir_band_files = nir_band_files.append(nir_band_files_l07)
nir_band_files = nir_band_files.append(nir_band_files_l05)
nir_band_files = nir_band_files.append(nir_band_files_l04)


def write_rgb(grn_band, red_band, nir_band, name):

    grn_band = rasterio.open(grn_band)  # red raster
    red_band = rasterio.open(red_band)  # red raster
    nir_band = rasterio.open(nir_band)  # nir raster

    grn = grn_band.read(1).astype('float64')
    red = red_band.read(1).astype('float64')
    nir = nir_band.read(1).astype('float64')

    grn = np.interp(grn,(grn.min(), grn.max()), (0, 255))
    red = np.interp(red,(red.min(), red.max()), (0, 255))
    nir = np.interp(nir,(nir.min(), nir.max()), (0, 255))

    grn = grn.astype('uint8')
    red = red.astype('uint8')
    nir = nir.astype('uint8')

    rgb = np.empty((red_band.height, red_band.width, 3), dtype="uint8")

    rgb[..., 0] = nir
    rgb[..., 1] = red
    rgb[..., 2] = grn

    img = Image.fromarray(rgb)
    img.save("../rgb_files/" + name + "_rgb.jpg")

    return True


nir_band_files.iloc[45]['img_name']
red_band_files.iloc[45][0]

grn_r = rasterio.open(grn_band_files.iloc[45][0])  # nir raster
red_r = rasterio.open(red_band_files.iloc[45][0])  # red raster
nir_r = rasterio.open(nir_band_files.iloc[45][0])  # nir raster
name = nir_band_files.iloc[45]['img_name']


red_file_name = 0
for row in nir_band_files.iterrows():
    img_file = row[1]['img_name']  # IMG main name
    nir_file_name = row[1][0]
    red_file_name = red_band_files[red_band_files.img_name == img_file].iloc[0][0]
    grn_file_name = grn_band_files[grn_band_files.img_name == img_file].iloc[0][0]
    if len(nir_file_name) > 30 and len(red_file_name) > 30 :
        write_rgb(grn_file_name, nir_file_name, red_file_name, img_file)

    # print(img_file)
