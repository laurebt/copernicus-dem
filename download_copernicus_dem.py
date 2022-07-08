import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import io
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import tempfile
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.mask import mask
from copy import deepcopy

s3 = boto3.client('s3', region_name='eu-central-1', config=Config(signature_version=UNSIGNED))

from rasterio.io import MemoryFile

def create_dataset(data, crs, transform):
    # Receives a 2D array, a transform and a crs to create a rasterio dataset
    memfile = MemoryFile()
    dataset = memfile.open(driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, crs=crs, 
                           transform=transform, dtype=data.dtype)
    dataset.write(data, 1)
        
    return dataset

@st.cache
def load_lat_lon():
    coords = pd.read_excel('tileList_30.xlsx')
    lon = coords['lon'].values
    lat = coords['lat'].values

    lon_plot = []
    lat_plot = []
    for i, l in enumerate(lat):
        ll = str(l)
        if ll[0] == 'S':
            lat_plot.append(-1*int(ll[1:]))
        else:
            lat_plot.append(1*int(ll[1:]))
    for i, l in enumerate(lon):
        ll = str(l)
        if ll[0] == 'W':
            lon_plot.append(-1*int(ll[1:]))
        else:
            lon_plot.append(1*int(ll[1:]))
    
    return lon_plot, lat_plot


def get_dem_points_in_poly(lon, lat, features, attributes_select, country_selector):
    
    need_to_expand = True
    df = pd.DataFrame({'lon':lon, 'lat':lat})
    df['coords'] = list(zip(df['lon'],df['lat']))
    df['coords'] = df['coords'].apply(Point)
    points = gpd.GeoDataFrame(df, geometry='coords', crs=features.crs)
    p = deepcopy(points)
    pointInPolys = gpd.tools.sjoin(p, features, op="within", how='left')
    
    pnt_FR = p[pointInPolys[attributes_select]==country_selector]
    print(len(pnt_FR))
    if len(pnt_FR) == 0:
        geom = features[features[attributes_select]==country_selector]['geometry']
        print(geom.centroid)
        # s = gpd.GeoSeries([geom])

        # print(s.centroid)
        del pointInPolys, pnt_FR
        pointInPolys = gpd.tools.sjoin_nearest(points,  gpd.GeoDataFrame(geometry=geom.centroid), how='left', distance_col="distances", max_distance = 1.0)
        print(pointInPolys)
        pnt_FR = pointInPolys.dropna() #points[pointInPolys[attributes_select]==country_selector]
        print(pnt_FR)
        need_to_expand = False

    return points, pointInPolys, pnt_FR, need_to_expand




# def keys(bucket_name = 'copernicus-dem-90m', prefix='/', delimiter='/', start_after=''):
#     all_listing = []
#     prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
#     start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
#     for page in s3_paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
#         for content in page.get('Contents', ()):
#             print(content['Key'])
#             all_listing.append(content['Key'])
#             # yield content['Key']
#     return all_listing


def get_from_lat_long(lat=8,n='N',lon=0, e='E', resolution='90'):
    
    lat_str = str(int(lat))
    for i in range(2-len(lat_str)):
        lat_str = '0' + lat_str
    lat_str = n + lat_str

    lon_str = str(int(lon))
    for i in range(3-len(lon_str)):
        lon_str = '0' + lon_str
    lon_str = e + lon_str
    if resolution == '90':
        name = f'Copernicus_DSM_COG_30_{lat_str}_00_{lon_str}_00_DEM'
        bucket = 'copernicus-dem-90m'
    elif resolution == '30':
        name = f'Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM'
        bucket = 'copernicus-dem-30m'

    file_name = f'{name}/{name}.tif'


    print(file_name)

    tf = tempfile.NamedTemporaryFile()
    
    s3.download_file(Bucket=bucket, Key=file_name, Filename = tf.name)#, Filename=f'{name}.tif')
    dataset= rasterio.open(tf.name)

    return file_name, dataset


def plt_locs(lon_plot, lat_plot, user_lon, user_lat):
    
    fig, ax = plt.subplots(figsize = (12,8))
    ax.plot(lon_plot, lat_plot, 'ko', ms = 0.5)
    ax.plot( user_lon, user_lat, 'ro', ms = 10.0)
    ax.grid(which='both')

    return fig

def retrieve_dem(polygon):
    pass


def st_ui():
    st.set_page_config(layout = "wide")
    st.title("Copernicus DEM download")
    lon = st.text_input("Longitude", 0)
    n = st.selectbox("Northing", ['N', 'S'])
    
    lat = st.text_input("Latitude", 8)
    e = st.selectbox("Easting", ['E', 'W'])

    button = st.button('View')
    if button:

        try:
            file, src = get_from_lat_long(lat=int(lat),n=n,lon=int(lon), e=e, resolution='90')
            print(src)
            buf =  BytesIO()
            plt.imshow(src.read()[0,:,:], cmap='pink')
            plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
            st.image(buf, use_column_width=False, caption='localization of DEM data')
            plt.close()
        except:
            st.write("Can't download")


if __name__ == "__main__":
    st_ui()



