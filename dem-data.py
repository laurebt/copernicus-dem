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
# s3.download_file(Bucket='copernicus-dem-30m', Key='tileList.txt', Filename='tileList_30.txt')
# s3_paginator = boto3.client('s3', region_name='eu-central-1', config=Config(signature_version=UNSIGNED)).get_paginator('list_objects_v2')

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


def get_from_lat_long(lat=0,n='N',lon=0, e='E', resolution='90'):
    
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
    shapefiles_list = ['data/World_Countries_Generalized/World_Countries__Generalized_.shp',
                        'data/cb_2018_us_county_20m/cb_2018_us_county_20m.shp',
                        'data/Basins_classified_by_sub_regime_Robertson_Basins_and_Plays/Basins classified by sub regime Robertson Basins and Plays.shp']
    shapes_select = st.sidebar.selectbox('Select a shapefile', shapefiles_list)
    lon_plot, lat_plot = load_lat_lon()
    features = gpd.GeoDataFrame.from_file(shapes_select)
    features = features.to_crs('epsg:4326')
    attributes = features.columns.values.tolist()
    print(attributes)
    attributes_select = st.sidebar.selectbox('Select an attribute', attributes)
    nations = features
    countries_list =list(set(features[attributes_select].values))
    
    country_selector = st.sidebar.selectbox(f'Select a {attributes_select}', sorted(countries_list))

    points, pointInPolys, pnt_FR, need_to_expand = get_dem_points_in_poly(np.array(lon_plot), np.array(lat_plot), features, attributes_select, country_selector)
    
   
    

    base = nations.boundary.plot(linewidth=0.1, edgecolor="black", figsize = (12,8), aspect = None)
    points.plot(ax=base, linewidth=1, color="blue", markersize=0.01)
    pnt_FR.plot(ax=base, linewidth=1, color="red", markersize=0.5)


    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
    
    st.image(buf, use_column_width=False, caption='localization of DEM data')
    plt.close()

    st.write(f"Found {len(pnt_FR)} rasters included in the area")
    if need_to_expand:
        offset=1
    else:
        offset=0
    lats = list(range(np.min(pnt_FR['lat'].values)-offset, np.max(pnt_FR['lat'].values)+offset+1))
    lons = list(range(np.min(pnt_FR['lon'].values)-offset, np.max(pnt_FR['lon'].values)+offset+1))
    pp = []
    for ll in lons:
        for l in lats:
            pp.append([ll,l])
    pp = np.array(pp)
    print(lons, lats)
    print(pp)
    print(pnt_FR['lon'].values, pnt_FR['lat'].values)
    shape = nations[nations[attributes_select]==country_selector]['geometry']

    buf = BytesIO()
    fig, ax = plt.subplots()
    ax.plot(pp[:,0], pp[:,1], 'o')
    ax.plot(pnt_FR['lon'].values, pnt_FR['lat'].values, 'o')
    shape.plot(ax=ax)
    plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
    
    st.image(buf, use_column_width=False, caption='localization of DEM data')
    plt.close()

    button = st.sidebar.button('Get list')
    if button:
        shape = nations[nations[attributes_select]==country_selector]['geometry']
        print(shape)
        
        

        st.write(f"Downloading tentatively {len(lats)*len(lons)} rasters")

        src_files_to_mosaic = []
        mybar = st.progress(0)
        ii = 0
        # for index, row in pnt_FR.iterrows():
        for l in lats:
            for ll in lons:
                multe = 1
                multn = 1
                e = 'E'
                n = 'N'
                print(l, ll)
                if int(ll) < 0:
                    e = 'W'
                    multe = -1
                if int(l) < 0:
                    n = 'S'
                    multn = -1
                try:
                    file, src = get_from_lat_long(lat=int(multn*int(l)),n=n,lon=int(multe*int(ll)), e=e, resolution='90')
                    src_files_to_mosaic.append(src)
                    st.write(file)

                except:
                    st.write("Can't download")
                    continue
                mybar.progress((ii+1)/(len(lats)*len(lons)))
                ii += 1
        mosaic, out_trans = merge(src_files_to_mosaic)
        src = mosaic

        src_ds = create_dataset(src[0], src_files_to_mosaic[0].profile['crs'], out_trans)
        out_image, out_transform = mask(src_ds, shape, crop=True)
        out_dataset = create_dataset(out_image[0], src_files_to_mosaic[0].profile['crs'], out_transform)
        buf = BytesIO()
        
        

        plt.imshow(out_image[0,:,:], cmap='pink')
        plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
        st.image(buf, use_column_width=False, caption='localization of DEM data')
        plt.close()






if __name__ == "__main__":
    st_ui()



