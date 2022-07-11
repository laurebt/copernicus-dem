from math import remainder
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pickle
from mercantile import feature
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
import time
import openpyxl
import pygeos
import pickle
from matplotlib.backends.backend_agg import RendererAgg
import base64
import os

_lock = RendererAgg.lock

# import pydaisi as pyd
# copernicus_dem_download = pyd.Daisi("laiglejm/Copernicus DEM download")
# copernicus_dem_download.workers.set(4)

s3 = boto3.client('s3', region_name='eu-central-1', config=Config(signature_version=UNSIGNED))
# s3.download_file(Bucket='copernicus-dem-30m', Key='tileList.txt', Filename='tileList_30.txt')
# s3_paginator = boto3.client('s3', region_name='eu-central-1', config=Config(signature_version=UNSIGNED)).get_paginator('list_objects_v2')
shapefiles_dict = {'World countries' : {'src':'data/World_Countries_Generalized/World_Countries__Generalized_.shp', 'main attribute':'COUNTRY', 'Attribute display':'Country'},
                        'US States' : {'src':'data/cb_2018_us_state_5m/cb_2018_us_state_5m.shp', 'main attribute':'NAME', 'Attribute display':'State'},
                        'US counties' : {'src':'data/cb_2018_us_county_20m/cb_2018_us_county_20m.shp', 'main attribute':'NAME', 'Attribute display':'County'},
                        'World sedimentary basins' : {'src':'data/Basins_classified_by_sub_regime_Robertson_Basins_and_Plays/Basins classified by sub regime Robertson Basins and Plays.shp', 'main attribute':'BASIN_NAME', 'Attribute display':'Basin'}}
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
    coords = pd.read_excel('tileList.xlsx')
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

def get_dem_points_in_poly(lon, lat, features, attributes_select=None, ft_selector=None):
    
    df = pd.DataFrame({'lon':lon, 'lat':lat})
    df['coords'] = list(zip(df['lon'],df['lat']))
    df['coords'] = df['coords'].apply(Point)
    points = gpd.GeoDataFrame(df, geometry='coords', crs='epsg:4026')
    p = deepcopy(points)
    if features is not None:
        pointInPolys = gpd.tools.sjoin(p, features, op="within", how='left')
        
        if attributes_select is not None:
            pnt_FR = p[pointInPolys[attributes_select]==ft_selector]
            print(len(pnt_FR))
            if len(pnt_FR) == 0:
                geom = features[features[attributes_select]==ft_selector]['geometry']

                del pointInPolys, pnt_FR
                pointInPolys = gpd.tools.sjoin_nearest(points, gpd.GeoDataFrame(geometry=geom.centroid), how='left', distance_col="distances", max_distance = 1.0)
                print(pointInPolys)
                pnt_FR = pointInPolys.dropna() 
                print(pnt_FR)
        else:
            if len(pointInPolys.dropna() ) == 0:
                geom = features['geometry']

                pointInPolys = gpd.tools.sjoin_nearest(points, gpd.GeoDataFrame(geometry=geom.centroid), how='left', distance_col="distances", max_distance = 1.0)
                print(pointInPolys)
                pnt_FR = pointInPolys.dropna() 
                print(pnt_FR)
    else:
        pointInPolys, pnt_FR = None, None


    return points, pointInPolys, pnt_FR




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
    
    s3.download_file(Bucket=bucket, Key=file_name, Filename = tf.name  + '.tiff') #, Filename=f'{name}.tif')
    # dataset= rasterio.open(tf.name)
    print(tf.name)

    return name + '.tif', tf.name + '.tiff'


def plt_locs(lon_plot, lat_plot, user_lon, user_lat):
    
    fig, ax = plt.subplots(figsize = (12,8))
    ax.plot(lon_plot, lat_plot, 'ko', ms = 0.5)
    ax.plot( user_lon, user_lat, 'ro', ms = 10.0)
    ax.grid(which='both')

    return fig

def retrieve_dem(user_polygon = None, pre_defined_shape = ['World countries', 'Andorra'], resolution = '90', return_type = 'image'):
    lon_plot, lat_plot = load_lat_lon()
    if user_polygon is not None:
        pass
    else:
    
        features = gpd.read_file(shapefiles_dict[pre_defined_shape[0]]['src'])
        attributes_select = shapefiles_dict[pre_defined_shape[0]]['main attribute']
        ft_selector = pre_defined_shape[1]
        features = features.to_crs('epsg:4326')
        shape = features[features[attributes_select]==ft_selector]['geometry']

        bounds =  features[features[attributes_select]==ft_selector].bounds
    points, pointInPolys, pnt_FR = get_dem_points_in_poly(np.array(lon_plot), np.array(lat_plot), features, attributes_select, ft_selector)
    offset_minx, offset_maxx, offset_miny, offset_maxy = 0,0,0,0
    if bounds['minx'].values < np.min(pnt_FR['lon'].values):
        offset_minx = 1
    if bounds['maxx'].values > np.max(pnt_FR['lon'].values):
        offset_maxx = 1
    if bounds['miny'].values < np.min(pnt_FR['lat'].values):
        offset_miny = 1
    if bounds['maxy'].values > np.max(pnt_FR['lat'].values):
        offset_maxy = 1
    lats = list(range(np.min(pnt_FR['lat'].values)-offset_miny, np.max(pnt_FR['lat'].values)+offset_maxy+1))
    lons = list(range(np.min(pnt_FR['lon'].values)-offset_minx, np.max(pnt_FR['lon'].values)+offset_maxx+1))
    
    src_files_to_mosaic = []
    args_list = []
    ii = 0
    print(f"Will retrieve tentatively {len(lats)*len(lons)} rasters.")
    for l in lats:
        for ll in lons:
            multe, multn, e, n = 1, 1, 'E', 'N'

            if int(ll) < 0: e, multe = 'W', -1
            if int(l) < 0: n, multn = 'S', -1
            print(l, ll, multe, multn, e, n)
            try:
                file, src = get_from_lat_long(lat=int(multn*int(l)),n=n,lon=int(multe*int(ll)), e=e, resolution=resolution)
                src_files_to_mosaic.append(src)
                print(ii, "Dowloaded")
                ii +=1
            except Exception as e:
                print(e)
                print(ii, "Couldn't download")
                ii+=1
                continue
    print(src_files_to_mosaic)
    mosaic, out_trans = merge(src_files_to_mosaic)

    print("Merge done")
    
    dataset= rasterio.open(src_files_to_mosaic[0])
    crs = dataset.profile['crs']
    dataset.close()

    src_ds = create_dataset(mosaic[0],crs, out_trans)
    out_image, out_transform = mask(src_ds, shape, crop=True)
    
    print("Cropping done")

    for s in src_files_to_mosaic:
        os.remove(s)
    
    # out_dataset = create_dataset(out_image[0], src_files_to_mosaic[0].profile['crs'], out_transform)
    # crs = src_files_to_mosaic[0].profile['crs']

    # for s in src_files_to_mosaic:
    #     s.close()

    if return_type == 'image':
        buf = BytesIO()
        data_top_plot = out_image[0,:,:]
        data_top_plot[data_top_plot == 0] = np.nan
        plt.imshow(data_top_plot, cmap='pink', aspect='equal')
        plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
        plt.close()

        return base64.b64encode(buf)

    else:
        # out_ds = create_dataset(out_image[0],crs, out_transform)
        # metadata = dict(**out_ds.profile)
        # metadata.update(crs=metadata["crs"].to_wkt(), transform=list(metadata["transform"]))

        # out_data = out_image[0]
        out_file = tempfile.NamedTemporaryFile()

        name = out_file.name + '.tiff'

        new_dataset = rasterio.open(name, 'w', driver='GTiff',
                                    height=out_image[0].shape[0],
                                    width=out_image[0].shape[1],
                                    count=1,
                                    dtype=out_image[0].dtype,
                                    crs=crs,
                                    transform=out_transform)
        new_dataset.write(out_image[0], 1)
        new_dataset.close()

        with open(name, 'rb') as f:
            to_return = f.read()
        os.remove(name)
        return to_return

def st_ui():
    st.set_page_config(layout = "wide")
    st.title("Copernicus DEM download")

    col1, col2 = st.columns(2)

    res = st.sidebar.radio("Copernicus DEM resolution", ("90m (recommended for faster download and larger areas", "30m"))

    resolution = '90'
    if res == "30m":
        resolution = '30'
    
    lon_plot, lat_plot = load_lat_lon()

    shp_input_select = st.sidebar.radio("Shapefile selection", ('Use a pre loaded shapefile', 'Bring your own data (Geojson)'))
    if shp_input_select == "Use a pre loaded shapefile":
        shapes_select = st.sidebar.selectbox('Select a shapefile', list(shapefiles_dict.keys()))
        features = gpd.read_file(shapefiles_dict[shapes_select]['src'])
        features = features.to_crs('epsg:4326')
    
    elif shp_input_select == "Bring your own data (Geojson)":
        st.sidebar.markdown('''*It is easy to draw a polygon on [Geojson.io](http://geojson.io)*''')
        user_shp = st.sidebar.file_uploader("Upload a Geojson file")
        if user_shp is not None:
            features = gpd.GeoDataFrame.from_file(user_shp)
            features = features.to_crs('epsg:4326')
    try:
        attributes = features.columns.values.tolist()
        print(attributes)
        if shp_input_select == "Use a pre loaded shapefile":
            attributes_select = shapefiles_dict[shapes_select]['main attribute']
            ft_list =list(set(features[attributes_select].values))
            display_name = shapefiles_dict[shapes_select]['Attribute display']
            ft_selector = st.sidebar.selectbox(f'Select a {display_name}', sorted(ft_list))
        else:
            if attributes != 'geometry':
                try:
                    attributes_select = st.sidebar.selectbox("Select an attribute", sorted(attributes))
                    ft_list =list(set(features[attributes_select].values))
                    ft_selector = st.sidebar.selectbox(f'Select a {attributes_select}', sorted(ft_list))
                except:
                    attributes_select = None
                    ft_selector = None
        

        points, pointInPolys, pnt_FR = get_dem_points_in_poly(np.array(lon_plot), np.array(lat_plot), features, attributes_select, ft_selector)
        
        bnds_shp = features.bounds
        minx = np.min(bnds_shp['minx'].values)
        maxx = np.max(bnds_shp['maxx'].values)
        miny = np.min(bnds_shp['miny'].values)
        maxy = np.max(bnds_shp['maxy'].values)

        base = features.boundary.plot(linewidth=0.1, edgecolor="black", figsize = (12,8), aspect = None)
        points.plot(ax=base, linewidth=1, color="blue", markersize=0.01)
        pnt_FR.plot(ax=base, linewidth=1, color="red", markersize=0.5)

        with col1:
            with _lock:
                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
                
                st.image(buf, use_column_width=True, caption='Coverage of Copernicus DEM data')
                plt.close()

        # st.write(f"Found {len(pnt_FR)} rasters included in the area")
        if attributes_select is not None:
            bounds =  features[features[attributes_select]==ft_selector].bounds
        else:
            bounds =  features.bounds
        
        offset_minx, offset_maxx, offset_miny, offset_maxy = 0,0,0,0
        if bounds['minx'].values < np.min(pnt_FR['lon'].values):
            offset_minx = 1
        if bounds['maxx'].values > np.max(pnt_FR['lon'].values):
            offset_maxx = 1
        if bounds['miny'].values < np.min(pnt_FR['lat'].values):
            offset_miny = 1
        if bounds['maxy'].values > np.max(pnt_FR['lat'].values):
            offset_maxy = 1
        lats = list(range(np.min(pnt_FR['lat'].values)-offset_miny, np.max(pnt_FR['lat'].values)+offset_maxy+1))
        lons = list(range(np.min(pnt_FR['lon'].values)-offset_minx, np.max(pnt_FR['lon'].values)+offset_maxx+1))
        pp = []
        for ll in lons:
            for l in lats:
                pp.append([ll,l])
        pp = np.array(pp)
        print(lons, lats)
        print(pp)
        print(pnt_FR['lon'].values, pnt_FR['lat'].values)
        if attributes_select is not None:
            shape = features[features[attributes_select]==ft_selector]['geometry']
        else:
            shape = features['geometry']

        buf = BytesIO()
        fig, ax = plt.subplots(figsize = (4,4))
        ax.plot(pp[:,0], pp[:,1], 'o')
        ax.plot(pnt_FR['lon'].values, pnt_FR['lat'].values, 'o')
        shape.plot(ax=ax)
        features.boundary.plot(ax = ax, linewidth=0.1, edgecolor="black", aspect = None)

        xlim = ([bounds['minx'].values -3, bounds['maxx'].values +3])
        ylim = ([bounds['miny'].values -3, bounds['maxy'].values +3])
        print(bnds_shp)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
        with col2:
            with _lock:
                st.image(buf, use_column_width=False, caption='localization of DEM data')
        plt.close()
        # download_option = st.sidebar.selectbox("Parallel download", ["No", "Yes"])

        nb_raster = len(lats)*len(lons)
        
        download_size = nb_raster * 5.2
        if resolution == '30':
            download_size = nb_raster * 37.5
        st.sidebar.write(f"Will retrieve tentatively {len(lats)*len(lons)} rasters. Estimated size : {download_size}Mb")
        if ft_selector is not None:
            button = st.sidebar.button(f'Render DEM for {ft_selector}')
        else:
            button = st.sidebar.button(f'Render DEM for your Geojson')

        download_option = "No"
        if button:
            start = time.time()

            src_files_to_mosaic = []
            mybar = st.progress(0)
            ii = 0
            args_list = []
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
                        if download_option == "No":
                            ttt = time.time()
                            # file, data, crs, transform = copernicus_dem_download.get_from_lat_long(lat=int(multn*int(l)),n=n,lon=int(multe*int(ll)), e=e, resolution='90').value
                            # src = create_dataset(data, crs, transform)
                            
                            file, src = get_from_lat_long(lat=int(multn*int(l)),n=n,lon=int(multe*int(ll)), e=e, resolution=resolution)
                            src_files_to_mosaic.append(src)
                            st.write(f"{file} retrieved in {round(time.time() - ttt, 2)} seconds")
                        args_list.append({"lat": int(multn*int(l)), "n": n, "lon":int(multe*int(ll)), "e":e})

                    except:
                        st.write(f"Couldn't download raster at latitude {int(multn*int(l))}{n} / longitude {int(multe*int(ll))}{e}. File not found")
                        continue
                    mybar.progress((ii+1)/(len(lats)*len(lons)))
                    ii += 1
            if  download_option == "Yes":
                k = 0
                chunk_size = 20
                results = dict()
                results["hjkhjk"] = None
                chunks = len(args_list) // chunk_size
                remainder = len(args_list) % chunk_size
                for i in range((chunks)):
                    tt = time.time()
                    argg = args_list[k:k+chunk_size]
                    k += chunk_size
                    dbe = copernicus_dem_download.map(func="compute", args_list=argg)
                
                    dbe.start()
                    with st.spinner("Starting process"):
                        while None in dbe.value.values():
                            time.sleep(1)
                        none_nb = [1 for q in dbe.value.values() if q == None]
                        st.write(len(none_nb))

                    res = dbe.value
                    st.write(time.time() - tt)
                    results = {**results, **res}
                # st.write(res)
                argg = args_list[k:k+remainder]

                dbe = copernicus_dem_download.map(func="compute", args_list=argg)
                
                dbe.start()
                with st.spinner("Starting process"):
                    while None in dbe.value.values():
                        time.sleep(1)

                res = dbe.value
                # st.write(res)
                results = {**results, **res}

                # st.write(results)
                for key, val in results.items():
                    try:
                        src = create_dataset(val[1], val[2], val[3])
                        src_files_to_mosaic.append(src)
                    except:
                        continue
            
            with st.spinner("Rendering"):
                mosaic, out_trans = merge(src_files_to_mosaic)
                src = mosaic
            
                src_ds = create_dataset(src[0], src_files_to_mosaic[0].profile['crs'], out_trans)
                out_image, out_transform = mask(src_ds, shape, crop=True)
                out_dataset = create_dataset(out_image[0], src_files_to_mosaic[0].profile['crs'], out_transform)

            st.write(time.time() - start)

            buf = BytesIO()
            data_top_plot = out_image[0,:,:]
            data_top_plot[data_top_plot == 0] = np.nan
            plt.imshow(data_top_plot, cmap='pink', aspect='equal')
            plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
            st.image(buf, use_column_width=False, caption='localization of DEM data')
            plt.close()

            tf = tempfile.NamedTemporaryFile()

            new_dataset = rasterio.open(tf.name, 'w', driver='GTiff',
                                        height=out_image[0].shape[0],
                                        width=out_image[0].shape[1],
                                        count=1,
                                        dtype=out_image[0].dtype,
                                        crs=out_dataset.profile['crs'],
                                        transform=out_transform)
            new_dataset.write(out_image[0], 1)
            new_dataset.close()
            print(tf.name + '.tiff')

            with open(tf.name, 'rb') as f:
                to_return = f.read()
            
            print(to_return)

            st.sidebar.download_button(label="Download this DEM", data=to_return, file_name='my_DEM.tif')
    except:
        points, pointInPolys, pnt_FR = get_dem_points_in_poly(np.array(lon_plot), np.array(lat_plot), features=None, attributes_select=None, ft_selector=None)
        fig, ax = plt.subplots()
        points.plot(ax=ax, linewidth=1, color="blue", markersize=0.01)


        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', transparent = True, dpi=200)
        
        st.image(buf, use_column_width=False, caption='Coverage of Copernicus DEM data')
        plt.close()



if __name__ == "__main__":
    # st_ui()
    out = retrieve_dem(pre_defined_shape = ['World countries', 'Andorra'], return_type = 'dem')
    print(out)
    # print(base64.b64encode(out))



