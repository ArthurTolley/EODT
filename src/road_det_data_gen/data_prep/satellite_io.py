# src/road-det-data-gen/data_prep/satellite_io.py
import ee
import requests
from datetime import datetime
import rasterio
import numpy as np
from PIL import Image
from pyproj import Transformer, CRS
import os
import math
import logging
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_ee_initialized = False

def authenticate_earth_engine(project_id):
    global _ee_initialized
    if _ee_initialized:
        return
    try:
        if not ee.data._credentials:
            logging.info("Authenticating Earth Engine...")
            ee.Authenticate()
        ee.Initialize(project=project_id)
        _ee_initialized = True
        logging.info("Earth Engine initialized successfully.")
    except Exception as e:
        logging.error(f"Error during Earth Engine authentication: {e}")
        raise e

def estimate_image_bytes(start_lat, start_lon, end_lat, end_lon, scale, satellite_config):
    """
    Estimates the size of a satellite image download by calculating dimensions
    in the target projection (EPSG:3857), which is what GEE uses for downloads.
    This is much more accurate than Haversine-based approximations.
    """
    # Define the source (WGS84) and target (Web Mercator) projections
    wgs84 = CRS.from_epsg(4326)
    web_mercator = CRS.from_epsg(3857)
    transformer = Transformer.from_crs(wgs84, web_mercator, always_xy=True)

    # Project the corner points of the bounding box
    min_x, min_y = transformer.transform(start_lon, start_lat)
    max_x, max_y = transformer.transform(end_lon, end_lat)

    # Calculate width and height in meters within the projected system
    width_m = max_x - min_x
    height_m = max_y - min_y

    # Calculate pixel dimensions
    width_px = math.ceil(width_m / scale)
    height_px = math.ceil(height_m / scale)

    # Determine band count and bytes per band based on satellite type
    sat_type = satellite_config.get('type', 'S2').upper()
    bands = 2 if sat_type == 'S1' else 3
    bytes_per_band = 4 # Assume float32 for all GEE downloads, which is the worst-case size

    # Use a small correction factor for metadata/compression overhead
    correction_factor = 4 

    estimated_bytes = (width_px * height_px * bands * bytes_per_band) * correction_factor

    logging.info(f"Projection-aware size estimate: {width_px}x{height_px}px -> {estimated_bytes/1e6:.2f}MB")

    return estimated_bytes

def split_bbox_if_needed(start_lat, end_lat, start_lon, end_lon, scale, satellite_config, max_bytes=50331648):
    """
    Splits a bounding box into smaller chunks if the estimated download size is too large.
    """
    est_bytes = estimate_image_bytes(start_lat, start_lon, end_lat, end_lon, scale, satellite_config)

    if est_bytes <= max_bytes:
        return [(start_lat, end_lat, start_lon, end_lon)]

    # Calculate the split factor based on the more accurate size estimation
    factor = math.ceil((est_bytes / max_bytes) ** 0.5)
    logging.info(f"Estimated size {est_bytes/1e6:.2f}MB > {max_bytes/1e6:.2f}MB. Splitting into {int(factor)}x{int(factor)} boxes.")
    
    lat_step = (end_lat - start_lat) / factor
    lon_step = (end_lon - start_lon) / factor

    boxes = []
    for i in range(int(factor)):
        for j in range(int(factor)):
            boxes.append((
                start_lat + i * lat_step, start_lat + (i + 1) * lat_step,
                start_lon + j * lon_step, start_lon + (j + 1) * lon_step
            ))
    return boxes

# --- The functions below are unchanged but included for completeness ---

def download_satellite_image(lat_st, lon_st, lat_ed, lon_ed, scale, output_path, satellite_config):
    if not ee.data._credentials:
        raise ValueError("Google Earth Engine is not authenticated.")
    region = ee.Geometry.Polygon(
        [[[lon_st, lat_ed], [lon_st, lat_st], [lon_ed, lat_st], [lon_ed, lat_ed], [lon_st, lat_ed]]]
    )
    sat_type = satellite_config.get('type', 'S2').upper()
    image = None
    logging.info(f"Querying for {sat_type} imagery...")

    if sat_type == 'S2':
        cloud_cover = satellite_config.get('cloud_cover', 10)
        collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate('2023-01-01', datetime.now().strftime('%Y-%m-%d')).filterBounds(region).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
        s2_image = collection.sort('system:time_start', False).first()
        if s2_image:
            image = s2_image.select(['B4', 'B3', 'B2'])
    elif sat_type == 'S1':
        orbit_pass = satellite_config.get('orbit_pass', 'ASCENDING')
        collection = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(region).filter(ee.Filter.eq('instrumentMode', 'IW')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
        image = collection.limit(10, 'system:time_start', False).mean().select(['VV', 'VH'])
    else:
        raise ValueError(f"Unsupported satellite type: '{sat_type}'.")

    if image is None or image.bandNames().size().getInfo() == 0:
        raise FileNotFoundError(f"No valid {sat_type} image found for the given criteria.")

    url = image.getDownloadURL({'region': region, 'scale': scale, 'crs': 'EPSG:3857', 'format': 'GEO_TIFF'})
    with open(output_path, "wb") as f:
        f.write(requests.get(url).content)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"Downloaded {sat_type} image size: {file_size_mb:.2f} MB")
    return output_path

def process_geotiff_image(tif_path, save_path, satellite_config, size=(512, 512)):
    sat_type = satellite_config.get('type', 'S2').upper()
    with rasterio.open(tif_path) as src:
        if src.nodata is not None and np.all(src.read(masked=True) == src.nodata):
            logging.warning(f"Tile at {tif_path} is empty, skipping.")
            return None
        if sat_type == 'S2':
            img_data = src.read([1, 2, 3]); img_data = np.transpose(img_data, (1, 2, 0))
            img_normalized = np.clip(img_data / 10000.0, 0, 1) * 255; img_to_save = img_normalized.astype(np.uint8)
        elif sat_type == 'S1':
            img_data = src.read([1, 2]).astype(np.float32); output_img = np.zeros((img_data.shape[1], img_data.shape[2], 3), dtype=np.uint8)
            vv, vh = img_data[0], img_data[1]
            output_img[:, :, 0] = np.clip((vv - (-25)) / (0 - (-25)), 0, 1) * 255
            output_img[:, :, 1] = np.clip((vh - (-30)) / (-5 - (-30)), 0, 1) * 255
            ratio = vv / (vh + 1e-6); output_img[:, :, 2] = np.clip((ratio - 0) / (10 - 0), 0, 1) * 255
            img_to_save = output_img
        else:
            raise ValueError(f"Cannot process unsupported satellite type: {sat_type}")
        img_resized = Image.fromarray(img_to_save).resize(size, Image.BILINEAR)
        if np.all(np.array(img_resized) == 0):
            logging.warning(f"Tile at {tif_path} resulted in a black image, skipping.")
            return None
        img_resized.save(save_path)
        logging.info(f"Saved processed tile to: {save_path}")
    with rasterio.open(tif_path) as src:
        return convert_bbox_3857_to_4326(src.bounds)

def get_image_resolution(tif_path):
    with rasterio.open(tif_path) as src:
        return src.width, src.height

def slice_geotiff_image(tif_path, output_folder, tile_size=(512, 512), overlap=0):
    tile_coords = {}
    with rasterio.open(tif_path) as src:
        width, height, bands, dtype = src.width, src.height, src.count, src.dtypes[0]
        stride_x, stride_y = tile_size[0] - overlap, tile_size[1] - overlap
        for i in range(0, width, stride_x):
            for j in range(0, height, stride_y):
                win_width, win_height = min(tile_size[0], width - i), min(tile_size[1], height - j)
                window = Window(i, j, win_width, win_height)
                transform = src.window_transform(window)
                tile_data = src.read(window=window)
                tile_full = np.zeros((bands, tile_size[1], tile_size[0]), dtype=dtype)
                tile_full[:, :win_height, :win_width] = tile_data
                output_path = os.path.join(output_folder, f"tile_{i}_{j}.tif")
                with rasterio.open(output_path, 'w', driver='GTiff', height=tile_size[1], width=tile_size[0], count=bands, dtype=dtype, crs=src.crs, transform=transform) as dst:
                    dst.write(tile_full)
                tile_coords[(i, j)] = {'geo_bounds': convert_bbox_3857_to_4326(src.window_bounds(window))}
    return tile_coords

def convert_bbox_3857_to_4326(bounds):
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    if hasattr(bounds, 'left'):
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    else:
        left, bottom, right, top = bounds
    lon_min, lat_min = transformer.transform(left, bottom)
    lon_max, lat_max = transformer.transform(right, top)
    return (lat_min, lat_max, lon_min, lon_max)