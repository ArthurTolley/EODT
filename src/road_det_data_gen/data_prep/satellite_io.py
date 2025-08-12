# src/road-det-data-gen/data_prep/satellite_io.py
import ee
import requests
from datetime import datetime
import rasterio
import numpy as np
from PIL import Image
from pyproj import Transformer
import os
import math
import logging
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_ee_initialized = False # Global flag

def authenticate_earth_engine(project_id):
    """
    Authenticate and initialize Google Earth Engine.
    This is safe to call multiple times; it will only initialize once per process.
    """
    global _ee_initialized
    if _ee_initialized:
        return
    try:
        if not ee.data._credentials:
            logging.info("Authenticating Earth Engine (may require user interaction)...")
            ee.Authenticate()
        ee.Initialize(project=project_id)
        _ee_initialized = True
        logging.info("Earth Engine initialized successfully for process.")
    except Exception as e:
        logging.error(f"Error during Earth Engine authentication: {e}")
        raise e

def download_satellite_image(lat_st, lon_st, lat_ed, lon_ed, scale, output_path, satellite_config):
    """
    Downloads a satellite image based on the provided configuration.
    :param satellite_config: Dict with satellite parameters, e.g., {'type': 'S2', 'cloud_cover': 10}.
    """
    if not ee.data._credentials:
        raise ValueError("Google Earth Engine is not authenticated. Call authenticate_earth_engine() first.")

    region = ee.Geometry.Polygon(
        [[[lon_st, lat_ed], [lon_st, lat_st], [lon_ed, lat_st], [lon_ed, lat_ed], [lon_st, lat_ed]]]
    )

    sat_type = satellite_config.get('type', 'S2').upper()
    image = None
    logging.info(f"Querying for {sat_type} imagery...")

    if sat_type == 'S2':
        cloud_cover = satellite_config.get('cloud_cover', 10)
        collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filterDate('2023-01-01', datetime.now().strftime('%Y-%m-%d')) \
            .filterBounds(region) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
        
        s2_image = collection.sort('system:time_start', False).first()
        if s2_image:
            image = s2_image.select(['B4', 'B3', 'B2']) # Red, Green, Blue bands

    elif sat_type == 'S1':
        orbit_pass = satellite_config.get('orbit_pass', 'ASCENDING')
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(region) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
        
        # Taking the mean of recent images is a robust way to handle speckle
        image = collection.limit(10, 'system:time_start', False).mean().select(['VV', 'VH'])

    else:
        raise ValueError(f"Unsupported satellite type: '{sat_type}'. Supported types are 'S1', 'S2'.")

    if image is None or image.bandNames().size().getInfo() == 0:
        raise FileNotFoundError(f"No valid {sat_type} image found for the given criteria.")

    url = image.getDownloadURL({
        'region': region, 'scale': scale, 'crs': 'EPSG:3857', 'format': 'GEO_TIFF'
    })

    with open(output_path, "wb") as f:
        f.write(requests.get(url).content)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"Downloaded {sat_type} image size: {file_size_mb:.2f} MB")
    return output_path

def process_geotiff_image(tif_path, save_path, satellite_config, size=(512, 512)):
    """
    Processes a GeoTIFF by normalizing and saving it as a PNG for visualization.
    """
    sat_type = satellite_config.get('type', 'S2').upper()
    with rasterio.open(tif_path) as src:
        if sat_type == 'S2':
            img_data = src.read([1, 2, 3]) # Read RGB
            img_data = np.transpose(img_data, (1, 2, 0))
            # Normalize from Sentinel-2's reflectance range
            img_normalized = np.clip(img_data / 10000.0, 0, 1) * 255
            img_to_save = img_normalized.astype(np.uint8)

        elif sat_type == 'S1':
            img_data = src.read([1, 2]) # Read VV, VH
            img_data = np.transpose(img_data, (1, 2, 0))
            # Create a 3-channel image for false-color visualization
            vis_img = np.zeros((img_data.shape[0], img_data.shape[1], 3), dtype=np.uint8)
            # Map VV to Red, VH to Blue, and VV-VH ratio to Green for a nice visual separation
            vv, vh = img_data[:, :, 0], img_data[:, :, 1]
            vis_img[:, :, 0] = np.clip((vv - (-25)) / (0 - (-25)), 0, 1) * 255 # VV (Red)
            vis_img[:, :, 2] = np.clip((vh - (-30)) / (-5 - (-30)), 0, 1) * 255 # VH (Blue)
            # Ratio for Green can highlight different surface types
            ratio = vv - vh
            vis_img[:, :, 1] = np.clip((ratio - 0) / (15 - 0), 0, 1) * 255 # Ratio (Green)
            img_to_save = vis_img

        else:
            raise ValueError(f"Cannot process unsupported satellite type: {sat_type}")

        img_resized = Image.fromarray(img_to_save).resize(size, Image.BILINEAR)

        # Check for excessive black pixels (no-data areas)
        arr = np.array(img_resized)
        if np.all(arr == 0, axis=-1).sum() > 0.97 * arr.shape[0] * arr.shape[1]:
            logging.warning(f"Tile at {tif_path} has too many black pixels, skipping.")
            return None

        img_resized.save(save_path)
        logging.info(f"Saved processed tile to: {save_path}")

    # Return the geographic bounds of the processed tile
    with rasterio.open(tif_path) as src:
        return convert_bbox_3857_to_4326(src.bounds)

# --- The utility functions below do not need to change as they are generic ---

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
                win_width = min(tile_size[0], width - i)
                win_height = min(tile_size[1], height - j)
                window = Window(i, j, win_width, win_height)
                transform = src.window_transform(window)
                tile_data = src.read(window=window)
                
                # Create a full-sized black canvas and paste the tile data
                tile_full = np.zeros((bands, tile_size[1], tile_size[0]), dtype=dtype)
                tile_full[:, :win_height, :win_width] = tile_data

                output_path = os.path.join(output_folder, f"tile_{i}_{j}.tif")
                with rasterio.open(output_path, 'w', driver='GTiff', height=tile_size[1], width=tile_size[0], count=bands, dtype=dtype, crs=src.crs, transform=transform) as dst:
                    dst.write(tile_full)
                
                # Store geographic bounds for this tile
                tile_coords[(i, j)] = {'geo_bounds': convert_bbox_3857_to_4326(src.window_bounds(window))}
    return tile_coords

def convert_bbox_3857_to_4326(bounds):
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    lon_min, lat_min = transformer.transform(left, bottom)
    lon_max, lat_max = transformer.transform(right, top)
    return (lat_min, lat_max, lon_min, lon_max)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def estimate_image_bytes(start_lat, start_lon, end_lat, end_lon, scale, bands=3):
    width_m = haversine(start_lat, start_lon, start_lat, end_lon) * 1000
    height_m = haversine(start_lat, start_lon, end_lat, start_lon) * 1000
    width_px, height_px = width_m / scale, height_m / scale
    # GEE GeoTIFFs are often compressed, this factor accounts for that
    return (width_px * height_px * bands * 4) * 2.0 

def split_bbox_if_needed(start_lat, end_lat, start_lon, end_lon, scale, max_bytes=50331648):
    est_bytes = estimate_image_bytes(start_lat, start_lon, end_lat, end_lon, scale)
    if est_bytes <= max_bytes:
        return [(start_lat, end_lat, start_lon, end_lon)]

    factor = math.ceil((est_bytes / max_bytes) ** 0.5)
    lat_step = (end_lat - start_lat) / factor
    lon_step = (end_lon - start_lon) / factor

    boxes = []
    for i in range(factor):
        for j in range(factor):
            boxes.append((
                start_lat + i * lat_step, start_lat + (i + 1) * lat_step,
                start_lon + j * lon_step, start_lon + (j + 1) * lon_step
            ))
    return boxes