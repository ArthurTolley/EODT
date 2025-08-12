# app/data_prep/sentinel.py
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

_ee_initialized = False # Global flag to prevent re-initialization within the same process

def authenticate_earth_engine(project_id):
    """
    Authenticate Google Earth Engine and initialize it.
    This function should be called before using any Earth Engine functionality in a given process.
    It handles interactive authentication if needed and initializes the API.
    """
    global _ee_initialized
    if _ee_initialized:
        logging.debug("Earth Engine already initialized in this process.")
        return

    try:
        if not ee.data._credentials:
            logging.info("Authenticating Earth Engine (may open browser)...")
            ee.Authenticate()
            logging.info("Earth Engine authentication complete.")
        ee.Initialize(project=project_id)
        _ee_initialized = True
        logging.info("Earth Engine authenticated successfully.")
    except Exception as e:
        logging.error("Error during Earth Engine authentication or initialization: %s", e)
        raise e

def download_sentinel_image(lat_st, lon_st, lat_ed, lon_ed, scale, output_path):
    """
    Download a Sentinel-2 image for the specified bounding box and save it as a GeoTIFF file.
    :param lat_st: Starting latitude of the bounding box.
    :param lon_st: Starting longitude of the bounding box.
    :param lat_ed: Ending latitude of the bounding box.
    :param lon_ed: Ending longitude of the bounding box.
    :param output_path: Path where the downloaded image will be saved.
    :return: Path to the downloaded image.
    """

    if not ee.data._credentials:
        raise ValueError("Google Earth Engine is not authenticated. Please call authenticate_earth_engine(project_id) first.")

    coords = [[[lon_st, lat_ed], [lon_st, lat_st], [lon_ed, lat_st], [lon_ed, lat_ed], [lon_st, lat_ed]]]
    region = ee.Geometry.Polygon(coords)

    sentinel = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterDate('2023-01-01', datetime.now().strftime('%Y-%m-%d')) \
        .filterBounds(region) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))

    image = sentinel.sort('system:time_start', False).first().select(['B4', 'B3', 'B2'])
    image_date = image.get('system:time_start').getInfo()
    logging.info("Image date: %s", datetime.utcfromtimestamp(image_date / 1000).strftime('%Y-%m-%d'))

    url = image.getDownloadURL({
        'region': region,
        'scale': scale,
        'crs': 'EPSG:3857', # 4326 is deg
        'format': 'GEO_TIFF'
    })

    with open(output_path, "wb") as f:
        f.write(requests.get(url).content)

    # Print downloaded file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info("Downloaded image size: %.2f MB", file_size_mb)

    return output_path

def get_image_resolution(tif_path):
    """
    Get the resolution of a GeoTIFF image.
    :param tif_path: Path to the GeoTIFF file.
    :return: Tuple containing (width, height) of the image.
    """
    with rasterio.open(tif_path) as src:
        return src.width, src.height

def process_geotiff_image(tif_path, save_path, size=(512, 512)):
    with rasterio.open(tif_path) as src:

        # Read bands 1, 2, 3 = (R, G, B)
        img = src.read([1, 2, 3])  # shape: (3, H, W)
        img = np.transpose(img, (1, 2, 0))  # shape: (H, W, 3)

        # Normalize from 0–10000 to 0–255 (Sentinel-2 reflectance range)
        img = np.clip(img / 10000, 0, 1) * 255
        img = img.astype(np.uint8)

        # Resize to desired output size (optional)
        img = Image.fromarray(img)
        img_resized = img.resize(size, Image.BILINEAR)
        print(np.shape(img_resized))

        # Count black pixels
        arr = np.array(img_resized)  # shape: (H, W, 3)
        black_mask = np.all(arr == 0, axis=-1)  # shape: (H, W), True if pixel is black
        black_pixels = np.sum(black_mask)
        total_pixels = arr.shape[0] * arr.shape[1]
        logging.info("Black pixels: %d out of %d (%.2f%%)", black_pixels, total_pixels, 100 * black_pixels / total_pixels)

        # Optional threshold
        if black_pixels > 0.97 * total_pixels:
            logging.warning("Too many black pixels, skipping.")
            return None

        # Save as PNG or JPEG
        img_resized.save(save_path)
        logging.info("Saved image: %s", save_path)

    bounds = src.bounds
    start_lat, end_lat, start_lon, end_lon = convert_bbox_3857_to_4326(bounds)

    return (start_lat, end_lat, start_lon, end_lon)

def slice_geotiff_image(tif_path, output_folder, tile_size=(512, 512), overlap=0):
    """
    Slice a GeoTIFF image into smaller tiles.
    :param tif_path: Path to the GeoTIFF file.
    :param output_folder: Folder where the sliced images will be saved.
    :param tile_size: Size of each tile (width, height).
    :param overlap: Overlap between tiles in pixels.
    """
    tile_coords = {}
    with rasterio.open(tif_path) as src:
        width = src.width
        height = src.height
        bands = src.count
        dtype = src.dtypes[0]

        stride_x = tile_size[0] - overlap
        stride_y = tile_size[1] - overlap

        for i in range(0, width, stride_x):
            for j in range(0, height, stride_y):
                # Actual size of the window from the original image
                win_width = min(tile_size[0], width - i)
                win_height = min(tile_size[1], height - j)

                window = Window(i, j, win_width, win_height)
                transform = src.window_transform(window)

                # Read partial tile
                tile_data = src.read(window=window)

                # Initialize black canvas (H, W, C) and paste in the real data
                tile_full = np.zeros((bands, tile_size[1], tile_size[0]), dtype=dtype)
                tile_full[:, :win_height, :win_width] = tile_data

                output_path = os.path.join(output_folder, f"tile_{i}_{j}.tif")

                with rasterio.open(output_path, 'w',
                                   driver='GTiff',
                                   height=tile_size[1], width=tile_size[0],
                                   count=bands, dtype=dtype,
                                   crs=src.crs,
                                   transform=transform) as dst:
                    dst.write(tile_full)

                logging.info("Saved tile: %s", output_path)

                # Store coordinates for later processing
                bounds = src.window_bounds(window)
                start_lat, end_lat, start_lon, end_lon = convert_bbox_3857_to_4326(bounds)
                tile_coords[(i, j)] = {
                    'start_lat': start_lat,
                    'end_lat': end_lat,
                    'start_lon': start_lon,
                    'end_lon': end_lon,
                    'end_width_pixels': win_width,
                    'end_height_pixels': win_height
                }
    return tile_coords

def convert_bbox_3857_to_4326(bounds):
    """
    Convert bounding box from EPSG:3857 to EPSG:4326.
    :param bounds: rasterio BoundingBox with left, bottom, right, top in EPSG:3857
    :return: (start_lat, end_lat, start_lon, end_lon) in EPSG:4326
    """
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    if not hasattr(bounds, 'left'):
        left, bottom, right, top = bounds
    else:
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

    # Unpack and convert
    lon_min, lat_min = transformer.transform(left, bottom)
    lon_max, lat_max = transformer.transform(right, top)

    # Return in (lat_start, lat_end, lon_start, lon_end) format if needed
    return lat_min, lat_max, lon_min, lon_max

def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def estimate_image_bytes(start_lat, start_lon, end_lat, end_lon, scale, bands=3, bytes_per_band=4, correction_factor=2.0):
    width_m = haversine(start_lat, start_lon, start_lat, end_lon) * 1000
    height_m = haversine(start_lat, start_lon, end_lat, start_lon) * 1000
    width_pixels = width_m / scale
    height_pixels = height_m / scale
    raw_bytes = width_pixels * height_pixels * bands * bytes_per_band
    total_bytes = raw_bytes * correction_factor
    return total_bytes, width_pixels, height_pixels

def split_bbox_if_needed(start_lat, end_lat, start_lon, end_lon, scale, max_bytes=50331648):
    est_bytes, width_px, height_px = estimate_image_bytes(start_lat, start_lon, end_lat, end_lon, scale)

    if est_bytes <= max_bytes:
        return [(start_lat, end_lat, start_lon, end_lon)]

    # Decide how many splits are needed
    factor = math.ceil((est_bytes / max_bytes) ** 0.5)

    lat_step = (end_lat - start_lat) / factor
    lon_step = (end_lon - start_lon) / factor

    boxes = []
    for i in range(factor):
        for j in range(factor):
            lat0 = start_lat + i * lat_step
            lat1 = lat0 + lat_step
            lon0 = start_lon + j * lon_step
            lon1 = lon0 + lon_step
            boxes.append((lat0, lat1, lon0, lon1))
    return boxes
