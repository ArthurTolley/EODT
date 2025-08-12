import logging
from app.data_prep.sentinel import authenticate_earth_engine, download_sentinel_image, \
                                   process_geotiff_image, get_image_resolution, \
                                   slice_geotiff_image, estimate_image_bytes, \
                                   split_bbox_if_needed
from app.data_prep.osm_utils import process_osm_graph
import json, os, math, shutil, time, sys
from multiprocessing import Pool, current_process

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config_name = "south_uk"
config_path = f"configs/{config_name}.json"
dataset_folder = f"data/training/{config_name}"

# Global variable for Earth Engine project ID
EE_PROJECT_ID = "uksa-training-course-materials"

os.makedirs(dataset_folder, exist_ok=True)

def worker_process_box(box_info):
    """
    Worker function to process a single bounding box.
    """
    box_idx, (slat, elat, slon, elon), config_data = box_info

    # Initialize Earth Engine for this worker process
    # This is crucial for multiprocessing with EE
    authenticate_earth_engine(EE_PROJECT_ID)

    # Create a unique temporary directory for this process/box
    process_id = current_process().pid
    worker_tif_folder = os.path.join(dataset_folder, f"tif_worker_{process_id}_{box_idx}")
    os.makedirs(worker_tif_folder, exist_ok=True)
    worker_tif_path = os.path.join(worker_tif_folder, "sentinel_tmp.tif")

    scale = config_data['scale']
    pixels = config_data['pixels']

    logging.info(f"[Worker {process_id}] Processing box {box_idx} (lat: {slat:.2f}-{elat:.2f}, lon: {slon:.2f}-{elon:.2f})")

    # Download Sentinel-2 Image
    try:
        logging.info(f"[Worker {process_id}] Downloading Sentinel-2 image to {worker_tif_path}...")
        download_sentinel_image(slat, slon, elat, elon, scale, worker_tif_path)
    except Exception as e:
        logging.error(f"[Worker {process_id}] Error downloading Sentinel-2 image for box {box_idx}: {e}")
        # Clean up temporary folder on error
        if os.path.exists(worker_tif_folder):
            shutil.rmtree(worker_tif_folder)
        return

    # Check Resolution of Image
    resolution = get_image_resolution(worker_tif_path)
    logging.info(f"[Worker {process_id}] Image Width (pixels): {resolution[0]}, Height (pixels): {resolution[1]}")

    # Slice the downloaded image into smaller tiles
    # slice_geotiff_image will create tiles like tile_0_0.tif, tile_512_0.tif etc.
    # These need to go into the worker's specific tif_folder
    tile_coords = slice_geotiff_image(worker_tif_path, worker_tif_folder, tile_size=(pixels, pixels), overlap=0)

    # Process each tile
    local_tile_index = 0
    for key in tile_coords:
        local_tile_index += 1
        # Create a globally unique ID for the tile
        title_id = f"AOI_Solent_box{box_idx:03d}_tile{local_tile_index:03d}"

        # The tile_tif_path is now specific to the worker's temporary folder
        current_tile_tif_path = os.path.join(worker_tif_folder, f"tile_{key[0]}_{key[1]}.tif")
        tile_img_path = os.path.join(dataset_folder, f"{title_id}_sat.png")
        output_prefix = os.path.join(dataset_folder, title_id)

        data_geo_bounds = (
            tile_coords[key]['start_lat'], tile_coords[key]['end_lat'],
            tile_coords[key]['start_lon'], tile_coords[key]['end_lon']
        )

        logging.info(f"[Worker {process_id}] Processing tile {title_id}...")
        tile_geo_bounds =  process_geotiff_image(
            current_tile_tif_path,
            tile_img_path,
            size=(pixels, pixels)
        )

        if tile_geo_bounds is None:
            logging.warning(f"[Worker {process_id}] Tile {title_id} has too many black pixels, skipping...")
            continue

        process_osm_graph(
            data_geo_bounds, tile_geo_bounds, output_prefix,
            img_pixel_width=tile_coords[key]['end_width_pixels'],
            img_pixel_height=tile_coords[key]['end_height_pixels']
        )
        logging.info(f"[Worker {process_id}] Tile {title_id} processed and saved.")

    # Clean up temporary folder after processing all tiles for this box
    if os.path.exists(worker_tif_folder):
        shutil.rmtree(worker_tif_folder)
    logging.info(f"[Worker {process_id}] Finished processing box {box_idx}.")

def main():
    """
    Main function to generate training data for Portsmouth dataset.
    It processes Sentinel-2 images and OSM data for specified regions.
    """

    # Authenticate Google Earth Engine
    # This handles the interactive part if needed and sets up credentials for child processes.
    authenticate_earth_engine(EE_PROJECT_ID)

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Import box from config
    start_lat = config_data['start_lat']
    end_lat = config_data['end_lat']
    start_lon = config_data['start_lon']
    end_lon = config_data['end_lon']
    scale = config_data['scale']
    pixels = config_data['pixels']

    # Estimate Image Size
    estimated_bytes, width_pixels, height_pixels = estimate_image_bytes(start_lat, start_lon, end_lat, end_lon, scale, bands=3)
    logging.info("Estimated total image size: %.2f MB", estimated_bytes / (1024 * 1024))

    boxes = split_bbox_if_needed(
        start_lat=start_lat,
        end_lat=end_lat,
        start_lon=start_lon,
        end_lon=end_lon,
        scale=scale,
    )
    logging.info(f"Split into {len(boxes)} bounding boxes for processing.")

    # Prepare arguments for the worker function
    tasks = [(i, box_coords, config_data) for i, box_coords in enumerate(boxes)]

    # Use a multiprocessing Pool to parallelize the processing of boxes
    num_processes = os.cpu_count() if os.cpu_count() else 4 # Default to 4 if cannot determine
    logging.info(f"Starting multiprocessing pool with {num_processes} processes.")
    with Pool(processes=num_processes) as pool:
        pool.map(worker_process_box, tasks)
    logging.info("All bounding boxes processed.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Time taken: %.2f seconds", elapsed_time)