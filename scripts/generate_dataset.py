# scripts/generate_dataset.py
import logging
import json
import os
import shutil
import time
import argparse  # Import argparse for command-line arguments
from multiprocessing import Pool, current_process

# Correct the import paths based on the new 'src' directory structure
from road_det_data_gen.data_prep.satellite_io import (
    authenticate_earth_engine,
    download_satellite_image,
    process_geotiff_image,
    get_image_resolution,
    slice_geotiff_image,
    split_bbox_if_needed,
    estimate_image_bytes
)
from road_det_data_gen.data_prep.osm_utils import process_osm_graph

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

def worker_process_box(args):
    """
    Worker function to process a single bounding box.
    It now accepts a single 'args' tuple to be compatible with Pool.map.
    """
    box_idx, box_coords, config, dataset_folder, ee_project_id = args
    slat, elat, slon, elon = box_coords

    # Initialize Earth Engine for this worker process. This is crucial for multiprocessing.
    authenticate_earth_engine(ee_project_id)

    # --- Setup unique temporary directory for this worker ---
    process_id = current_process().pid
    worker_tif_folder = os.path.join(dataset_folder, f"tmp_worker_{process_id}_{box_idx}")
    os.makedirs(worker_tif_folder, exist_ok=True)
    raw_geotiff_path = os.path.join(worker_tif_folder, "source.tif")

    # Extract satellite configuration from the main config
    satellite_config = config['satellite_config']
    satellite_type = satellite_config['type']

    logging.info(f"Processing box {box_idx} with {satellite_type} data...")

    # --- Download Satellite Image ---
    try:
        download_satellite_image(
            lat_st=slat, lon_st=slon, lat_ed=elat, lon_ed=elon,
            scale=config['scale'],
            output_path=raw_geotiff_path,
            satellite_config=satellite_config  # Pass the entire satellite config dict
        )
    except Exception as e:
        logging.error(f"Error downloading {satellite_type} image for box {box_idx}: {e}")
        if os.path.exists(worker_tif_folder):
            shutil.rmtree(worker_tif_folder)
        return

    # --- Slice the downloaded GeoTIFF into smaller tiles ---
    tile_coords = slice_geotiff_image(
        raw_geotiff_path,
        worker_tif_folder,
        tile_size=(config['pixels'], config['pixels']),
        overlap=config.get('overlap_pixels', 0)
    )

    # --- Process each tile ---
    for i, key in enumerate(tile_coords):
        tile_id_str = f"box{box_idx:03d}_tile{i:03d}"
        tile_tif_path = os.path.join(worker_tif_folder, f"tile_{key[0]}_{key[1]}.tif")
        output_sat_image_path = os.path.join(dataset_folder, f"{tile_id_str}_sat.png")
        output_prefix = os.path.join(dataset_folder, tile_id_str)

        logging.info(f"Processing tile {tile_id_str}...")

        # Process the raw tile into a normalized PNG for visualization and get its final geo-bounds
        tile_geo_bounds = process_geotiff_image(
            tile_tif_path,
            output_sat_image_path,
            satellite_config=satellite_config, # Pass config to handle different band processing
            size=(config['pixels'], config['pixels'])
        )

        if tile_geo_bounds is None:
            logging.warning(f"Tile {tile_id_str} has too many black pixels, skipping.")
            continue

        # Now process the corresponding OSM data for this tile
        process_osm_graph(
            data_geo_bounds=tile_coords[key]['geo_bounds'],
            tile_geo_bounds=tile_geo_bounds,
            output_prefix=output_prefix,
            img_pixel_width=config['pixels'],
            img_pixel_height=config['pixels']
        )
        logging.info(f"Tile {tile_id_str} processed and saved.")

    # --- Cleanup ---
    if os.path.exists(worker_tif_folder):
        shutil.rmtree(worker_tif_folder)
    logging.info(f"Finished processing box {box_idx}.")


def main():
    """
    Main driver script to generate a training dataset.
    Parses command-line arguments to select a config file and orchestrates the data generation.
    """
    parser = argparse.ArgumentParser(description="Generate satellite and OpenStreetMap training data.")
    parser.add_argument("--config", type=str, required=True, help="Path to the area configuration JSON file.")
    parser.add_argument("--project", type=str, required=True, help="Google Earth Engine Project ID.")
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = json.load(f)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    dataset_folder = f"data/training/{config_name}"
    os.makedirs(dataset_folder, exist_ok=True)

    # Authenticate GEE in the main process
    authenticate_earth_engine(args.project)

    # --- Bounding Box Splitting ---
    # Split the main bounding box into smaller chunks if it's too large for a single GEE request
    boxes = split_bbox_if_needed(
        start_lat=config['start_lat'],
        end_lat=config['end_lat'],
        start_lon=config['start_lon'],
        end_lon=config['end_lon'],
        scale=config['scale'],
    )
    logging.info(f"Processing area '{config_name}'. Split into {len(boxes)} bounding boxes.")

    # --- Prepare arguments for the worker pool ---
    tasks = [(i, box_coords, config, dataset_folder, args.project) for i, box_coords in enumerate(boxes)]

    # --- Run Multiprocessing ---
    num_processes = os.cpu_count() or 4
    logging.info(f"Starting multiprocessing pool with {num_processes} processes.")
    with Pool(processes=num_processes) as pool:
        pool.map(worker_process_box, tasks)

    logging.info("All bounding boxes processed.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"Total script time: {end_time - start_time:.2f} seconds")